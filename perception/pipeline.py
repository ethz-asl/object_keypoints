import numpy as np
import torch
import cv2
from sklearn import cluster
from perception.utils import camera_utils, clustering_utils, linalg
from scipy.spatial.transform import Rotation
from perception.models import nms
from sklearn import metrics

from perception.datasets.video import _gaussian_kernel, default_length_scale
from numba import jit
import matplotlib.pyplot as plt

class InferenceComponent:
    name = "inference"

    def __init__(self, model, cuda):
        self.cuda = cuda
        model = torch.jit.load(model)
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model.cpu().float()

    def __call__(self, frames):
        if self.cuda:
            frames = frames.cuda()
        heatmaps, depth, centers = [t.cpu() for t in self.model(frames)]
        return heatmaps, depth, centers

class KeypointExtractionComponent:
    name = "keypoints"
    PROBABILITY_CUTOFF = 0.5

    def __init__(self, keypoint_config, prediction_size, bandwidth=1.0):
        # Add center point.
        self.keypoint_config = [1] + keypoint_config['keypoint_config']
        self.n_keypoints = sum(self.keypoint_config)
        print("[KeypointExtractionComponent]: {} keypoints are (include center)".format(self.keypoint_config))

        prediction_size = prediction_size
        self.kernel = torch.ones((1, 1, 5, 5), dtype=torch.float32)
        self.image_indices = torch.zeros((*prediction_size, 2), dtype=torch.int)
        for i in range(prediction_size[0]):
            for j in range(prediction_size[1]):
                self.image_indices[i, j, 0] = i
                self.image_indices[i, j, 1] = j

    def _compute_points(self, indices, probabilities):
        height, width, _ = self.image_indices.shape
        points = []
        confidences = []
        for index in indices:
            y, x = index
            y, x = y.item(), x.item()
            x_start = max(x - 2, 0)
            x_end = min(x + 3, width)
            y_start = max(y - 2, 0)
            y_end = min(y + 3, height)
            p = probabilities[y_start:y_end, x_start:x_end]
            local_indices = self.image_indices[y_start:y_end, x_start:x_end]
            index = ((p[:, :, None] * local_indices).sum(dim=[0, 1]) / p.sum()).numpy()
            points.append(index)
            confidences.append(p.sum())
        return points, confidences

    def _extract_keypoints(self, heatmap):
        out_points = []
        confidences = []
        assert heatmap.shape[0] == len(self.keypoint_config)
        for i, n_keypoints in enumerate(self.keypoint_config):
            probabilities = torch.tensor(heatmap[i].astype(np.float32))[None, None]
            weights_c = torch.nn.functional.conv2d(probabilities, self.kernel, bias=None, stride=1,
                    padding=2)
            surpressed = nms(weights_c)  
            indices = self.image_indices[surpressed[0, 0] > self.PROBABILITY_CUTOFF]
            points, confidence = self._compute_points(indices, probabilities[0, 0])
          
            points = [x[::-1] for x in points]
            out_points.append(points)
            confidences.append(confidence)
        return out_points, confidences

    def __call__(self, frames):
        N = frames.shape[0]

        keypoints = []
        confidence = []
        for i in range(frames.shape[0]):
            kp, c = self._extract_keypoints(frames[i])
            keypoints.append(kp)
            confidence.append(c)
        return keypoints, confidence

class ObjectExtraction:
    def __init__(self, keypoint_config, prediction_size):
        self.keypoint_config = keypoint_config['keypoint_config']
        self.prediction_size = prediction_size
        self.max = np.array(self.prediction_size[::-1], dtype=np.int32) - 1
        self.min = np.zeros(2, dtype=np.int32)
        self.image_indices = np.zeros((2, *self.prediction_size))
        for i in range(self.prediction_size[0]):
            for j in range(self.prediction_size[1]):
                self.image_indices[:, i, j] = (j + 0.5, i + 0.5)

    def __call__(self, keypoints, confidence, centers):
        if len(keypoints[0]) == 0:
            return []
        p_centers = self.image_indices + centers
        objects = []
        center_points = np.stack(keypoints[0])
        for center in center_points:
            obj = { 'center': center, 'heatmap_points': [], 'p_centers': [] }
            obj['heatmap_points'] = [[] for _ in range((len(keypoints) - 1))]
            obj['confidence'] = [[] for _ in range((len(keypoints) - 1))]
            objects.append(obj)
        for i, points in enumerate(keypoints[1:]):
            for j, point in enumerate(points):
                xy = np.clip(point.round().astype(np.int32), self.min, self.max)
                predicted_center = p_centers[i, :, xy[1], xy[0]]
                distance_to_centers = np.linalg.norm(center_points - predicted_center[None], 2, axis=1)
                distance = distance_to_centers.min()
                if distance > 20.0:
                    # Outlier, skip this point
                    print(f"skipping point with center distance: {distance}.")
                    continue
                obj = objects[distance_to_centers.argmin(axis=0)]
                obj['p_centers'].append(predicted_center)
                obj['heatmap_points'][i].append(point)
                obj['confidence'][i].append(confidence[i+1][j])

        for obj in objects:
            for i in range(len(obj['heatmap_points'])):
                if len(obj['heatmap_points'][i]) > 0:
                    points = np.stack(obj['heatmap_points'][i])
                    confidences = np.stack(obj['confidence'][i])
                    if points.shape[0] > self.keypoint_config[i]:
                        # This object has more keypoint detections for keypoint type i
                        # than it is supposed to have. Likely a double detection of false
                        # positive.
                        if self.keypoint_config[i] == 1:
                            # Should have been only one keypoint of this type, pick
                            # the one that has higher confidence.
                            points = points[confidences.argmax(axis=0)][None]
                        else:
                            # There is several points of this type.
                            # should probably cluster.
                            clusterer = cluster.KMeans(init='random', n_clusters=self.keypoint_config[i])
                            clusterer.fit(points)
                            points = clusterer.cluster_centers_
                    obj['heatmap_points'][i] = points
                else:
                    # Keypoint was associated with the wrong object.
                    obj['heatmap_points'][i] = np.array([])
        return objects

class DetectionToPoint:
    def __init__(self):
        pass

    def reset(self, camera):
        self.camera = camera
        self.min_index = np.zeros(2, dtype=np.int)
        self.max_index = camera.image_size.astype(np.int) - 1

    def __call__(self, xy, p_depth):
        if xy.shape[0] == 0:
            return None
        xy = self.camera.undistort(xy)
        xy_int = xy.round().astype(np.int)
        xy_int = np.clip(xy_int, self.min_index, self.max_index)
        zs = p_depth[xy_int[:, 1], xy_int[:, 0]]
        return self.camera.unproject(xy, zs)

class ObjectKeypointPipeline:
    def __init__(self, prediction_size, keypoint_config):
        self.keypoint_extraction = KeypointExtractionComponent(keypoint_config, prediction_size)
        self.object_extraction = ObjectExtraction(keypoint_config, prediction_size)
        self.detection_to_point = DetectionToPoint()

    def reset(self, camera):
        self.detection_to_point.reset(camera)

    def __call__(self, heatmap, p_depth, p_centers):
        assert heatmap.shape[0] == 1, "One at the time, please."
        heatmap = heatmap.numpy()
        p_centers = p_centers[0].numpy()
        p_depth = p_depth[0].numpy()
        points, confidence = self.keypoint_extraction(heatmap)        
        detected_objects = self.object_extraction(points[0], confidence[0], p_centers)

        objects = []
        for obj in detected_objects:
            world_points = [self.detection_to_point(obj['center'][None], p_depth[0])]
            for i in range(len(obj['heatmap_points'])):
                point = self.detection_to_point(obj['heatmap_points'][i], p_depth[1 + i])
                world_points.append(point)
            new_object = {'p_centers': obj['p_centers'],
                          'keypoints': [obj['center'][None]] + obj['heatmap_points'],
                          'p_C': world_points}   
            objects.append(new_object)
        return objects

class LearnedKeypointTrackingPipeline(ObjectKeypointPipeline):
    def __init__(self, model, cuda=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference = InferenceComponent(model, cuda)

    def __call__(self, frame):
        heatmap, depth, centers = self.inference(frame)
        return super().__call__(heatmap, depth, centers), heatmap

