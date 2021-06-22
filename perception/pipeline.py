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

class InferenceComponent:
    name = "inference"

    def __init__(self, model, cuda):
        self.cuda = cuda
        model = torch.jit.load(model)
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model.cpu().float()

    def __call__(self, left_frames, right_frames):
        N = left_frames.shape[0]
        if self.cuda:
            frames = torch.cat([left_frames, right_frames], 0).cuda()
        else:
            frames = torch.cat([left_frames, right_frames], 0)
        heatmaps, centers = [t.cpu() for t in self.model(frames)]
        left = heatmaps[:N], centers[:N]
        right = heatmaps[N:], centers[N:]
        return left, right

class KeypointExtractionComponent:
    name = "keypoints"
    PROBABILITY_CUTOFF = 0.25

    def __init__(self, keypoint_config, prediction_size, bandwidth=1.0):
        # Add center point.
        self.keypoint_config = [1] + keypoint_config['keypoint_config']
        self.n_keypoints = sum(self.keypoint_config)
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
            indices = self.image_indices[surpressed[0, 0] > 1.0]
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

class AssociationComponent:
    def __init__(self):
        self.stereo_camera = None

    def reset(self, stereo_camera):
        self.stereo_camera = stereo_camera
        default_point = np.array([0.0, 0.0, 0.6])[None]
        self.cutoff = stereo_camera.left_camera.image_size[0] / 2.0
        self.diff_LR = (self.stereo_camera.left_camera.project(default_point) -
                self.stereo_camera.right_camera.project(default_point, self.stereo_camera.T_RL))

    def __call__(self, left_points, right_points):
        """
        returns: integer array of length equal to the amount of left keypoints.
        Each index corresponds to the index of the right keypoint the left keypoint corresponds to.
        If a correspondence can't be found, -1 is set for that left keypoint.
        """
        ones_left = np.ones((left_points.shape[0], 1))
        ones_right = np.ones((right_points.shape[0], 1))
        left_keypoints = np.concatenate([left_points, ones_left], axis=1)
        right_keypoints = np.concatenate([right_points, ones_right], axis=1)
        distances = np.zeros((left_keypoints.shape[0], right_keypoints.shape[0]))
        for i, left in enumerate(left_keypoints):
            for j, right in enumerate(right_keypoints):
                distances[i, j] = np.abs(right[None, :] @ self.stereo_camera.F @ left[:, None])

        correspondences = np.zeros((left_keypoints.shape[0],), dtype=np.int32)
        for i in range(left_keypoints.shape[0]):
            within_bounds  = distances[i, :] < self.cutoff
            possible_matches = within_bounds.sum()
            if possible_matches == 1:
                correspondences[i] = distances[i, :].argmin()
            elif possible_matches > 1:
                # If there are several matches along the epipolar line,
                # assume a constant depth, and figure out which point
                # is closer.
                left = left_points[i]
                indices = np.argwhere(within_bounds).ravel()
                right = right_points[indices, :] + self.diff_LR
                diffs = np.linalg.norm(left - right, axis=1)
                correspondences[i] = indices[diffs.argmin()]
            else:
                correspondences[i] = -1
        return correspondences

class TriangulationComponent:
    name = "triangulation"

    def __init__(self):
        self.stereo_camera = None

    def reset(self, stereo_camera):
        self.stereo_camera = stereo_camera

    def __call__(self, left_keypoints, right_keypoints):
        return self.stereo_camera.triangulate(left_keypoints, right_keypoints)

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

    def __call__(self, xy, p_depth):
        if xy.shape[0] == 0:
            return None
        xy_int = xy.round().astype(np.int)
        zs = p_depth[xy_int[:, 1], xy_int[:, 0]]
        return self.camera.unproject(xy, zs)

class ObjectKeypointPipeline:
    def __init__(self, prediction_size, points_3d, keypoint_config):
        self.keypoint_extraction = KeypointExtractionComponent(keypoint_config, prediction_size)
        self.object_extraction = ObjectExtraction(keypoint_config, prediction_size)
        self.detection_to_point = DetectionToPoint()

    def reset(self, stereo_camera):
        self.detection_to_point.reset(stereo_camera.left_camera)

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
            objects.append({
                'p_centers': obj['p_centers'],
                'keypoints_left': [obj['center'][None]] + obj['heatmap_points'],
                'p_L': world_points
            })
        return objects

class LearnedKeypointTrackingPipeline(ObjectKeypointPipeline):
    def __init__(self, model, cuda=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference = InferenceComponent(model, cuda)

    def __call__(self, left_frame, right_frame):
        left, right = self.inference(left_frame, right_frame)
        l_hm, l_c = left
        r_hm, r_c = right
        return super().__call__(l_hm, l_c, r_hm, r_c), (l_hm, r_hm)

