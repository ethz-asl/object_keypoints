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
    PROBABILITY_CUTOFF = 0.1

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

    def __call__(self, left, right):
        N = left.shape[0]

        keypoints_left = []
        confidence_left = []
        keypoints_right = []
        confidence_right = []
        for i in range(left.shape[0]):
            kp, c = self._extract_keypoints(left[i])
            keypoints_left.append(kp)
            confidence_left.append(c)
            kp, c = self._extract_keypoints(right[i])
            keypoints_right.append(kp)
            confidence_right.append(c)

        return (keypoints_left, confidence_left), (keypoints_right, confidence_right)

def _get_depth(xy, depth):
    indices = np.argwhere(depth > 0.1)
    yx = xy[:, ::-1]
    distances = np.linalg.norm(indices - yx[:, None], axis=2)
    depth_index = indices[distances.argmin(axis=1)]
    return depth[depth_index[:, 0], depth_index[:, 1]]

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

class PoseSolveComponent:
    def __init__(self, points_3d):
        self.points_3d = np.zeros((points_3d.shape[0] + 1, 3))
        self.points_3d[0] = points_3d.mean(axis=0)
        self.points_3d[1:] = points_3d

    def __call__(self, keypoints):
        keypoints = keypoints[:, :, :3]
        T = np.zeros((keypoints.shape[0], 4, 4))
        for i in range(keypoints.shape[0]):
            object_keypoints = keypoints[i]
            p_center = object_keypoints.mean(axis=0)
            gt_center = self.points_3d.mean(axis=0)
            translation = p_center - gt_center
            T[i, :3, 3] = translation
            # Sort object keypoints along z-axis, this should make the detections
            # more consistent across time steps.
            object_keypoints = sorted(object_keypoints, key=lambda v: v[2])
            to_align = (object_keypoints - translation)
            rotation, _ = Rotation.align_vectors(to_align, self.points_3d)
            T[i, :3, :3] = rotation.as_matrix()
        return T

class ObjectKeypointPipeline:
    def __init__(self, prediction_size, points_3d, keypoint_config):
        self.keypoint_extraction = KeypointExtractionComponent(keypoint_config, prediction_size)
        self.object_extraction = ObjectExtraction(keypoint_config, prediction_size)
        self.association = AssociationComponent()
        self.triangulation = TriangulationComponent()
        self.K = self.Kp = self.D = self.Dp = self.T_RL = None

    def reset(self, stereo_camera):
        self.association.reset(stereo_camera)
        self.triangulation.reset(stereo_camera)

    def __call__(self, heatmap_left, centers_left, heatmap_right, centers_right):
        assert heatmap_left.shape[0] == 1, "One at the time, please."
        heatmap_left = heatmap_left.numpy()
        heatmap_right = heatmap_right.numpy()
        centers_left = centers_left[0].numpy()
        centers_right = centers_right[0].numpy()
        (left_points, left_c), (right_points, right_c) = self.keypoint_extraction(heatmap_left, heatmap_right)
        left_points, right_points = left_points[0], right_points[0]
        left_c, right_c = left_c[0], right_c[0]
        objects_left = self.object_extraction(left_points, left_c, centers_left)
        objects_right = self.object_extraction(right_points, right_c, centers_right)
        objects = []
        for object_left, object_right in zip(objects_left, objects_right):
            for i, (left, right) in enumerate(zip(object_left['heatmap_points'], object_right['heatmap_points'])):
                if left.shape[0] > 0 and right.shape[0] > 0:
                    associations = self.association(left, right)
                    associations = np.unique(associations[associations >= 0])
                    N_points = associations.shape[0]
                    object_right['heatmap_points'][i] = right[associations]
                    object_left['heatmap_points'][i] = left[:N_points]

                # if len(left) > len(right):
                #     # Some keypoints were not appropriately detected in both frames.
                #     import ipdb; ipdb.set_trace()
                #     object_right['heatmap_points'][i] = self._recover_from_left(depth_left[1 + i], left, right)
                # elif len(right) > len(left):
                #     import ipdb; ipdb.set_trace()
                #     object_left['heatmap_points'][i] = self._recover_from_right(depth_right[1 + i], left, right)

            world_points = [self.triangulation(object_left['center'][None], object_right['center'][None])]
            for i in range(len(object_left['heatmap_points'])):
                # if len(object_left['heatmap_points'][i]) == 0:
                #     continue
                if len(object_left['heatmap_points'][i]) != 0 and len(object_right['heatmap_points'][i]) != 0:
                    if len(object_left['heatmap_points'][i]) != len(object_right['heatmap_points'][i]):
                        import ipdb; ipdb.set_trace()
                    p_L = self.triangulation(object_left['heatmap_points'][i], object_right['heatmap_points'][i])
                    world_points.append(p_L)
                else:
                    world_points.append(None)
            objects.append({
                'centers_left': object_left['p_centers'],
                'centers_right': object_right['p_centers'],
                'keypoints_left': [object_left['center'][None]] + object_left['heatmap_points'],
                'keypoints_right': [object_right['center'][None]] + object_right['heatmap_points'],
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

