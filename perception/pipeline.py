import numpy as np
import torch
import cv2
from sklearn import cluster
from perception.utils import camera_utils, clustering_utils, linalg
from scipy.spatial.transform import Rotation
from sklearn import metrics
import skimage
from scipy import ndimage

class InferenceComponent:
    name = "inference"

    def __init__(self, model, cuda):
        self.cuda = cuda
        model = torch.jit.load(model)
        if cuda:
            self.model = model.half().cuda()
        else:
            self.model = model.cpu().float()

    def __call__(self, left_frames, right_frames):
        N = left_frames.shape[0]
        if self.cuda:
            frames = torch.cat([left_frames, right_frames], 0).half().cuda()
        else:
            frames = torch.cat([left_frames, right_frames], 0)
        heatmaps, depth, centers = [t.cpu() for t in self.model(frames)]
        left = heatmaps[:N], depth[:N], centers[:N]
        right = heatmaps[N:], depth[N:], centers[N:]

        return left, right

class KeypointExtractionComponent:
    name = "keypoints"
    PROBABILITY_CUTOFF = 0.2

    def __init__(self, keypoint_config, prediction_size):
        # Add center point.
        self.keypoint_config = [1] + keypoint_config['keypoint_config']
        self.n_keypoints = sum(self.keypoint_config)
        prediction_size = prediction_size
        self.clustering = clustering_utils.KeypointClustering(bandwidth=5.0)

    def _cluster(self, heatmap):
        indices = np.argwhere(heatmap > self.PROBABILITY_CUTOFF)
        if indices.size == 0:
            return []
        out = [x for x in self.clustering(indices)[:, ::-1] + 0.5]
        return out

    def _extract_keypoints(self, heatmap):
        out_points = []
        assert heatmap.shape[0] == len(self.keypoint_config)
        for i, n_keypoints in enumerate(self.keypoint_config):
            points = self._cluster(heatmap[i])
            out_points.append(points)
        return out_points

    @staticmethod
    def _preprocess_heatmap(heatmap):
        out = heatmap.copy().astype(np.float32)
        out[heatmap > 0.75] = 1.0
        return out * 2.0

    def _points(self, heatmap):
        out_points = []
        heatmap_processed = self._preprocess_heatmap(heatmap)

        for i, n_keypoints in enumerate(self.keypoint_config):
            image_max = ndimage.maximum_filter(heatmap_processed[i], size=1, mode='constant')
            points = skimage.feature.peak_local_max(image_max, min_distance=5,
                    threshold_abs=self.PROBABILITY_CUTOFF * 2.0, p_norm=1)
            out_points.append(points[:, ::-1] + 1.0)
        return out_points

    def __call__(self, left, right):
        N = left.shape[0]

        keypoints_left = []
        keypoints_right = []
        for i in range(left.shape[0]):
            # keypoints_left.append(self._points(left[i]))
            # keypoints_right.append(self._points(right[i]))
            keypoints_left.append(self._extract_keypoints(left[i]))
            keypoints_right.append(self._extract_keypoints(right[i]))

        return keypoints_left, keypoints_right

def _get_depth(xy, depth):
    indices = np.argwhere(depth > 0.1)
    yx = xy[:, ::-1]
    distances = np.linalg.norm(indices - yx[:, None], axis=2)
    depth_index = indices[distances.argmin(axis=1)]
    return depth[depth_index[:, 0], depth_index[:, 1]]

def project_3d(points, depth, K):
    xys = points.round().astype(np.int32)
    zs = _get_depth(xys, depth[1:])
    p_W = np.zeros((points.shape[0], 3))
    p_W[:, 0] = (points[:, 0] - K[0, 2]) * zs / K[0, 0]
    p_W[:, 1] = (points[:, 1] - K[1, 2]) * zs / K[1, 1]
    p_W[:, 2] = zs
    return p_W

class AssociationComponent:
    def __init__(self):
        self.K = None
        self.D = None
        self.Kp = None
        self.Dp = None
        self.T_RL = None
        self.F = None

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.K = camera_utils.scale_camera_matrix(K, 1.0 / scaling_factor)
        self.Kp = camera_utils.scale_camera_matrix(Kp, 1.0 / scaling_factor)
        self.D = D
        self.Dp = Dp
        self.T_RL = T_RL
        self.Kinv = np.linalg.inv(K)
        self.Kpinv = np.linalg.inv(Kp)
        R = T_RL[:3, :3]
        t = T_RL[:3, 3]

        C = linalg.skew_matrix(self.K @ R.T @ t)
        self.F = np.linalg.inv(self.Kp.T) @ R @ self.K.T @ C
        self.scaling_factor = scaling_factor

    def __call__(self, left_keypoints, left_depth, right_keypoints, right_depth):
        """
        returns: integer array of length equal to the amount of left keypoints.
        Each index corresponds to the index of the right keypoint the left keypoint corresponds to.
        If a correspondence can't be found, -1 is set for that left keypoint.
        """
        left_keypoints = cv2.fisheye.undistortPoints(left_keypoints[:, None, :], self.K, self.D, P=self.K).reshape(-1, 2)
        right_keypoints = cv2.fisheye.undistortPoints(right_keypoints[:, None, :], self.Kp, self.Dp, P=self.Kp).reshape(-1, 2)
        ones_left = np.ones((left_keypoints.shape[0], 1))
        ones_right = np.ones((right_keypoints.shape[0], 1))
        left_keypoints = np.concatenate([left_keypoints, ones_left], axis=1)
        right_keypoints = np.concatenate([right_keypoints, ones_right], axis=1)
        distances = np.zeros((left_keypoints.shape[0], right_keypoints.shape[0]))
        for i, left in enumerate(left_keypoints):
            for j, right in enumerate(right_keypoints):
                distances[i, j] = np.abs(right[None, :] @ self.F @ left[:, None])

        correspondences = np.zeros((left_keypoints.shape[0],), dtype=np.int32)
        for i in range(left_keypoints.shape[0]):
            if (distances[i, :] < 50.0).any():
                correspondences[i] = distances[i, :].argmin()
            else:
                correspondences[i] = -1
        return correspondences

class TriangulationComponent:
    name = "triangulation"

    def __init__(self):
        self.K = None
        self.Kp = None
        self.D = None
        self.Dp = None
        self.T_RL = None

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.K = camera_utils.scale_camera_matrix(K, 1.0 / scaling_factor)
        self.Kp = camera_utils.scale_camera_matrix(Kp, 1.0 / scaling_factor)
        self.D = D
        self.Dp = Dp
        self.T_RL = T_RL

    def __call__(self, left_keypoints, right_keypoints):
        left_keypoints = left_keypoints[:, None, :].astype(np.float32)
        right_keypoints = right_keypoints[:, None, :].astype(np.float32)
        left_keypoints = cv2.fisheye.undistortPoints(left_keypoints, self.K, self.D, P=self.K)[:, 0, :]
        right_keypoints = cv2.fisheye.undistortPoints(right_keypoints, self.Kp, self.Dp, P=self.Kp)[:, 0, :]
        P1 = self.K @ np.eye(3, 4, dtype=np.float32)
        P2 = self.Kp @ self.T_RL[:3].astype(np.float32)
        p_LK = cv2.triangulatePoints(
            P1, P2, left_keypoints.T, right_keypoints.T
        ).T  # N x 4

        p_LK = p_LK / p_LK[:, 3:4]
        return p_LK[:, :3]

class ObjectExtraction:
    def __init__(self, keypoint_config):
        self.keypoint_config = keypoint_config
        self.prediction_size = [180, 320]
        self.max = np.array(self.prediction_size[::-1], dtype=np.int32) - 1
        self.min = np.zeros(2, dtype=np.int32)
        self.image_indices = np.zeros((2, *self.prediction_size))
        for i in range(self.prediction_size[0]):
            for j in range(self.prediction_size[1]):
                self.image_indices[:, i, j] = (j + 0.5, i + 0.5)

    def __call__(self, keypoints, centers):
        p_centers = self.image_indices + centers
        objects = []
        center_points = np.stack(keypoints[0])
        for center in center_points:
            obj = { 'center': center, 'heatmap_points': [], 'p_centers': [] }
            obj['heatmap_points'] = [[] for _ in range((len(keypoints) - 1))]
            objects.append(obj)
        for i, points in enumerate(keypoints[1:]):
            for point in points:
                xy = np.clip(point.round().astype(np.int32), self.min, self.max)
                predicted_center = centers[i, :, xy[1], xy[0]] + point
                distance_to_centers = np.linalg.norm(center_points - predicted_center[None], 2, axis=1)
                obj = objects[distance_to_centers.argmin(axis=0)]
                obj['p_centers'].append(predicted_center)
                obj['heatmap_points'][i].append(point)
        for obj in objects:
            for i in range(len(obj['heatmap_points'])):
                if len(obj['heatmap_points'][i]) > 0:
                    obj['heatmap_points'][i] = np.stack(obj['heatmap_points'][i])
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
        self.object_extraction = ObjectExtraction(keypoint_config)
        self.association = AssociationComponent()
        self.triangulation = TriangulationComponent()
        self.K = self.Kp = self.D = self.Dp = self.T_RL = None

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.K = camera_utils.scale_camera_matrix(K, 1.0 / scaling_factor)
        self.Kp = camera_utils.scale_camera_matrix(Kp, 1.0 / scaling_factor)
        self.D = D
        self.Dp = Dp
        self.T_RL = T_RL
        self.T_LR = np.linalg.inv(T_RL)
        self.association.reset(K, Kp, D, Dp, T_RL, scaling_factor)
        self.triangulation.reset(K, Kp, D, Dp, T_RL, scaling_factor)

    def __call__(self, heatmap_left, depth_left, centers_left, heatmap_right, depth_right, centers_right):
        assert heatmap_left.shape[0] == 1, "One at the time, please."
        depth_left = depth_left[0].numpy()
        depth_right = depth_right[0].numpy()
        heatmap_left = heatmap_left.numpy()
        heatmap_right = heatmap_right.numpy()
        left_points, right_points = self.keypoint_extraction(heatmap_left, heatmap_right)
        left_points, right_points = left_points[0], right_points[0]
        objects_left = self.object_extraction(left_points, centers_left[0].numpy())
        objects_right = self.object_extraction(right_points, centers_right[0].numpy())
        objects = []
        for object_left, object_right in zip(objects_left, objects_right):
            for i, (left, right) in enumerate(zip(object_left['heatmap_points'], object_right['heatmap_points'])):
                if left.shape[0] > 0 and right.shape[0] > 0:
                    associations = self.association(left, depth_left[i], right, depth_right[i])
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
                    p_W = self.triangulation(object_left['heatmap_points'][i], object_right['heatmap_points'][i])
                    world_points.append(p_W)
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
        l_hm, l_d, l_c = left
        r_hm, r_d, r_c = right
        return super().__call__(l_hm, l_d, l_c, r_hm, r_d, r_c)

