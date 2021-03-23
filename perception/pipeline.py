import numpy as np
import torch
import cv2
from perception.utils import camera_utils, clustering_utils, linalg


class InferenceComponent:
    name = "inference"

    def __init__(self, model):
        self.model = torch.jit.load(model).half()

    def __call__(self, left_frames, right_frames):
        N = left_frames.shape[0]
        frames = torch.cat([left_frames, right_frames], 0).half()
        predictions = self.model(frames).cpu().numpy()
        left_pred = predictions[:N]
        right_pred = predictions[N:]
        return left_pred, right_pred


class KeypointExtractionComponent:
    name = "keypoints"
    PROBABILITY_CUTOFF = 0.1

    def __init__(self, K):
        self.K = K
        self.reset()
        prediction_size = 90, 160
        self.indices = np.zeros((*prediction_size, 2), dtype=np.float32)
        for y in range(prediction_size[0]):
            for x in range(prediction_size[1]):
                self.indices[y, x, 0] = np.float32(x) + 1.0
                self.indices[y, x, 1] = np.float32(y) + 1.0

    def reset(self):
        self.clustering_left = clustering_utils.KeypointClustering(self.K, 2.0)
        self.clustering_right = clustering_utils.KeypointClustering(self.K, 2.0)

    def _to_probabilities(self, prediction):
        return (prediction + 1.0) * 0.5

    def _extract_keypoints(self, heatmap, clustering):
        keypoints = np.zeros((1 + self.K, 2), dtype=heatmap.dtype)
        # Center point
        center_point = heatmap[0]
        where_larger = center_point > self.PROBABILITY_CUTOFF
        left_center = self.indices[where_larger, :]
        probabilities_center = center_point[where_larger]
        probabilities_center /= probabilities_center.sum()
        keypoints[0, :] = (left_center * probabilities_center[:, None]).sum(axis=0)

        # Three spoke points.
        spokes = heatmap[1]
        where_larger = spokes > self.PROBABILITY_CUTOFF
        spoke_indices = self.indices[where_larger, :]
        probabilities_spoke = spokes[where_larger]

        clusters = clustering(spoke_indices, probabilities_spoke)

        keypoints[1 : 1 + clusters.shape[0], :] = clusters
        return keypoints

    def __call__(self, left, right):
        left = self._to_probabilities(left)
        right = self._to_probabilities(right)
        N = left.shape[0]

        keypoints = np.zeros((2, N, 1 + self.K, 2), dtype=left.dtype)
        for i in range(left.shape[0]):
            keypoints[0, i] = self._extract_keypoints(left[i], self.clustering_left)
            keypoints[1, i] = self._extract_keypoints(right[i], self.clustering_right)

        return keypoints[0], keypoints[1]


class TriangulationComponent:
    name = "association"

    def __init__(self, n_points):
        self.n_points = n_points
        self.K = None
        self.Kp = None
        self.T_RL = None
        self.scaling_factor = None
        self.F = None

    def reset(self, K, Kp, T_RL, scaling_factor):
        self.K = K
        self.Kp = Kp
        self.T_RL = T_RL
        R = self.T_RL[:3, :3]
        t = self.T_RL[:3, 3]
        Kp_inv = np.linalg.inv(self.Kp)

        C = linalg.skew_matrix(self.K @ R.T @ t)
        self.F = Kp_inv.T @ R @ self.K.T @ C
        self.scaling_factor = scaling_factor

    def _associate(self, left_keypoints, right_keypoints):
        distances = np.zeros(
            (left_keypoints.shape[0], right_keypoints.shape[0]),
            dtype=left_keypoints.dtype,
        )
        one = np.ones((1,), dtype=left_keypoints.dtype)
        for i in range(left_keypoints.shape[0]):
            v1 = np.concatenate([left_keypoints[i, :], one], axis=0)[:, None]
            for j in range(right_keypoints.shape[0]):
                v2 = np.concatenate([right_keypoints[j, :], one], axis=0)[:, None]
                distances[i, j] = np.abs(v2.T @ self.F @ v1)
        return distances.argmin(axis=1)

    def _scale_points(self, points):
        # points: N x 2 image points on the prediction heatmap.
        # returns: N x 2 image points scaled to the full image size.
        return points * self.scaling_factor

    def _triangulate(self, left_keypoints, T_LW, right_keypoints):
        left_keypoints = self._scale_points(left_keypoints)
        right_keypoints = self._scale_points(right_keypoints)
        associations = self._associate(left_keypoints, right_keypoints)
        # Permute keypoints into the same order as left.
        right_keypoints = right_keypoints[associations]

        P1 = camera_utils.projection_matrix(self.K, np.eye(4))
        P2 = self.Kp @ np.eye(3, 4) @ self.T_RL
        p_LK = cv2.triangulatePoints(
            P1, P2, left_keypoints.T, right_keypoints.T
        ).T  # N x 4
        p_LK = p_LK / p_LK[:, 3:4]
        p_WK = (np.linalg.inv(T_LW) @ p_LK[:, :, None])[:, :, 0]
        p_WK /= p_WK[:, 3:4]
        return p_WK

    def __call__(self, left_keypoints, T_LW, right_keypoints, T_RW):
        N = left_keypoints.shape[0]
        out_points = np.zeros((N, 1+self.n_points, 4), dtype=np.float32)
        for i in range(left_keypoints.shape[0]):
            out_points[i, :, :] = self._triangulate(
                left_keypoints[i], T_LW[i], right_keypoints[i]
            )
        return out_points


class KeypointPipeline:
    def __init__(self, model, n_keypoints):
        self.inference = InferenceComponent(model)
        self.keypoint_extraction = KeypointExtractionComponent(n_keypoints)
        self.triangulation = TriangulationComponent(n_keypoints)

    def reset(self, K, Kp, T_RL, scaling_factor):
        self.keypoint_extraction.reset()
        self.triangulation.reset(K, Kp, T_RL, scaling_factor)

    def __call__(self, left, T_LW, right, T_RW):
        heatmap_l, heatmap_r = self.inference(left, right)
        keypoints_l, keypoints_r = self.keypoint_extraction(heatmap_l, heatmap_r)
        out_points = self.triangulation(keypoints_l, T_LW, keypoints_r, T_RW)
        out = {
            "heatmap_left": heatmap_l,
            "heatmap_right": heatmap_r,
            "keypoints_world": out_points,
        }
        return out
