import unittest
import numpy as np
import cv2
import sklearn
from perception.utils import camera_utils
from perception.datasets.video import StereoVideoDataset
from perception.pipeline import *

keypoints_distinct = np.array([
    [0.0, 0.0, 1.1],
    [0.1, 0.0, 1.0],
    [-0.1, 0.0, 1.0]])
keypoints_two_kinds = np.array([
    [0.0, 0.0, 1.0],
    [0.25, 0.15, 1.0],
    [-0.25, -0.25, 1.0],
    [0.25, -0.25, 1.0]])


config_distinct = {
    'keypoint_config': [1, 1, 1]
}
config_two_kinds = {
    'keypoint_config': [1, 3]
}
scaling_factor = 180 / StereoVideoDataset.height
points_left_distinct = np.array([[641.00771598, 368.16440843],
    [641.00771598, 368.16440843],
    [710.73402561, 368.16440843],
    [571.28140636, 368.16440843]]) * scaling_factor
points_right_distinct = np.array([[680.43097119, 368.90194905],
    [677.89404795, 368.8918863],
    [750.85353605, 368.91588343],
    [612.46064648, 368.88886675]]) * scaling_factor

def compute_heatmaps(keypoints, keypoint_config, T_LW, T_RW, K, D, Kp, Dp):
    config = [1] + keypoint_config['keypoint_config']
    heatmap_left = np.zeros((len(config), StereoVideoDataset.height, StereoVideoDataset.width))
    heatmap_right = np.zeros_like(heatmap_left)
    R_L, _ = cv2.Rodrigues(T_LW[:3, :3])
    R_R, _ = cv2.Rodrigues(T_RW[:3, :3])
    p_L, _ = cv2.fisheye.projectPoints(keypoints[:, None, :], R_L, T_LW[:3, 3], K, D)
    p_L = p_L[:, 0, :]
    p_R, _ = cv2.fisheye.projectPoints(keypoints[:, None, :], R_R, T_RW[:3, 3], Kp, Dp)
    p_R = p_R[:, 0, :]
    current_point = 0
    keypoints_per_object = sum(config)
    keypoint_sets = keypoints.shape[0] // keypoints_per_object
    for k in range(keypoint_sets):
        for map_index, n_keypoints in enumerate(config):
            for i in range(n_keypoints):
                StereoVideoDataset._add_kernel(heatmap_left[map_index], p_L[current_point][None])
                StereoVideoDataset._add_kernel(heatmap_right[map_index], p_R[current_point][None])
                current_point += 1

    heatmap_left /= heatmap_left.max()
    heatmap_right /= heatmap_right.max()
    return heatmap_left, heatmap_right, p_L, p_R

class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params = camera_utils.load_calibration_params('./config/calibration.yaml')
        cls.K = params['K']
        cls.D = params['D']
        cls.Kp = params['Kp']
        cls.Dp = params['Dp']
        cls.T_LR = params['T_LR']
        cls.T_RL = np.linalg.inv(cls.T_LR)
        cls.keypoints_distinct = np.zeros((keypoints_distinct.shape[0] + 1, 3))
        cls.keypoints_distinct[0] = keypoints_distinct.mean(axis=0)
        cls.keypoints_distinct[1:] = keypoints_distinct
        cls.keypoints_two_kinds = np.zeros((keypoints_two_kinds.shape[0] + 1, 3))
        cls.keypoints_two_kinds[0] = keypoints_two_kinds.mean(axis=0)
        cls.keypoints_two_kinds[1:] = keypoints_two_kinds

    def test_extract_single_points(self):
        T_LW = np.eye(4)
        T_RW = self.T_RL @ T_LW
        heatmap_left, heatmap_right, p_L, p_R = compute_heatmaps(self.keypoints_distinct, config_distinct,
                T_LW, T_RW, self.K, self.D, self.Kp, self.Dp)
        prediction_left = np.zeros((4, 180, 320))
        prediction_right = np.zeros((4, 180, 320))

        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))

        extract_component = KeypointExtractionComponent(config_distinct, [180, 320])
        left_points, right_points = extract_component(prediction_left[None], prediction_right[None])
        for i in range(self.keypoints_distinct.shape[0]):
            p_L_hat = left_points[0][i][0]
            p_R_hat = right_points[0][i][0]
            # Check that error is less than a pixel.
            self.assertLess(np.linalg.norm(p_L_hat - p_L[i] * scaling_factor), 0.5)
            self.assertLess(np.linalg.norm(p_R_hat - p_R[i] * scaling_factor), 0.5)

    def test_extract_multiple(self):
        T_LW = np.eye(4)
        T_RW = np.linalg.inv(self.T_LR) @ T_LW
        heatmap_left, heatmap_right, p_L, p_R = compute_heatmaps(self.keypoints_two_kinds, config_two_kinds,
                T_LW, T_RW, self.K, self.D, self.Kp, self.Dp)
        prediction_left = np.zeros((heatmap_left.shape[0], 180, 320))
        prediction_right = np.zeros((heatmap_left.shape[0], 180, 320))

        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))

        extract_component = KeypointExtractionComponent(config_two_kinds, [180, 320])
        left_points, right_points = extract_component(prediction_left[None], prediction_right[None])
        left_points = sum(left_points[0], [])
        right_points = sum(right_points[0], [])
        scaling_factor = 180 / StereoVideoDataset.height
        p_L = p_L * scaling_factor
        p_R = p_R * scaling_factor
        for i in range(self.keypoints_two_kinds.shape[0]):
            p_L_hat = left_points[i]
            p_R_hat = right_points[i]
            distance_l = np.linalg.norm(p_L - p_L_hat, axis=1).min()
            distance_r = np.linalg.norm(p_R - p_R_hat, axis=1).min()
            self.assertLess(distance_l, 1.0)
            self.assertLess(distance_r, 1.0)

    def test_two_objects(self):
        T_LW = np.eye(4)
        T_RW = np.linalg.inv(self.T_LR) @ T_LW
        keypoints1 = self.keypoints_distinct
        keypoints2 = self.keypoints_distinct + np.array([[-0.5, 0.0, 0.0]])
        all_keypoints = np.concatenate([keypoints1, keypoints2], axis=0)
        heatmap_left, heatmap_right, p_L, p_R = compute_heatmaps(all_keypoints, config_distinct,
                T_LW, T_RW, self.K, self.D, self.Kp, self.Dp)

        prediction_left = np.zeros((4, 180, 320))
        prediction_right = np.zeros((4, 180, 320))
        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))

        extract_component = KeypointExtractionComponent(config_distinct, [180, 320])
        left_points, right_points = extract_component(prediction_left[None], prediction_right[None])
        left_points = left_points[0]
        right_points = right_points[0]
        p_L = p_L.reshape(2, -1, 2)
        p_R = p_R.reshape(2, -1, 2)
        for i, (left_keypoints, right_keypoints) in enumerate(zip(left_points, right_points)):
            distances_left = sklearn.metrics.pairwise_distances(p_L[:, i] * scaling_factor, left_keypoints).min(axis=1)
            distances_right = sklearn.metrics.pairwise_distances(p_R[:, i] * scaling_factor, right_keypoints).min(axis=1)
            np.testing.assert_array_less(distances_left, np.ones(distances_left.shape) * 0.5)
            np.testing.assert_array_less(distances_right, np.ones(distances_left.shape) * 0.5)

    def test_association(self):
        p_L = points_left_distinct.copy()
        p_R = points_right_distinct.copy()
        for _ in range(5):
            shuffled = p_R.copy()
            np.random.shuffle(shuffled)
            association = AssociationComponent()
            association.reset(self.K, self.Kp, self.D, self.Dp, self.T_LR,
                    1.0 / np.array([scaling_factor, scaling_factor]))
            left, right = association(p_L, shuffled)
            np.testing.assert_equal(p_L, left)
            # Should be subpixel error.
            # Some points might be associated to the same points if they are very close to each other.
            np.testing.assert_array_less(np.linalg.norm(p_R - right, axis=1), 1.0)

    def test_triangulation(self):
        p_L = points_left_distinct.copy()
        p_R = points_right_distinct.copy()
        triangulation = TriangulationComponent()
        triangulation.reset(self.K, self.Kp, self.D, self.Dp, self.T_RL,
                1.0 / np.array([scaling_factor, scaling_factor]))
        p_W = triangulation(p_L[None], p_R[None])[0]
        np.testing.assert_array_less(np.linalg.norm(p_W - self.keypoints_distinct, axis=1), 1e-5)

    def test_full_pipeline(self):
        T_LW = np.eye(4)
        T_RW = self.T_RL @ T_LW
        heatmap_left, heatmap_right, p_L, p_R = compute_heatmaps(self.keypoints_two_kinds, config_two_kinds,
                T_LW, T_RW, self.K, self.D, self.Kp, self.Dp)
        prediction_left = np.zeros((heatmap_left.shape[0], 180, 320))
        prediction_right = np.zeros((heatmap_left.shape[0], 180, 320))
        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))
        keypoint_extraction = KeypointExtractionComponent(config_two_kinds, [180, 320])
        association = AssociationComponent()
        triangulation = TriangulationComponent()
        pose_solve = PoseSolveComponent(keypoints_distinct)
        association.reset(self.K, self.Kp, self.D, self.Dp, self.T_RL, np.ones(2) / scaling_factor)
        triangulation.reset(self.K, self.Kp, self.D, self.Dp, self.T_RL, np.ones(2) / scaling_factor)
        left, right = keypoint_extraction(prediction_left[None], prediction_right[None])
        left, right = left[0], right[0]
        points = []
        for i in range(len(left)):
            left_points, right_points = association(np.stack(left[i]), np.stack(right[i]))
            points.append(triangulation(left_points[None], right_points[None])[0])
        self.assertEqual(points[0].shape, (1, 3))
        self.assertEqual(points[1].shape, (1, 3))
        self.assertEqual(points[2].shape, (3, 3))
        self.assertLess(np.linalg.norm(points[0][0] - self.keypoints_two_kinds[0]), 5e-2)
        self.assertLess(np.linalg.norm(points[1][0] - self.keypoints_two_kinds[1]), 5e-2)
        distances = sklearn.metrics.pairwise_distances(points[2], self.keypoints_two_kinds[2:]).min(axis=1)
        print(distances)
        left = np.stack(left[2])
        right = np.stack(right[2])
        from matplotlib import pyplot
        pyplot.imshow(prediction_right[2])
        pyplot.scatter(right_points[:, 0] - 0.5, right_points[:, 1] - 0.5, alpha=0.5, c='red')
        pyplot.show()

    def test_object_association(self, )


if __name__ == "__main__":
    unittest.main()
