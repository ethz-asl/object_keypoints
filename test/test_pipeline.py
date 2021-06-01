import unittest
import numpy as np
import cv2
import sklearn
from perception.utils import camera_utils
from perception.datasets.video import StereoVideoDataset, _compute_kernel
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
    [571.28140636, 368.16440843]])
points_right_distinct = np.array([[600.68550127, 360.58934273],
    [603.22381954, 360.59871037],
    [668.67557233, 360.56260433],
    [530.24191134, 360.61583473]])

keypoints_X = np.array([[0.0, 0.0, 1.0],
    [0.0, 0.25, 1.0],
    [0.0, -0.25, 1.0]])

def compute_heatmaps(keypoints, keypoint_config, T_LW, T_RW, left_camera, right_camera):
    config = [1] + keypoint_config['keypoint_config']
    heatmap_left = np.zeros((len(config), StereoVideoDataset.height, StereoVideoDataset.width))
    heatmap_right = np.zeros_like(heatmap_left)
    p_L = left_camera.project(keypoints, T_LW)
    p_R = right_camera.project(keypoints, T_RW)
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

def compute_depth(image_points, keypoints, T_CW):
    depth_map = np.zeros((180, 320))
    for point, X in zip(image_points, keypoints):
        xy = point.round().astype(np.int32)
        depth_map[xy[1]-5:xy[1]+5, xy[0]-5:xy[0]] = np.linalg.norm(T_CW @ np.concatenate([X, np.ones(1)]), axis=0)
    return depth_map


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params = camera_utils.load_calibration_params('./config/calibration.yaml')
        K = params['K']
        D = params['D']
        Kp = params['Kp']
        Dp = params['Dp']
        T_RL = params['T_RL']
        cls.left_camera = camera_utils.FisheyeCamera(K, D, params['image_size'])
        cls.right_camera = camera_utils.FisheyeCamera(Kp, Dp, params['image_size'])
        cls.T_RL = T_RL
        cls.T_LR = linalg.inv_transform(T_RL)
        cls.stereo = camera_utils.StereoCamera(cls.left_camera, cls.right_camera, T_RL)
        cls.keypoints_distinct = np.zeros((keypoints_distinct.shape[0] + 1, 3))
        cls.keypoints_distinct[0] = keypoints_distinct.mean(axis=0)
        cls.keypoints_distinct[1:] = keypoints_distinct
        cls.keypoints_two_kinds = np.zeros((keypoints_two_kinds.shape[0] + 1, 3))
        cls.keypoints_two_kinds[0] = keypoints_two_kinds.mean(axis=0)
        cls.keypoints_two_kinds[1:] = keypoints_two_kinds
        StereoVideoDataset.kernel = _compute_kernel(50, 25, 10.0)
        left_cam = cls.stereo.left_camera.scale(scaling_factor)
        right_cam = cls.stereo.right_camera.scale(scaling_factor)
        cls.stereo_small = camera_utils.StereoCamera(left_cam, right_cam, cls.T_RL)

    def test_extract_single_points(self):
        T_LW = np.eye(4)
        T_RW = self.T_RL @ T_LW
        heatmap_left, heatmap_right, p_L, p_R = compute_heatmaps(self.keypoints_distinct, config_distinct,
                T_LW, T_RW, self.left_camera, self.right_camera)
        prediction_left = np.zeros((4, 180, 320))
        prediction_right = np.zeros((4, 180, 320))

        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))

        extract_component = KeypointExtractionComponent(config_distinct, [180, 320], bandwidth=3.0)
        (left_points, _), (right_points, _) = extract_component(prediction_left[None], prediction_right[None])
        for i in range(self.keypoints_distinct.shape[0]):
            p_L_hat = left_points[0][i][0]
            p_R_hat = right_points[0][i][0]
            # Check that error is less than a pixel.
            self.assertLess(np.linalg.norm(p_L_hat - p_L[i] * scaling_factor), 1.0)
            self.assertLess(np.linalg.norm(p_R_hat - p_R[i] * scaling_factor), 1.0)

    def test_extract_multiple(self):
        T_LW = np.eye(4)
        T_RW = np.linalg.inv(self.T_LR) @ T_LW
        heatmap_left, heatmap_right, p_L, p_R = compute_heatmaps(self.keypoints_two_kinds, config_two_kinds,
                T_LW, T_RW, self.left_camera, self.right_camera)
        prediction_left = np.zeros((heatmap_left.shape[0], 180, 320))
        prediction_right = np.zeros((heatmap_left.shape[0], 180, 320))

        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_left[i] /= prediction_left[i].max()
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))
            prediction_right[i] /= prediction_right[i].max()

        extract_component = KeypointExtractionComponent(config_two_kinds, [180, 320], bandwidth=3.0)
        (left_points, _), (right_points, _) = extract_component(prediction_left[None], prediction_right[None])
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
                T_LW, T_RW, self.left_camera, self.right_camera)

        prediction_left = np.zeros((4, 180, 320))
        prediction_right = np.zeros((4, 180, 320))
        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_left[i] /= prediction_left[i].max()
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))
            prediction_right[i] /= prediction_right[i].max()

        extract_component = KeypointExtractionComponent(config_distinct, [180, 320], bandwidth=3.0)
        (left_points, _), (right_points, _) = extract_component(prediction_left[None], prediction_right[None])
        left_points = left_points[0]
        right_points = right_points[0]
        p_L = p_L.reshape(2, -1, 2)
        p_R = p_R.reshape(2, -1, 2)
        for i, (left_keypoints, right_keypoints) in enumerate(zip(left_points, right_points)):
            distances_left = sklearn.metrics.pairwise_distances(p_L[:, i] * scaling_factor, left_keypoints).min(axis=1)
            distances_right = sklearn.metrics.pairwise_distances(p_R[:, i] * scaling_factor, right_keypoints).min(axis=1)
            np.testing.assert_array_less(distances_left, np.ones(distances_left.shape) * 0.5)
            np.testing.assert_array_less(distances_right, np.ones(distances_left.shape) * 0.5)

    def test_triangulation(self):
        p_L = points_left_distinct.copy()
        p_R = points_right_distinct.copy()
        triangulation = TriangulationComponent()
        triangulation.reset(self.stereo)
        p_W = triangulation(p_L, p_R)
        np.testing.assert_array_less(np.linalg.norm(p_W - self.keypoints_distinct, axis=1), 1e-3)

    def test_extraction_plus_triangulation(self):
        T_LW = np.eye(4)
        T_RW = self.T_RL @ T_LW
        heatmap_left, heatmap_right, p_L, p_R = compute_heatmaps(self.keypoints_two_kinds, config_two_kinds,
                T_LW, T_RW, self.left_camera, self.right_camera)
        prediction_left = np.zeros((heatmap_left.shape[0], 180, 320))
        prediction_right = np.zeros((heatmap_left.shape[0], 180, 320))
        for i in range(heatmap_left.shape[0]):
            prediction_left[i] = cv2.resize(heatmap_left[i], (320, 180))
            prediction_right[i] = cv2.resize(heatmap_right[i], (320, 180))

        keypoint_extraction = KeypointExtractionComponent(config_two_kinds, [180, 320], bandwidth=3.0)
        association = AssociationComponent()
        triangulation = TriangulationComponent()
        triangulation.reset(self.stereo_small)
        (left, _), (right, _) = keypoint_extraction(prediction_left[None], prediction_right[None])
        left, right = left[0], right[0]
        points = []
        for i in range(len(left)):
            left_points, right_points = np.stack(left[i]), np.stack(right[i])
            self.assertEqual(left_points.shape[0], right_points.shape[0])
            self.assertIn(left_points.shape[0], [1, 3])
            points.append(triangulation(left_points, right_points))
        self.assertEqual(points[0].shape, (1, 3))
        self.assertEqual(points[1].shape, (1, 3))
        self.assertEqual(points[2].shape, (3, 3))
        self.assertLess(np.linalg.norm(points[0][0] - self.keypoints_two_kinds[0]), 5e-2)
        self.assertLess(np.linalg.norm(points[1][0] - self.keypoints_two_kinds[1]), 5e-2)

    def test_association_simple(self):
        T_LW = np.eye(4)
        T_RW = self.T_RL @ T_LW
        T_WR = np.linalg.inv(T_RW)
        R, _ = cv2.Rodrigues(T_LW[:3, :3])
        points_left = self.left_camera.project(keypoints_X, T_LW) * 0.25
        # points_left = points_left.reshape(-1, 2) * 0.25
        points_right = self.right_camera.project(keypoints_X, T_RW) * 0.25
        association = AssociationComponent()
        association.reset(self.stereo)
        for _ in range(5):
            shuffled_right = points_right.copy()
            np.random.shuffle(shuffled_right)
            associations = association(points_left, shuffled_right)
            self.assertTrue((associations != -1).all())
            np.testing.assert_equal(points_right, shuffled_right[associations])

    def test_association_two_same(self):
        points_left = np.array([[160.251929, 92.04110211],
            [160.251929, 135.25386897],
            [160.251929, 48.82833525]])
        points_right = np.array([[149.9327, 139.14128],
            [149.93279695, 133.14128143],
            [149.88808034, 47.08818382]])
        T_LW = np.eye(4)
        T_RW = self.T_RL @ T_LW
        depth_left = compute_depth(points_left, keypoints_X, T_LW)
        depth_right = compute_depth(points_right, keypoints_X, T_RW)
        association = AssociationComponent()
        association.reset(self.stereo)
        associations = association(points_left, points_right)
        self.assertEqual(associations[0], -1)
        self.assertEqual(associations[1], 1)
        self.assertEqual(associations[2], 2)

    def test_association_tricky(self):
        points_left = np.array([[35.5, 25.5], [26.5, 39.5], [38.5, 39.5]])
        points_right = np.array([[29.5, 25.5], [20.5, 38.5], [33.5, 39.5]])
        K = np.array([[62.31692844, 0., 31.92640056],
            [ 0., 62.38274914, 32.92623658],
            [ 0., 0., 1.]])
        Kp = np.array([[62.07155716, 0., 31.79527486],
            [ 0., 62.14031698, 32.54056898],
            [ 0., 0., 1.]])
        D = np.array([-1.73678913e-01,  2.69084607e-02, -2.66312740e-04, -1.11094300e-04])
        Dp = np.array([-0.17596905,  0.02856535, -0.00036341, -0.00021308])
        camera_left = camera_utils.FisheyeCamera(K, D, [64, 64])
        camera_right = camera_utils.FisheyeCamera(Kp, Dp, [64, 64])
        camera_stereo = camera_utils.StereoCamera(camera_left, camera_right, self.T_RL)
        association = AssociationComponent()
        association.reset(camera_stereo)
        associations = association(points_left, points_right)
        self.assertEqual(associations.shape[0], 3)
        self.assertEqual(np.unique(associations).size, 3)

if __name__ == "__main__":
    unittest.main()
