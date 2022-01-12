import numpy as np
import cv2
import yaml
from . import linalg

class PinholeCamera:
    def __init__(self, K, D, image_size):
        # Camera matrix
        self.K = K
        self.Kinv = np.linalg.inv(K)
        # Distortion parameters
        self.D = D
        # height, width
        self.image_size = np.array(image_size)
        assert np.abs(K[0, 2] * 2.0 - image_size[1]) < 0.05 * image_size[1]

    def scale(self, scale):
        K = scale_camera_matrix(self.K, np.ones(2) * scale)
        return FisheyeCamera(K, self.D, self.image_size * scale)

    def cut(self, offset):
        cx = self.K[0, 2] - offset[0]
        cy = self.K[1, 2] - offset[1]
        K = self.K.copy()
        K[0, 2] = cx
        K[1, 2] = cy
        image_size = self.image_size - 2.0 * offset[::-1]
        return FisheyeCamera(K, self.D, image_size)

    def unproject(self, xys, zs):
        xs = np.concatenate([xys, np.ones((xys.shape[0], 1))], axis=1)
        X = (self.Kinv @ xs[:, :, None])[:, :, 0] * zs[:, None]
        return X

    def in_frame(self, x):
        """
        x: N x 2 array of points in image frame
        returns: N array of boolean values
        """
        under = (x <= 0.0).any(axis=1)
        over  = (x >= self.image_size).any(axis=1)
        return np.bitwise_or(under, over) == False

class RadTanPinholeCamera(PinholeCamera):
    def project(self, X, T_CW=np.eye(4)):
        """
        X: N x 3 points in world frame as define by T_CW
        returns: N x 2 points in image coordinates.
        """
        R, _ = cv2.Rodrigues(T_CW[:3, :3])
        x, _ = cv2.projectPoints(X[:, None, :], R, T_CW[:3, 3], self.K, self.D)
        x = x[:, 0]
        return x

    def undistort(self, xy):
        """
        xy: N x 2 image points
        returns: N x 2 undistorted image points.
        """
        return cv2.undistortPoints(xy[:, None, :], self.K, self.D,
                P=self.K)[:, 0, :]

class FisheyeCamera(PinholeCamera):
    def project(self, X, T_CW=np.eye(4)):
        """
        X: N x 3 points in world frame as define by T_CW
        returns: N x 2 points in image coordinates.
        """
        R, _ = cv2.Rodrigues(T_CW[:3, :3])
        x, _ = cv2.fisheye.projectPoints(X[:, None, :], R, T_CW[:3, 3], self.K, self.D)
        x = x[:, 0]
        return x

    def undistort(self, xy):
        """
        xy: N x 2 image points
        returns: N x 2 undistorted image points.
        """
        return cv2.fisheye.undistortPoints(xy[:, None, :], self.K, self.D,
                P=self.K)[:, 0, :]


class StereoCamera:
    def __init__(self, left_camera, right_camera, T_RL):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.T_RL = T_RL
        self.T_LR = linalg.inv_transform(T_RL)
        self.F = fundamental_matrix(T_RL, self.left_camera.K, self.right_camera.K)

    def triangulate(self, left_keypoints, right_keypoints):
        left_keypoints = left_keypoints[:, None, :].astype(np.float32)
        right_keypoints = right_keypoints[:, None, :].astype(np.float32)
        undistorted_left = cv2.fisheye.undistortPoints(left_keypoints, self.left_camera.K, self.left_camera.D,
                P=self.left_camera.K)[:, 0, :]
        undistorted_right = cv2.fisheye.undistortPoints(right_keypoints, self.right_camera.K, self.right_camera.D,
                P=self.right_camera.K)[:, 0, :]

        corrected_left, corrected_right = cv2.correctMatches(self.F, undistorted_left[None], undistorted_right[None])
        corrected_left, corrected_right = corrected_left[0], corrected_right[0]

        P1 = self.left_camera.K @ np.eye(3, 4)
        P2 = self.right_camera.K @ self.T_RL[:3]
        p_LK = cv2.triangulatePoints(
            P1, P2, corrected_left.T, corrected_right.T
        ).T  # N x 4
        p_LK = p_LK[:, :3] / p_LK[:, 3:4]

        return p_LK

def camera_matrix(intrinsics):
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]])

def projection_matrix(camera_matrix, T_CW):
    """
    camera_matrix: 3 x 3 camera calibration matrix.
    T_CW: 4x4 matrix transform from global to camera frame.
    """
    return camera_matrix @ T_CW[:3, :]

def from_calibration(calibration_file):
    with open(calibration_file, 'rt') as f:
        calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)
        camera = calibration['cam0']

        K = camera_matrix(camera['intrinsics'])
        D = np.array(camera['distortion_coeffs'])
        if camera['distortion_model'] == 'equidistant' and camera['camera_model'] == 'pinhole':
            return FisheyeCamera(K, D, camera['resolution'][::-1])
        elif camera['distortion_model'] == 'radtan' and camera['camera_model'] == 'pinhole':
            return RadTanPinholeCamera(K, D, camera['resolution'][::-1])
        else:
            raise ValueError(f"Unrecognized calibration type {camera['distortion_model']}.")

def load_calibration_params(calibration_file):
    with open(calibration_file, 'rt') as f:
        calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)

    left = calibration['cam0']
    K = camera_matrix(left['intrinsics'])
    right = calibration['cam1']
    Kp = camera_matrix(right['intrinsics'])
    D = np.array(calibration['cam0']['distortion_coeffs'])
    Dp = np.array(calibration['cam1']['distortion_coeffs'])

    T_RL = np.array(calibration['cam1']['T_cn_cnm1'])
    T_LR = np.eye(4)
    T_LR[:3, :3] = T_RL[:3, :3].transpose()
    T_LR[:3, 3] = -T_LR[:3, :3] @ T_RL[:3, 3]
    image_size = calibration['cam1']['resolution'][::-1]
    return {
        'K': K,
        'Kp': Kp,
        'D': D,
        'Dp': Dp,
        'T_LR': T_LR,
        'T_RL': T_RL,
        'image_size': image_size
    }

def scale_camera_matrix(K, scaling_factor):
    """
    K: 3 x 3 camera matrix
    scaling_factor: array of length 2, x and y scaling factor.
    """
    out = K.copy()
    out[0, 0] = K[0, 0] * scaling_factor[0]
    out[1, 1] = K[1, 1] * scaling_factor[1]
    out[0, 2] = K[0, 2] * scaling_factor[0]
    out[1, 2] = K[1, 2] * scaling_factor[1]
    return out

def fundamental_matrix(T_RL, K, Kp):
    R = T_RL[:3, :3]
    t = T_RL[:3, 3]

    C = linalg.skew_matrix(K @ R.T @ t)
    return np.linalg.inv(Kp).T @ R @ K.T @ C

