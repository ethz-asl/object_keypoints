import numpy as np
import yaml

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
    image_size = calibration['cam1']['resolution'][::-1]
    return {
        'K': K,
        'Kp': Kp,
        'D': D,
        'Dp': Dp,
        'T_LR': np.linalg.inv(T_RL),
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

def shift_scale_camera_matrix(K, original_size, scaled_size, shift):
    out = K.copy()
    scaling_factor = scaled_size / original_size
    out[0, 0] = K[0, 0] * scaling_factor[0]
    out[1, 1] = K[1, 1] * scaling_factor[1]
    out[0, 2] = K[0, 2] * scaling_factor[0] - shift[0] * scaling_factor[0]
    out[1, 2] = K[1, 2] * scaling_factor[1] - shift[1] * scaling_factor[1]
    return out

