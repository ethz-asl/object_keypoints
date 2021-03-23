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

    T_RL = np.array(calibration['cam1']['T_cn_cnm1'])
    image_size = calibration['cam1']['resolution'][::-1]
    return {
        'K': K,
        'Kp': Kp,
        'T_RL': T_RL,
        'image_size': image_size
    }

