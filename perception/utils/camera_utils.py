import numpy as np

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

