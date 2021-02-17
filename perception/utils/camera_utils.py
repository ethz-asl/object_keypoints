import numpy as np

def camera_matrix(intrinsics):
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]])

def projection_matrix(camera_matrix, p_WC, R_WC):
    """
    camera_matrix: 3 x 3 camera calibration matrix.
    p_WC: the camera center point expressed in world frame.
    R_WC: the camera rotation expressed in world frame.
    """
    RT = np.zeros((3, 4))
    RT[:3, :3] = R_WC.T
    RT[:3, 3] = -p_WC
    return camera_matrix @ RT

