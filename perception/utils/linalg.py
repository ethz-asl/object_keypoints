import numpy as np
from scipy.spatial.transform import Rotation

def skew_matrix(v):
    return np.array([[0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]], dtype=v.dtype)

def inv_transform(T):
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3,:3] @ T[:3, 3]
    return out

def transform_points(T, points):
    """
    T: 4 x 4 numpy matrix
    points: ... x 3 numpy matrix
    """
    return (T[:3, :3] @ points[..., None])[..., 0] + T[:3, 3]

def angle_between(R1, R2):
    return Rotation.from_matrix(R1.T @ R2).as_euler('xyz', degrees=False)
