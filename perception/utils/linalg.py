import numpy as np

def skew_matrix(v):
    return np.array([[0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]], dtype=v.dtype)
