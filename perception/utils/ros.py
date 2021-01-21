import numpy as np
from scipy.spatial.transform import Rotation

def transform_message_to_matrix(message):
    T = np.eye(4)
    t = message.transform.translation
    r = message.transform.rotation
    R = Rotation.from_quat([r.x, r.y, r.z, r.w])
    T[:3, 3]  = np.array([t.x, t.y, t.z])
    T[:3, :3] = R.as_matrix()
    return T

