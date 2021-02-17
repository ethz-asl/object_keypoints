import hud
import numpy as np

def _to_camera_matrix(proj):
    return np.array([[proj[0], 0., proj[2]],
        [0., proj[1], proj[3]],
        [0., 0., 1.]], dtype=np.float64)

KEYPOINT_FILENAME = 'keypoints.json'
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
IMAGE_RECT = hud.Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
image_size = (int(IMAGE_RECT.width), int(IMAGE_RECT.height))
KEYPOINT_COLOR = np.array([1.0, 0.0, 0.0, 1.0])

