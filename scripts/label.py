import argparse
import os
import hud
import json
import threading
import time
import h5py
import numpy as np
import cv2
from skvideo import io as video_io
from perception.utils import linalg

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="Which directory to encoded video directories in.")
    return parser.parse_args()

def to_camera(proj):
    return np.array([[proj[0], 0., proj[2], 0.],
        [0., proj[1], proj[3], 0.],
        [0., 0., 1., 0.]], dtype=np.float64)

KEYPOINT_FILENAME = 'keypoints.json'
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280
IMAGE_RECT = hud.Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
image_size = (int(IMAGE_RECT.width), int(IMAGE_RECT.height))
KEYPOINT_COLOR = np.array([1.0, 0.0, 0.0, 1.0])

LEFT_DIST  = [-0.17461027, 0.02754274, 0.00006249, 0.0000911 ]
LEFT_PROJ  = [ 697.87732212, 697.28594061, 648.08562626, 371.49958099]
RIGHT_DIST = [-0.17586632, 0.02861099, -0.0000037, -0.00002747]
RIGHT_PROJ = [ 695.49625866, 694.75757307, 646.83920067, 367.62123711]
T_LR = np.array([[1., 0., 0., 0.06297434],
	[0., 1., 0., 0.00017803],
        [0., 0., 1., 0.00006404],
        [0., 0., 0., 1.]], dtype=np.float64)
LEFT_K = to_camera(LEFT_PROJ)
RIGHT_K = to_camera(RIGHT_PROJ)

def _write_points(out_file, Xs, left_keypoints, right_keypoints):
    """Writes keypoints to file as json. """
    contents = {
        'left_points': [[kp.x, kp.y] for kp in left_keypoints], # 2d keypoints in left image frame.
        'right_points': [[kp.x, kp.y] for kp in right_keypoints], # 2d keypoints in right image frame
        '3d_points': [x.ravel().tolist() for x in Xs] # Triangulated 3D points in world frame.
    } # Points are ordered and correspond to each other.
    with open(out_file, 'w') as f:
        f.write(json.dumps(contents))


class PointCommand:
    def __init__(self, point, rect):
        """
        point: hud.Point in rect coordinates
        rect: the hud rectangle which was clicked.
        """
        self.point = point
        self.image_point = hud.utils.scale_to_view(point, rect, IMAGE_RECT)
        self.ndc_point = hud.utils.to_normalized_device_coordinates(point, rect)
        self.rect = rect

class AddLeftPointCommand(PointCommand):
    def forward(self, labeler):
        labeler.left_keypoints.append(self.image_point)
        labeler.left_points.add_point(self.ndc_point, np.array([1.0, 0.5, 0.5, 1.0]))
        labeler._set_colors()

    def undo(self, labeler):
        if len(labeler.left_keypoints) == 0:
            return
        labeler.left_keypoints.pop()
        labeler.left_points.pop()
        labeler._set_colors()

class AddRightPointCommand(PointCommand):
    def forward(self, labeler):
        labeler.right_keypoints.append(self.image_point)
        labeler.right_points.add_point(self.ndc_point, np.array([1.0, 0.5, 0.5, 1.0]))
        labeler._set_colors()

    def undo(self, labeler):
        if len(labeler.right_keypoints) == 0:
            return
        labeler.right_keypoints.pop()
        labeler.right_points.pop()
        labeler._set_colors()


class LabelingApp:
    def __init__(self):
        self.current_dir = None
        self.left_video = None
        self.right_video = None
        self.K = None
        self.Kp = None
        self.commands = []
        self.left_keypoints = []
        self.right_keypoints = []
        self.alpha = 1.0
        self._create_views()
        self.done = False

    def _create_views(self):
        self.window = hud.AppWindow("StereoLabeler", 1920, 540)
        self.left_image_pane = hud.ImagePane()
        self.right_image_pane = hud.ImagePane()
        self.left_points = hud.PointLayer([])
        self.left_back_projected_points = hud.PointLayer([])
        left_pane = hud.ZStack()
        left_pane.add_view(self.left_image_pane)
        left_pane.add_view(self.left_points)
        left_pane.add_view(self.left_back_projected_points)
        self.left_image_pane.add_click_handler(self._left_pane_clicked)

        right_pane = hud.ZStack()
        self.right_points = hud.PointLayer([])
        right_pane.add_view(self.right_image_pane)
        right_pane.add_view(self.right_points)

        self.right_image_pane.add_click_handler(self._right_pane_clicked)

        main_view = hud.HStack()
        main_view.add_view(left_pane)
        main_view.add_view(right_pane)
        self.window.set_view(main_view)
        self.window.add_key_handler(self._key_callback)

    def set_current(self, path):
        self.done = False
        self.left_back_projected_points.clear_points()
        self.left_keypoints = []
        self.right_keypoints = []
        self.left_points.clear_points()
        self.right_points.clear_points()

        keypoint_path = os.path.join(path, KEYPOINT_FILENAME)
        if os.path.exists(keypoint_path):
            self._load_points(keypoint_path)

        self.hdf = h5py.File(os.path.join(path, 'data.hdf5'), 'r')
        self._load_camera_params()

        self.current_dir = path
        if self.left_video is not None:
            self.left_video.close()
            self.right_video.close()
        self.left_video = video_io.vreader(os.path.join(path, 'left.mp4'))
        self.right_video = video_io.vreader(os.path.join(path, 'right.mp4'))
        left_frame = next(iter(self.left_video))
        right_frame = next(iter(self.right_video))
        self.left_image_pane.set_texture(left_frame)
        self.right_image_pane.set_texture(right_frame)

    def _load_points(self, keypoint_file):
        with open(keypoint_file, 'rt') as f:
            keypoints = json.loads(f.read())
            self.left_keypoints = [hud.Point(k[0], k[1]) for k in keypoints['left_points']]
            self.right_keypoints = [hud.Point(k[0], k[1]) for k in keypoints['right_points']]
            for left, right in zip(self.left_keypoints, self.right_keypoints):
                left_ndc = hud.utils.to_normalized_device_coordinates(left, IMAGE_RECT)
                right_ndc = hud.utils.to_normalized_device_coordinates(right, IMAGE_RECT)
                self.left_points.add_point(left_ndc, KEYPOINT_COLOR)
                self.right_points.add_point(right_ndc, KEYPOINT_COLOR)
                self.commands.append(AddLeftPointCommand(left, IMAGE_RECT))
                self.commands.append(AddRightPointCommand(right, IMAGE_RECT))

    def _load_camera_params(self):
        left = self.hdf['calibration/left']
        right = self.hdf['calibration/right']
        self.K = LEFT_K
        self.Kp = RIGHT_K
        self.D = np.array(LEFT_DIST)
        self.Dp = np.array(RIGHT_DIST)

    def _left_pane_clicked(self, event):
        print(f"left pane clicked {event.p.x} {event.p.y}")
        command = AddLeftPointCommand(event.p, self.left_image_pane.get_rect())
        command.forward(self)
        self.commands.append(command)
        self._save()

    def _right_pane_clicked(self, event):
        print(f"left pane clicked {event.p.x} {event.p.y}")
        command = AddRightPointCommand(event.p, self.right_image_pane.get_rect())
        command.forward(self)
        self.commands.append(command)
        self._save()

    def _set_colors(self):
        left_colors = np.zeros((len(self.left_keypoints), 4))
        left_colors[:, :] = np.array([1.0, 0.5, 0.5, 1.0])[None]
        if len(self.left_keypoints) > len(self.right_keypoints):
            left_colors[len(self.right_keypoints)] = np.array([0.5, 1.0, 0.5, 1.0])
        self.left_points.set_colors(left_colors)

    def _key_callback(self, event):
        if event.key == 'Z' and event.modifiers & hud.modifiers.CTRL:
            self.undo()
        elif event.key == '\x00':
            self.next_example()

    def undo(self):
        if len(self.commands) != 0:
            command = self.commands.pop()
            command.undo(self)

    def next_example(self):
        self.done = True

    def update(self):
        if self.done:
            return False
        else:
            return self.window.update()

    def _save(self):
        if len(self.left_keypoints) == len(self.right_keypoints):
            self.left_back_projected_points.set_points([], np.zeros((0, 4)))
            Xs = []
            for left, right in zip(self.left_keypoints, self.right_keypoints):
                X, point = self._triangulate(left, right)
                self.left_back_projected_points.add_point(point, KEYPOINT_COLOR)
                Xs.append(X)
            out_file = os.path.join(self.current_dir, KEYPOINT_FILENAME)
            print("Saving points")
            _write_points(out_file, Xs, self.left_keypoints, self.right_keypoints)

    def _triangulate(self, left_point, right_point):
        T_BL = self.hdf['left/camera_transform'][0]
        T_BR = self.hdf['right/camera_transform'][0]
        T_LB = np.linalg.inv(T_BL)
        T_LR = T_LB @ T_BR

        x = np.array([left_point.x, left_point.y])[:, None]
        xp = np.array([right_point.x, right_point.y])[:, None]

        out = cv2.triangulatePoints(self.K[:3, :3] @ np.eye(3, 4), self.Kp[:3, :3] @ T_LR[:3], x, xp)
        out = out / out[3] # Normalize.

        rvec, _ = cv2.Rodrigues(np.eye(3))
        projected_x, _ = cv2.projectPoints(out[:3], rvec=rvec, tvec=np.zeros(3), cameraMatrix=self.K[:3, :3], distCoeffs=self.D)
        projected_x = projected_x.ravel()
        left_backprojected = hud.Point(projected_x[0], projected_x[1])
        left_backprojected = hud.utils.to_normalized_device_coordinates(left_backprojected, IMAGE_RECT)
        return out, left_backprojected


def main():
    flags = read_args()

    app = LabelingApp()

    for subdir in os.listdir(flags.base_dir):
        path = os.path.join(flags.base_dir, subdir)
        app.set_current(path)
        while app.update():
            time.sleep(0.01)
            app.window.wait_events()

if __name__ == "__main__":
    main()
