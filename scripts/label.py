import argparse
import os
import hud
import random
import threading
import time
import h5py
import numpy as np
from skvideo import io as video_io
from perception.utils import linalg

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="Which directory to encoded video directories in.")
    return parser.parse_args()

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

class AddLeftPointCommand:
    def __init__(self, point):
        self.point = point

    def forward(self, labeler):
        labeler.left_keypoints.append(self.point)
        labeler.left_points.add_point(self.point, np.array([1.0, 0.5, 0.5, 1.0]))
        labeler._set_colors()

    def undo(self, labeler):
        if len(labeler.left_keypoints) == 0:
            return
        labeler.left_keypoints.pop()
        labeler.left_points.pop()
        labeler._set_colors()

class AddRightPointCommand:
    def __init__(self, point):
        self.point = point

    def forward(self, labeler):
        labeler.right_keypoints.append(self.point)
        labeler.right_points.add_point(self.point, np.array([1.0, 0.5, 0.5, 1.0]))
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
        self._create_views()

    def _create_views(self):
        self.window = hud.AppWindow("StereoLabeler", 1920, 540)
        self.left_image_pane = hud.ImagePane()
        self.right_image_pane = hud.ImagePane()
        self.left_points = hud.PointLayer([])
        left_pane = hud.ZStack()
        left_pane.add_view(self.left_image_pane)
        left_pane.add_view(self.left_points)
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
        self.current_dir = path
        if self.left_video is not None:
            self.left_video.close()
            self.right_video.close()
        self.left_video = video_io.vreader(os.path.join(path, 'left.mp4'))
        self.right_video = video_io.vreader(os.path.join(path, 'right.mp4'))
        i = random.randint(0, 50)
        left_frame = next(iter(self.left_video))
        right_frame = next(iter(self.right_video))
        self.left_image_pane.set_texture(left_frame)
        self.right_image_pane.set_texture(right_frame)

        self.hdf = h5py.File(os.path.join(path, 'data.hdf5'), 'r')
        self._load_camera_params()

    def _load_camera_params(self):
        left = self.hdf['calibration/left']
        right = self.hdf['calibration/right']
        K = np.empty((9,))
        Kp = np.empty((9,))
        left['K'].read_direct(K)
        right['K'].read_direct(Kp)
        self.K = np.zeros((3, 3))
        self.Kp = np.zeros((3, 3))
        self.K[:3, :3] = K.reshape(3, 3)
        self.Kp[:3, :3] = K.reshape(3, 3)

    def _left_pane_clicked(self, event):
        point = hud.utils.to_normalized_device_coordinates(event.p, self.left_image_pane.get_rect())
        command = AddLeftPointCommand(point)
        command.forward(self)
        self.commands.append(command)

    def _right_pane_clicked(self, event):
        point = hud.utils.to_normalized_device_coordinates(event.p, self.left_image_pane.get_rect())
        command = AddRightPointCommand(point)
        command.forward(self)
        self.commands.append(command)

    def _set_colors(self):
        left_colors = np.zeros((len(self.left_keypoints), 4))
        left_colors[:, :] = np.array([1.0, 0.5, 0.5, 1.0])[None]
        if len(self.left_keypoints) > len(self.right_keypoints):
            left_colors[len(self.right_keypoints)] = np.array([0.5, 1.0, 0.5, 1.0])
        self.left_points.set_colors(left_colors)

    def _key_callback(self, event):
        if event.key == 'Z' and event.modifiers & hud.modifiers.CTRL:
            self.undo()

    def undo(self):
        command = self.commands.pop()
        command.undo(self)

    def update(self):
        return self.window.update()

    def _add_right_view_line(self):
        image_rect = hud.Rect(0, 0, 1280, 720)
        right_rect = self.right_image_pane.get_rect()
        p = hud.utils.scale_to_view(self.left_keypoints[0], right_rect, image_rect)
        x = np.array([p.x, p.y, 1.0])[:, None]
        T_BL = self.hdf['left/camera_transform'][0]
        T_BR = self.hdf['right/camera_transform'][0]
        T_LB = np.linalg.inv(T_BL)
        T_LR = T_LB @ T_BR
        R = T_LR[:3, :3]
        t = T_LR[:3, 3]

        epipole = self.K @ R.T @ t

        F = np.linalg.pinv(self.Kp.T) @ R @ self.K.T @ linalg.skew_matrix(epipole)


def main():
    flags = read_args()

    app = LabelingApp()

    for subdir in os.listdir(flags.base_dir):
        path = os.path.join(flags.base_dir, subdir)
        app.set_current(path)
        while app.update() == True:
            time.sleep(0.01)
            app.window.wait_events()

if __name__ == "__main__":
    main()
