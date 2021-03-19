import argparse
import os
import json
import time
import hud
import h5py
import numpy as np
import cv2
import constants
import yaml
from skvideo import io as video_io
from perception.utils import camera_utils, Rate
hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="Which directory to encoded video directories in.")
    parser.add_argument('--calibration', default='config/calibration.yaml', help="Calibration yaml file.")
    return parser.parse_args()

KEYPOINT_FILENAME = 'keypoints.json'

class ViewModel:
    def __init__(self, flags, directory):
        self.flags = flags
        self._read_keypoints(directory)
        self._load_video(directory)
        self._load_metadata(directory)
        self.current_frame = 0

    def _read_keypoints(self, base_dir):
        filepath = os.path.join(base_dir, KEYPOINT_FILENAME)
        with open(filepath, 'r') as f:
            contents = json.loads(f.read())
        self.world_points = [np.array(p) for p in contents['3d_points']]

    def _load_video(self, base_dir):
        self.left_video = video_io.vreader(os.path.join(base_dir, 'left.mp4'))
        self.right_video = video_io.vreader(os.path.join(base_dir, 'right.mp4'))

    def _load_metadata(self, base_dir):
        self.hdf = h5py.File(os.path.join(base_dir, 'data.hdf5'), 'r')
        self.num_frames = min(self.hdf['left/camera_transform'].shape[0],
                self.hdf['right/camera_transform'].shape[0])

        calibration_file = self.flags.calibration
        with open(calibration_file, 'rt') as f:
            calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)
        left = calibration['cam0']
        right = calibration['cam1']

        self.K = camera_utils.camera_matrix(left['intrinsics'])
        self.Kp = camera_utils.camera_matrix(right['intrinsics'])
        self.D = np.array(left['distortion_coeffs'])
        self.Dp = np.array(right['distortion_coeffs'])
        self.T_LR = np.array(right['T_cn_cnm1'])
        self.T_RL = np.linalg.inv(self.T_LR)

    def close(self):
        self.hdf.close()
        self.left_video.close()
        self.right_video.close()

    def __iter__(self):
        return self

    def __next__(self):
        print(f"Current frame {self.current_frame}, num frames: {self.num_frames}" + 5 * " ", end="\r")
        if self.current_frame >= self.num_frames:
            raise StopIteration()
        T_WL = self.hdf['left/camera_transform'][self.current_frame]
        T_WR = self.hdf['right/camera_transform'][self.current_frame]
        T_LW = np.linalg.inv(T_WL)
        T_RW = np.linalg.inv(T_WR)
        T_RL = T_RW @ T_WL
        T_LR = np.linalg.inv(T_RL)
        left_frame_points = []
        right_frame_points = []
        for p_WK in self.world_points:
            p_LK = T_LW @ p_WK
            p_RK = T_RW @ p_WK

            p_l = self.K @ np.eye(3, 4) @ T_LW @ p_WK
            p_r = self.Kp @ np.eye(3, 4) @ T_RW @ p_WK
            p_l = p_l / p_l[2]
            p_r = p_r / p_r[2]

            T_WK = np.eye(4)
            T_WK[:, 3] = p_WK

            left_frame_points.append(
                    hud.utils.to_normalized_device_coordinates(
                        hud.Point(p_l[0], p_l[1]),
                        constants.IMAGE_RECT))

            right_frame_points.append(
                hud.utils.to_normalized_device_coordinates(
                    hud.Point(p_r[0], p_r[1]),
                    constants.IMAGE_RECT))

        left_frame = next(self.left_video)
        right_frame = next(self.right_video)

        self.current_frame += 1
        return left_frame, left_frame_points, right_frame, right_frame_points

    def _transform_point(self, T_WL, point):
        T_LW = np.linalg.inv(T_WL)
        return T_LW @ point


class PointVisualizer:
    def __init__(self, flags):
        self.flags = flags
        self.done = False
        self.window = hud.AppWindow("Keypoints", 1280, 360)
        self._create_views()

    def _create_views(self):
        self.left_image_pane = hud.ImagePane()
        self.left_image_points = hud.PointLayer([])
        self.right_image_pane = hud.ImagePane()
        self.right_image_points = hud.PointLayer([])
        z_stack = hud.ZStack()
        z_stack.add_view(self.left_image_pane)
        z_stack.add_view(self.left_image_points)

        z_stack_r = hud.ZStack()
        z_stack_r.add_view(self.right_image_pane)
        z_stack_r.add_view(self.right_image_points)

        h_stack = hud.HStack()
        h_stack.add_view(z_stack)
        h_stack.add_view(z_stack_r)
        self.window.set_view(h_stack)
        self.window.add_key_handler(self._key_callback)

    def _key_callback(self, event):
        if event.key == 'Q':
            self.done = True

    def run(self):
        rate = Rate(30)
        directories = os.listdir(self.flags.base_dir)
        for directory in directories:
            try:
                view_model = ViewModel(self.flags, os.path.join(self.flags.base_dir, directory))
                for left_frame, left_points, right_frame, right_points in view_model:
                    self.left_image_pane.set_texture(left_frame)
                    self.right_image_pane.set_texture(right_frame)
                    self.left_image_points.set_points(left_points, constants.KEYPOINT_COLOR[None].repeat(len(left_points), 0))
                    self.right_image_points.set_points(right_points, constants.KEYPOINT_COLOR[None].repeat(len(right_points), 0))
                    if not self.window.update() or self.done:
                        return
                    self.window.poll_events()
                    rate.sleep()
            finally:
                view_model.close()

def main():
    flags = read_args()

    app = PointVisualizer(flags)
    app.run()

if __name__ == "__main__":
    main()
