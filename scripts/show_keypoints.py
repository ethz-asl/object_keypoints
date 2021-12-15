import argparse
import os
import json
import time
import hud
import h5py
import numpy as np
import cv2
import yaml
import random
from skvideo import io as video_io
from perception.constants import *
from perception.utils import camera_utils, Rate, linalg

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="Which directory to encoded video directories in.")
    parser.add_argument('--calibration', default='config/calibration.yaml', help="Calibration yaml file.")
    parser.add_argument('--rate', '-r', default=30, help="Frames per second.")
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


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
        self.video = video_io.vreader(os.path.join(base_dir, 'frames.mp4'))

    def _load_metadata(self, base_dir):
        self.hdf = h5py.File(os.path.join(base_dir, 'data.hdf5'), 'r')
        self.num_frames = self.hdf['camera_transform'].shape[0]

        calibration_file = self.flags.calibration
        with open(calibration_file, 'rt') as f:
            calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)
        camera = calibration['cam0']

        self.K = camera_utils.camera_matrix(camera['intrinsics'])
        self.D = np.array(camera['distortion_coeffs'])

    def close(self):
        self.hdf.close()
        self.video.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame >= self.num_frames:
            raise StopIteration()
        T_WC = self.hdf['camera_transform'][self.current_frame]
        T_CW = linalg.inv_transform(T_WC)
        R_c, _ = cv2.Rodrigues(T_CW[:3, :3])
        frame_points = []
        for p_WK in self.world_points:
            p_c, _ = cv2.fisheye.projectPoints(p_WK[None, None, :3], R_c, T_CW[:3, 3], self.K, self.D)
            p_c = p_c.ravel()

            frame_points.append(
                    hud.utils.to_normalized_device_coordinates(
                        hud.Point(p_c[0], p_c[1]), IMAGE_RECT))


        frame = next(self.video)

        self.current_frame += 1
        return frame, frame_points

    def _transform_point(self, T_WC, point):
        T_CW = np.linalg.inv(T_WC)
        return T_CW @ point


class PointVisualizer:
    def __init__(self, flags):
        self.flags = flags
        self.paused = False
        self.done = False
        self.window = hud.AppWindow("Keypoints", 640, 360)
        self._create_views()

    def _create_views(self):
        self.image_pane = hud.ImagePane()
        self.image_points = hud.PointLayer([])
        z_stack = hud.ZStack()
        z_stack.add_view(self.image_pane)
        z_stack.add_view(self.image_points)

        self.window.set_view(z_stack)
        self.window.add_key_handler(self._key_callback)

    def _key_callback(self, event):
        if event.key == 'Q':
            self.done = True
        elif event.key == ' ':
            self.paused = not self.paused

    def run(self):
        random.seed(self.flags.seed)
        rate = Rate(self.flags.rate)
        if os.path.isfile(os.path.join(self.flags.base_dir, 'keypoints.json')):
            directories = [os.path.basename(self.flags.base_dir)]
            base_dir = os.path.dirname(self.flags.base_dir)
        else:
            directories = os.listdir(self.flags.base_dir)
            base_dir = self.flags.base_dir
            random.shuffle(directories)
        for directory in directories:
            try:
                view_model = ViewModel(self.flags, os.path.join(base_dir, directory))
                print(f"Sequence {directory}")
                for frame, points in view_model:
                    print(f"Current frame {view_model.current_frame}, num frames: {view_model.num_frames}" + 5 * " ", end="\r")
                    self.image_pane.set_texture(frame)
                    self.image_points.set_points(points, KEYPOINT_COLOR[None].repeat(len(points), 0))
                    if not self.window.update() or self.done:
                        return
                    self.window.poll_events()
                    while self.paused:
                        self.window.poll_events()
                        rate.sleep()
                    rate.sleep()
            finally:
                view_model.close()

def main():
    flags = read_args()

    app = PointVisualizer(flags)
    app.run()

if __name__ == "__main__":
    main()
