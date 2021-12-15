import argparse
import os
import hud
import json
import threading
import time
import h5py
import numpy as np
import cv2
import random
import yaml
from skvideo import io as video_io
from perception.utils import linalg
from constants import *
from perception.utils import camera_utils

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="Which directory to encoded video directories in.")
    parser.add_argument('--calibration', type=str, default='config/calibration.yaml',
            help="Path to kalibr calibration file.")
    return parser.parse_args()

def _write_points(out_file, Xs):
    """Writes keypoints to file as json. """
    contents = {
        '3d_points': [x.ravel().tolist() for x in Xs] # Triangulated 3D points in world frame.
    } # Points are ordered and correspond to each other.
    with open(out_file, 'w') as f:
        f.write(json.dumps(contents))

def _project(p_WK, T_WC, K, D):
    T_CW = np.linalg.inv(T_WC.astype(np.float64))
    p_WK = p_WK / p_WK[3]
    R, _ = cv2.Rodrigues(T_CW[:3, :3])
    point, _ = cv2.fisheye.projectPoints(p_WK[None, None, :3, 0], R, T_CW[:3, 3], K, D)
    return point.ravel()

BACKPROJECTED_POINT_COLOR = np.array([0.0, 0.0, 1.0, 1.0])

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
    def __init__(self, flags):
        self.flags = flags
        self.current_dir = None
        self.video = None
        self.K = None
        self.commands = []
        self.left_keypoints = []
        self.right_keypoints = []
        self.world_points = []
        self.alpha = 1.0
        self._create_views()
        self.hdf = None
        self.done = False

    def _create_views(self):
        self.window = hud.AppWindow("StereoLabeler", 1920, 540)
        self.left_image_pane = hud.ImagePane()
        self.right_image_pane = hud.ImagePane()
        self.left_points = hud.PointLayer([])
        self.left_backprojected_points = hud.PointLayer([])
        self.right_points = hud.PointLayer([])
        self.right_backprojected_points = hud.PointLayer([])

        left_pane = hud.ZStack()
        left_pane.add_view(self.left_image_pane)
        left_pane.add_view(self.left_points)
        left_pane.add_view(self.left_backprojected_points)
        self.left_image_pane.add_click_handler(self._left_pane_clicked)

        right_pane = hud.ZStack()
        right_pane.add_view(self.right_image_pane)
        right_pane.add_view(self.right_points)
        right_pane.add_view(self.right_backprojected_points)
        self.right_image_pane.add_click_handler(self._right_pane_clicked)

        main_view = hud.HStack()
        main_view.add_view(left_pane)
        main_view.add_view(right_pane)
        self.window.set_view(main_view)
        self.window.add_key_handler(self._key_callback)

    def _find_furthest(self):
        video_length = self.hdf['camera_transform'].shape[0]
        smallest_index = (None, None)
        value = 1.0
        stride = video_length // 30
        for i in range(0, video_length, stride):
            for j in range(i, video_length, stride):
                T_WL = self.hdf['camera_transform'][i]
                T_WR = self.hdf['camera_transform'][j]
                if np.linalg.norm(T_WL[:3, 3] - T_WR[:3, 3]) < 0.1:
                    # Skip if the viewpoints are too close to each other.
                    continue
                # Points are 1 meter along the z-axis from the camera position.
                z_L = T_WL[2, :3]
                z_R = T_WR[2, :3]

                dot = np.abs(z_L.dot(z_R))
                if dot < value:
                    value = dot
                    smallest_index = (i, j)
        print("Furthest frames: ", *smallest_index)
        return smallest_index

    def set_current(self, path):
        self.done = False
        self.left_keypoints = []
        self.right_keypoints = []
        self.world_points = []
        self.left_backprojected_points.clear_points()
        self.right_backprojected_points.clear_points()
        self.left_points.clear_points()
        self.right_points.clear_points()

        if self.hdf is not None:
            self.hdf.close()
        self.hdf = h5py.File(os.path.join(path, 'data.hdf5'), 'r')
        self._load_camera_params()

        self.current_dir = path
        self.left_frame_index, self.right_frame_index = self._find_furthest()

        self.video = video_io.vread(os.path.join(path, 'frames_preview.mp4'))
        print(self.hdf['camera_transform'].shape[0], "poses")
        left_frame = self.video[self.left_frame_index]
        right_frame = self.video[self.right_frame_index]
        self.left_image_pane.set_texture(left_frame)
        self.right_image_pane.set_texture(right_frame)

        keypoint_path = os.path.join(path, KEYPOINT_FILENAME)
        if os.path.exists(keypoint_path):
            self._load_points(keypoint_path)

    def _load_points(self, keypoint_file):
        #TODO: Either use backprojected keypoints or store which frame was used to select keypoints.
        # This wasn't a problem before swapping the frame was added.
        with open(keypoint_file, 'rt') as f:
            keypoints = json.loads(f.read())
            self.world_points = [np.array(x).reshape(4, 1) for x in keypoints['3d_points']]
            self.left_keypoints = []
            self.right_keypoints = []
            for point in self.world_points:
                T_WL = self.hdf['camera_transform'][self.left_frame_index]
                T_WR = self.hdf['camera_transform'][self.right_frame_index]
                left_point = _project(point, T_WL, self.K, self.D)
                right_point = _project(point, T_WR, self.K, self.D)
                left_hud = hud.Point(left_point[0], left_point[1])
                left_ndc = hud.utils.to_normalized_device_coordinates(left_hud, IMAGE_RECT)
                right_hud = hud.Point(right_point[0], right_point[1])
                right_ndc = hud.utils.to_normalized_device_coordinates(right_hud, IMAGE_RECT)
                self.left_points.add_point(left_ndc, KEYPOINT_COLOR)
                self.right_points.add_point(right_ndc, KEYPOINT_COLOR)
                self.commands.append(AddLeftPointCommand(left_hud, IMAGE_RECT))
                self.commands.append(AddRightPointCommand(right_hud, IMAGE_RECT))
                self.left_keypoints.append(left_hud)
                self.right_keypoints.append(right_hud)

    def _load_camera_params(self):
        calibration_file = self.flags.calibration
        with open(calibration_file, 'rt') as f:
            calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)
        camera = calibration['cam0']

        self.K = camera_utils.camera_matrix(camera['intrinsics'])
        self.D = np.array(camera['distortion_coeffs'])
        self.camera = camera_utils.FisheyeCamera(self.K, self.D, camera['resolution'][::-1])

    def _left_pane_clicked(self, event):
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

    def _swap_left_frame(self):
        self.left_frame_index = random.randint(0, self.hdf['camera_transform'].shape[0]-1)
        left_frame = self.video[self.left_frame_index]
        self.left_image_pane.set_texture(left_frame)
        self.left_keypoints = []
        self.left_points.clear_points()
        self._recompute_points()

    def _swap_right_frame(self):
        self.right_frame_index = random.randint(0, self.video.shape[0]-1)
        right_frame = self.video[self.right_frame_index]
        self.right_image_pane.set_texture(right_frame)
        self.right_keypoints = []
        self.right_points.clear_points()
        self._recompute_points()

    def _recompute_points(self):
        self.left_backprojected_points.clear_points()
        self.right_backprojected_points.clear_points()
        for world_point in self.world_points:
            T_WL = self.hdf['camera_transform'][self.left_frame_index]
            T_WR = self.hdf['camera_transform'][self.right_frame_index]
            left_point = self._backproject_left(world_point)
            right_point = self._backproject_right(world_point)
            self.left_backprojected_points.add_point(left_point, BACKPROJECTED_POINT_COLOR)
            self.right_backprojected_points.add_point(right_point, BACKPROJECTED_POINT_COLOR)

    def _set_colors(self):
        left_colors = np.zeros((len(self.left_keypoints), 4))
        left_colors[:, :] = np.array([1.0, 0.5, 0.5, 1.0])[None]
        if len(self.left_keypoints) > len(self.right_keypoints):
            left_colors[len(self.right_keypoints)] = np.array([0.5, 1.0, 0.5, 1.0])
        self.left_points.set_colors(left_colors)

    def _key_callback(self, event):
        if event.key == 'Q':
            self.quit()
        elif event.key == 'Z' and event.modifiers & hud.modifiers.CTRL:
            self.undo()
        elif event.key == 'A':
            self._swap_left_frame()
        elif event.key == 'B':
            self._swap_right_frame()
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
            self.left_backprojected_points.set_points([], np.zeros((0, 4)))
            self.right_backprojected_points.set_points([], np.zeros((0, 4)))
            self.world_points = []
            for left, right in zip(self.left_keypoints, self.right_keypoints):
                X = self._triangulate(left, right)
                left = self._backproject_left(X)
                right = self._backproject_right(X)
                self.left_backprojected_points.add_point(left, BACKPROJECTED_POINT_COLOR)
                self.right_backprojected_points.add_point(right, BACKPROJECTED_POINT_COLOR)
                self.world_points.append(X)
            out_file = os.path.join(self.current_dir, KEYPOINT_FILENAME)
            print("Saving points")
            _write_points(out_file, self.world_points)

    def _triangulate(self, left_point, right_point):
        T_WL = self.hdf['camera_transform'][self.left_frame_index]
        T_WR = self.hdf['camera_transform'][self.right_frame_index]
        T_LW = np.linalg.inv(T_WL)
        T_RW = np.linalg.inv(T_WR)
        T_LR = T_LW @ T_WR
        T_RL = T_RW @ T_WL

        x = np.array([left_point.x, left_point.y])[:, None]
        xp = np.array([right_point.x, right_point.y])[:, None]

        P1 = camera_utils.projection_matrix(self.K, np.eye(4))
        P2 = self.K @ np.eye(3, 4) @ T_RL

        x = cv2.fisheye.undistortPoints(x[None, None, :, 0], self.K, self.D, P=self.K).ravel()[:, None]
        xp = cv2.fisheye.undistortPoints(xp[None, None, :, 0], self.K, self.D, P=self.K).ravel()[:, None]

        p_LK = cv2.triangulatePoints(P1, P2, x, xp)
        p_LK = p_LK / p_LK[3]
        p_WK = T_WL @ p_LK
        return p_WK

    def _backproject_left(self, p_WK):
        T_WL = self.hdf['camera_transform'][self.left_frame_index]
        T_LW = np.linalg.inv(T_WL)
        p_WK = (p_WK / p_WK[3])
        R, _ = cv2.Rodrigues(T_LW[:3, :3])
        projected_x, _ = cv2.fisheye.projectPoints(p_WK[None, None, :3, 0], R, T_LW[:3, 3], self.K, self.D)
        projected_x = projected_x.ravel()
        left_backprojected = hud.Point(projected_x[0], projected_x[1])
        return hud.utils.to_normalized_device_coordinates(left_backprojected, IMAGE_RECT)

    def _backproject_right(self, p_WK):
        T_WR = self.hdf['camera_transform'][self.right_frame_index]
        T_RW = np.linalg.inv(T_WR)
        R, _ = cv2.Rodrigues(T_RW[:3, :3])
        projected_x, _ = cv2.fisheye.projectPoints(p_WK[None, None, :3, 0], R, T_RW[:3, 3], self.K, self.D)
        projected_x = projected_x.ravel()
        right_backprojected = hud.Point(projected_x[0], projected_x[1])
        return hud.utils.to_normalized_device_coordinates(right_backprojected, IMAGE_RECT)

    def quit(self):
        print("Mischief managed.")
        exit(0)


def main():
    flags = read_args()

    app = LabelingApp(flags)

    if os.path.exists(os.path.join(flags.base_dir, 'data.hdf5')):
        # Labeling single sequence
        sequence_directories = [flags.base_dir]
    else:
        sequence_directories = [os.path.join(flags.base_dir, d) for d in os.listdir(flags.base_dir)]
        sequence_directories.sort()
    for path in sequence_directories:
        print(f"Labeling: {path}")
        app.set_current(path)
        while app.update():
            time.sleep(0.01)
            app.window.wait_events()

if __name__ == "__main__":
    main()
