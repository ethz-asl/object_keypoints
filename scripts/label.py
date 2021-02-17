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

# Ros imports
import rospy
import tf2_ros
from perception.utils import ros as ros_utils
from geometry_msgs import msg as geometry_msgs
from scipy.spatial.transform import Rotation

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', help="Which directory to encoded video directories in.")
    parser.add_argument('--calibration', type=str, default='config/calibration.yaml',
            help="Path to kalibr calibration file.")
    return parser.parse_args()

def _write_points(out_file, Xs, left_keypoints, right_keypoints):
    """Writes keypoints to file as json. """
    contents = {
        'left_points': [[kp.x, kp.y] for kp in left_keypoints], # 2d keypoints in left image frame.
        'right_points': [[kp.x, kp.y] for kp in right_keypoints], # 2d keypoints in right image frame
        '3d_points': [x.ravel().tolist() for x in Xs] # Triangulated 3D points in world frame.
    } # Points are ordered and correspond to each other.
    with open(out_file, 'w') as f:
        f.write(json.dumps(contents))

def _project(p_WK, T_WC, K):
    T_CW = np.linalg.inv(T_WC)
    p_CK = T_CW @ p_WK
    p_CK /= p_CK[3]
    point = K @ p_CK[:3]
    return point / point[2]

def to_point_message(point):
    if point.shape[0] == 4:
        # Normalize.
        point /= point[3]
    msg = geometry_msgs.PointStamped()
    msg.header.frame_id = 'world'
    msg.header.stamp = rospy.Time.now()
    msg.point.x = point[0]
    msg.point.y = point[1]
    msg.point.z = point[2]
    return msg

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
        self.left_video = None
        self.right_video = None
        self.K = None
        self.Kp = None
        self.commands = []
        self.left_keypoints = []
        self.right_keypoints = []
        self.alpha = 1.0
        self._create_views()
        self.hdf = None
        self.done = False
        self._init_ros()

    def _init_ros(self):
        self.node = rospy.init_node("stereo_label")
        self.tf_publisher = tf2_ros.TransformBroadcaster()
        self.point_publisher = rospy.Publisher('keypoint', geometry_msgs.PointStamped, queue_size=1)

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

        if self.hdf is not None:
            self.hdf.close()
        self.hdf = h5py.File(os.path.join(path, 'data.hdf5'), 'r')
        self._load_camera_params()

        self.current_dir = path
        if self.left_video is not None:
            self.left_video.close()
            self.right_video.close()
        left_video_length = self.hdf['left/camera_transform'].shape[0]
        right_video_length = self.hdf['right/camera_transform'].shape[0]
        self.left_frame_index = random.randint(0, left_video_length // 4)
        self.right_frame_index = random.randint(0, right_video_length // 4)
        self.left_video = video_io.vreader(os.path.join(path, 'left.mp4'))
        self.right_video = video_io.vreader(os.path.join(path, 'right.mp4'))
        for _, left_frame in zip(range(self.left_frame_index + 1), self.left_video):
            continue
        for _, right_frame in zip(range(self.right_frame_index + 1), self.right_video):
            continue
        left_frame = cv2.undistort(left_frame, self.K, self.D)
        right_frame = cv2.undistort(right_frame, self.Kp, self.Dp)
        self.left_image_pane.set_texture(left_frame)
        self.right_image_pane.set_texture(right_frame)

        keypoint_path = os.path.join(path, KEYPOINT_FILENAME)
        if os.path.exists(keypoint_path):
            self._load_points(keypoint_path)


    def _load_points(self, keypoint_file):
        with open(keypoint_file, 'rt') as f:
            keypoints = json.loads(f.read())
            world_points = keypoints['3d_points']
            self.left_keypoints = []
            self.right_keypoints = []
            for point in world_points:
                T_WL = self.hdf['left/camera_transform'][self.left_frame_index]
                T_WR = self.hdf['right/camera_transform'][self.right_frame_index]
                left_point = _project(point, T_WL, self.K)
                right_point = _project(point, T_WR, self.Kp)
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

                T_WK = np.eye(4)
                T_WK[:3, 3] = point[:3]
                print("keypoint:", point)
                T_WK_msg = ros_utils.transform_to_message(T_WK, 'base_link', 'keypoint', rospy.Time.now())
                self.tf_publisher.sendTransform(T_WK_msg)

                ts = rospy.Time.now()
                T_WL_msg = ros_utils.transform_to_message(T_WL, 'base_link', 'camera_left', ts)
                T_WR_msg = ros_utils.transform_to_message(T_WR, 'base_link', 'camera_right', ts)
                self.tf_publisher.sendTransform(T_WL_msg)
                self.tf_publisher.sendTransform(T_WR_msg)


    def _load_camera_params(self):
        calibration_file = self.flags.calibration
        with open(calibration_file, 'rt') as f:
            calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)
        left = calibration['cam0']
        right = calibration['cam1']

        self.K = camera_utils.camera_matrix(left['intrinsics'])
        self.Kp = camera_utils.camera_matrix(right['intrinsics'])
        self.D = np.array(left['distortion_coeffs'])
        self.Dp = np.array(right['distortion_coeffs'])
        self.T_RL = np.array(right['T_cn_cnm1'])
        self.T_LR = np.linalg.inv(self.T_RL)

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
        if event.key == 'Q':
            self.quit()
        elif event.key == 'Z' and event.modifiers & hud.modifiers.CTRL:
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
        T_WL = self.hdf['left/camera_transform'][self.left_frame_index]
        T_WR = self.hdf['right/camera_transform'][self.right_frame_index]
        T_LR = np.linalg.inv(T_WL) @ T_WR
        T_RL = np.linalg.inv(T_WR) @ T_WL

        ts = rospy.Time.now()
        T_WL_msg = ros_utils.transform_to_message(T_WL, 'base_link', 'camera_left', ts)
        T_WR_msg = ros_utils.transform_to_message(T_WR, 'base_link', 'camera_right', ts)
        self.tf_publisher.sendTransform(T_WL_msg)
        self.tf_publisher.sendTransform(T_WR_msg)

        x = np.array([left_point.x, left_point.y])[:, None]
        xp = np.array([right_point.x, right_point.y])[:, None]

        P1 = camera_utils.projection_matrix(self.K, np.zeros(3), np.eye(3))
        P2 = self.Kp @ np.eye(3, 4) @ T_RL

        p_LK = cv2.triangulatePoints(P1, P2, x, xp)
        p_LK = p_LK / p_LK[3]
        print("p_LK:", p_LK.T)
        p_WK = T_WL @ p_LK

        projected_x = P1 @ p_LK
        projected_x /= projected_x[2]
        print('projected_x: ', projected_x.T)
        left_backprojected = hud.Point(projected_x[0], projected_x[1])
        left_backprojected = hud.utils.to_normalized_device_coordinates(left_backprojected, IMAGE_RECT)

        msg = to_point_message(p_WK.ravel())
        self.point_publisher.publish(msg)
        T_p = np.eye(4)
        T_p[:3, 3] = p_WK[:3].ravel()
        T_p_msg = ros_utils.transform_to_message(T_p, 'base_link', 'keypoint', rospy.Time.now())
        self.tf_publisher.sendTransform(T_p_msg)

        return p_WK, left_backprojected

    def quit(self):
        print("Mischief managed.")
        exit(0)


def main():
    flags = read_args()

    app = LabelingApp(flags)

    for subdir in os.listdir(flags.base_dir):
        path = os.path.join(flags.base_dir, subdir)
        app.set_current(path)
        while app.update():
            time.sleep(0.01)
            app.window.wait_events()

if __name__ == "__main__":
    main()
