import os
import json
import h5py
import numpy as np
import yaml
import torch
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from perception.utils import camera_utils
from skvideo import io as video_io

import albumentations as A

def _gaussian_kernel(x, y, length_scale=10.0):
    return np.exp(-np.linalg.norm(x - y)**2 / length_scale**2)

def _compute_kernel(size, center):
    center = np.array([center, center], dtype=np.float32)
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x = np.array([i, j], dtype=np.float32)
            kernel[i, j] = _gaussian_kernel(center, x)
    return kernel / kernel.sum()

class StereoVideoDataset(IterableDataset):
    kernel_size = 50
    kernel_center = 25
    kernel = _compute_kernel(kernel_size, kernel_center)
    width = 1280
    height = 720

    def __init__(self, base_dir, augment=False):
        self.base_dir = os.path.expanduser(base_dir)
        self._init_videos()
        self._init_metadata()
        self._init_calibration()
        self._init_points()

        if augment:
            self.preprocess = A.Compose([
                A.RandomCrop(height=360, width=640),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2)
                ], additional_targets={'image': 'image', 'target0': 'mask', 'target1': 'mask'})
        else:
            self.preprocess = A.Compose([
                A.Resize(height=360, width=640)
            ], additional_targets={'image': 'image', 'target0': 'mask', 'target1': 'mask'})
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    def __del__(self, *args):
        self.left_video.close()
        self.right_video.close()
        self.hdf.close()

    def _init_videos(self):
        left_video = os.path.join(self.base_dir, "left.mp4")
        right_video = os.path.join(self.base_dir, "right.mp4")
        self.left_video = video_io.vreader(left_video)
        self.right_video = video_io.vreader(right_video)

    def _init_metadata(self):
        filepath = os.path.join(self.base_dir, "data.hdf5")
        self.hdf = h5py.File(filepath)

    def _init_calibration(self):
        calibration_file = os.path.join(self.base_dir, 'calibration.yaml')
        with open(calibration_file, 'rt') as f:
            calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)
        left = calibration['cam0']
        right = calibration['cam1']

        self.K = camera_utils.camera_matrix(left['intrinsics'])
        self.Kp = camera_utils.camera_matrix(right['intrinsics'])
        # self.D = np.array(left['distortion_coeffs'])
        # self.Dp = np.array(right['distortion_coeffs'])

    def _init_points(self):
        filepath = os.path.join(self.base_dir, 'keypoints.json')
        with open(filepath, 'r') as f:
            contents = json.loads(f.read())
        self.world_points = np.array(contents['3d_points'])
        assert(self.world_points.shape[0] == 4)

    def _add_kernel(self, target, kernel, points):
        """
        target: height x width regression target
        kernel: kernel to add at point locations
        points
        """
        center = np.array([self.kernel_center, self.kernel_center], dtype=np.float32)
        for i in range(points.shape[0]):
            point = points[i]
            x = round(point[0])
            y = round(point[1])
            x_start = max(x - self.kernel_center, 0)
            x_end = min(x + self.kernel_center, self.width)
            y_start = max(y - self.kernel_center, 0)
            y_end = min(y + self.kernel_center, self.height)
            y_range_start = 0
            y_range_end = self.kernel_size
            x_range_start = 0
            x_range_end = self.kernel_size
            if y - self.kernel_center < 0:
                y_range_start = abs(y - self.kernel_center)
            if y + self.kernel_center >= self.height:
                y_range_end = y + self.kernel_center - self.height
            if x - self.kernel_center < 0:
                x_range_start = abs(x - self.kernel_center)
            if x + self.kernel_center > self.width:
                x_range_end = x + self.kernel_center - self.width
            target[y_start:y_end, x_start:x_end] += self.kernel[y_range_start:y_range_end, x_range_start:x_range_end]


    def __iter__(self):
        p_Wcenter = self.world_points[0]
        p_Wspokes = self.world_points[1:]
        for i, (left_frame, right_frame) in enumerate(zip(self.left_video, self.right_video)):
            T_WL = self.hdf['left/camera_transform'][i]
            T_WR = self.hdf['right/camera_transform'][i]
            T_LW = np.linalg.inv(T_WL)
            T_RW = np.linalg.inv(T_WR)

            p_WK = self.world_points[:, :, None]
            I = np.eye(3, 4)
            x_l = self.K @ I @ T_LW @ p_WK
            x_r = self.K @ I @ T_RW @ p_WK
            x_l = x_l[:, :, 0]
            x_l /= x_l[:, 2:]
            x_r = x_r[:, :, 0]
            x_r /= x_r[:, 2:]

            left_target = np.zeros((2, self.height, self.width), dtype=np.float32)
            right_target = np.zeros((2, self.height, self.width), dtype=np.float32)

            self._add_kernel(left_target[0], self.kernel, x_l[0:1])
            self._add_kernel(left_target[1], self.kernel, x_l[1:])
            self._add_kernel(right_target[0], self.kernel, x_r[0:1])
            self._add_kernel(right_target[1], self.kernel, x_r[1:])

            left_frame = (left_frame.astype(np.float32) / 255.0 - self.mean) / self.std
            right_frame = (right_frame.astype(np.float32) / 255.0 - self.mean) / self.std
            out_l = self.preprocess(image=left_frame, target0=left_target[0], target1=left_target[1])
            out_r = self.preprocess(image=right_frame, target0=right_target[0], target1=right_target[1])
            left_frame = out_l['image'].transpose([2, 0, 1])
            left_target = np.zeros((2, 360, 640), dtype=np.float32)
            left_target[0] = out_l['target0']
            left_target[1] = out_l['target1']

            right_target = np.zeros((2, 360, 640), dtype=np.float32)
            right_frame = out_r['image'].transpose([2, 0, 1])
            right_target[0] = out_r['target0']
            right_target[1] = out_r['target1']

            yield (left_frame, left_target), (right_frame, right_target)


