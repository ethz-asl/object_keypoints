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

from torch.nn import functional as F
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
RGB_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
RGB_STD = np.array([0.25, 0.25, 0.25], dtype=np.float32)

class StereoVideoDataset(IterableDataset):
    LEFT = 0
    RIGHT = 1
    kernel_size = 50
    kernel_center = 25
    kernel = _compute_kernel(kernel_size, kernel_center)
    kernel_max = kernel.max()
    width = 1280
    height = 720

    def __init__(self, base_dir, augment=False, target_size=[90, 160], camera=None, random_crop=False):
        if camera is None:
            camera = self.LEFT
        self.base_dir = os.path.expanduser(base_dir)
        self.metadata_path = os.path.join(self.base_dir, "data.hdf5")
        self._init_points()
        self.target_size = target_size
        self.camera = camera
        self.image_size = 360, 640

        if augment:
            augmentations = []
            if random_crop:
                augmentations += [A.RandomCrop(height=self.image_size[0], width=self.image_size[1])]
            else:
                augmentations += [A.Resize(height=self.image_size[0], width=self.image_size[1])]
            augmentations += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)]
            self.augmentations = A.Compose(augmentations, additional_targets={'image': 'image', 'target0': 'mask', 'target1': 'mask'})
        else:
            self.augmentations = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1])
            ], additional_targets={'image': 'image', 'target0': 'mask', 'target1': 'mask'})
        self.mean = RGB_MEAN
        self.std = RGB_STD

    def _init_videos(self):
        return left_video, right_video

    def _calibration(self):
        calibration_file = os.path.join(self.base_dir, 'calibration.yaml')
        with open(calibration_file, 'rt') as f:
            calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)

        if self.camera == self.LEFT:
            left = calibration['cam0']
            return camera_utils.camera_matrix(left['intrinsics'])
        elif self.camera == self.RIGHT:
            right = calibration['cam1']
            return camera_utils.camera_matrix(right['intrinsics'])

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
            x_end = max(min(x + self.kernel_center, self.width), 0)
            y_start = max(y - self.kernel_center, 0)
            y_end = max(min(y + self.kernel_center, self.height), 0)
            y_range_start = 0
            y_range_end = self.kernel_size
            x_range_start = 0
            x_range_end = self.kernel_size
            if y_start == 0:
                y_range_start = abs(y - self.kernel_center)
            if y + self.kernel_center >= self.height:
                past_end = max(y + self.kernel_center - self.height, 0)
                y_range_end = y_range_start + self.kernel_size - past_end
            if x_start == 0:
                x_range_start = abs(x - self.kernel_center)
            if x + self.kernel_center > self.width:
                past_end = max(x + self.kernel_center - self.width, 0)
                x_range_end = x_range_start + self.kernel_size - past_end
            if (y_range_end - y_range_start) < 0 or (x_range_end - x_range_start) < 0:
                continue

            target[y_start:y_end, x_start:x_end] += kernel[y_range_start:y_range_end, x_range_start:x_range_end]


    def __iter__(self):
        video_file = 'left.mp4' if self.camera == self.LEFT else 'right.mp4'
        video_file = os.path.join(self.base_dir, video_file)
        video = video_io.vreader(video_file)
        try:
            with h5py.File(self.metadata_path, 'r') as f:
                K = self._calibration()
                if self.camera == self.LEFT:
                    poses = f['left/camera_transform']
                elif self.camera == self.RIGHT:
                    poses = f['right/camera_transform']

                for i, frame in enumerate(video):
                    yield self._extract_example(poses, i, frame, K)
        finally:
            video.close()

    def _extract_example(self, poses, i, frame, K):
        T_WC = poses[i]
        T_CW = np.linalg.inv(T_WC)

        p_WK = self.world_points[:, :, None]
        I = np.eye(3, 4)
        x = K @ I @ T_CW @ p_WK
        x = x[:, :, 0]
        x /= x[:, 2:]

        target = np.zeros((2, self.height, self.width), dtype=np.float32)

        self._add_kernel(target[0], self.kernel, x[0:1])
        self._add_kernel(target[1], self.kernel, x[1:])

        frame = (frame.astype(np.float32) / 255.0 - self.mean) / self.std
        out = self.augmentations(image=frame, target0=target[0], target1=target[1])

        frame = torch.tensor(out['image'].transpose([2, 0, 1]))
        target = np.zeros((2, 360, 640), dtype=np.float32)
        target[0] = out['target0']
        target[1] = out['target1']
        target = torch.tensor(target)
        target = F.interpolate(target[None], size=self.target_size, mode='bilinear', align_corners=False)[0]

        return frame, (target / self.kernel_max * 2.0)  - 1.0

    @staticmethod
    def to_image(image):
        """
        Converts from torch.tensor to numpy array and reverses the image normalization process.
        Gives you a np.array with uint8 values in the range 0-255.
        image: numpy array 3 x H X W
        returns: np.uint8 array H x W x 3
        """
        image = np.transpose(image, [1, 2, 0])
        return np.clip((image * RGB_STD + RGB_MEAN) * 255.0, 0.0, 255.0).astype(np.uint8)



