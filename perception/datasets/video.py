import os
import json
import h5py
import numpy as np
import yaml
import torch
import cv2
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

def _pixel_indices(height, width):
    out = np.zeros((2, height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            out[:, i, j] = np.array([j + 0.5, i + 0.5])
    return out

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class StereoVideoDataset(IterableDataset):
    LEFT = 0
    RIGHT = 1
    kernel_size = 24
    kernel_center = 12
    kernel = _compute_kernel(kernel_size, kernel_center)
    kernel_max = kernel.max()
    width = 1280
    height = 720

    def __init__(self, base_dir, keypoint_config, augment=False, target_size=[180, 320], camera=None, random_crop=False,
            include_pose=False):
        if camera is None:
            camera = self.LEFT
        self.base_dir = os.path.expanduser(base_dir)
        self.metadata_path = os.path.join(self.base_dir, "data.hdf5")
        self.camera = camera
        self.keypoint_config = [1] + keypoint_config['keypoint_config']
        self._init_points()
        self._load_calibration()
        self.target_size = target_size
        self.image_size = 360, 640
        self.resize_target = self.image_size != tuple(target_size)
        self.include_pose = include_pose
        self.target_pixel_indices = _pixel_indices(*target_size)

        assert(target_size[0] / self.image_size[0] == target_size[1] / self.image_size[1])
        targets = {'image': 'image', 'keypoints': 'keypoints'}
        for i in range(self.keypoint_maps):
            targets[f'target{i}'] = 'mask'

        if augment:
            augmentations = []
            if random_crop:
                augmentations += [A.RandomCrop(height=self.image_size[0], width=self.image_size[1])]
            else:
                augmentations += [A.Resize(height=self.image_size[0], width=self.image_size[1])]
            augmentations += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)]
            self.augmentations = A.Compose(augmentations, additional_targets=targets,
                    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, check_each_transform=False))
        else:
            self.augmentations = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1])
            ], additional_targets=targets, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, check_each_transform=False))
        self.mean = RGB_MEAN
        self.std = RGB_STD

    def __len__(self):
        with h5py.File(self.metadata_path, 'r') as f:
            if self.camera == self.LEFT:
                return f['left/camera_transform'].shape[0]
            else:
                return f['right/camera_transform'].shape[0]

    def _init_videos(self):
        return left_video, right_video

    def _load_calibration(self):
        calibration_file = os.path.join(self.base_dir, 'calibration.yaml')
        calibration = camera_utils.load_calibration_params(calibration_file)

        if self.camera == self.LEFT:
            self.K = calibration['K']
            self.D = calibration['D']
        elif self.camera == self.RIGHT:
            self.K = calibration['Kp']
            self.D = calibration['Dp']

    def _init_points(self):
        filepath = os.path.join(self.base_dir, 'keypoints.json')
        with open(filepath, 'r') as f:
            contents = json.loads(f.read())
        world_points = np.array(contents['3d_points'])
        self.n_keypoints = sum(self.keypoint_config)
        self.n_objects = world_points.shape[0] // (self.n_keypoints - 1)
        self.keypoint_maps = len(self.keypoint_config)

        self.world_points = np.zeros((self.n_keypoints * self.n_objects, 3))
        n_real_keypoints = self.n_keypoints - 1
        # Add center points.
        for i in range(self.n_objects):
            start = i * self.n_keypoints
            end = (i+1) * self.n_keypoints
            object_points = world_points[i * n_real_keypoints:(i+1) * n_real_keypoints, :3]
            self.world_points[start] = object_points.mean(axis=0)
            self.world_points[start+1:end] = object_points

        # Should be an equal amount of points for each object.
        if world_points.shape[0] != ((self.n_keypoints-1) * self.n_objects):
            print(f"""
Wrong number of total keypoints {world_points.shape[0]} n_keypoints: {self.n_keypoints-1}\
 with {self.n_objects} objects in sequence {self.base_dir}
            """)
            assert False , "Wrong number of keypoints"

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
                if self.camera == self.LEFT:
                    poses = f['left/camera_transform']
                elif self.camera == self.RIGHT:
                    poses = f['right/camera_transform']

                for i, frame in enumerate(video):
                    yield self._extract_example(poses, i, frame)
        finally:
            video.close()

    def _extract_example(self, poses, i, frame):
        T_WC = poses[i]
        T_CW = np.linalg.inv(T_WC)

        p_WK = self.world_points[:, :, None]

        R, _ = cv2.Rodrigues(T_CW[:3, :3])

        projected, _ = cv2.fisheye.projectPoints(self.world_points[:, None, :3], R, T_CW[:3, 3], self.K, self.D)
        projected = projected[:, 0, :]

        target = np.zeros((self.keypoint_maps, self.height, self.width), dtype=np.float32)
        for object_index in range(self.n_objects):
            keypoint_start = object_index * self.n_keypoints
            keypoint_end = (object_index + 1) * self.n_keypoints
            points = projected[keypoint_start:keypoint_end]
            for i, n_points in enumerate(self.keypoint_config):
                start = sum(self.keypoint_config[:i])
                end = start + n_points
                self._add_kernel(target[i], self.kernel, points[start:end])

        frame = (frame.astype(np.float32) / 255.0 - self.mean) / self.std
        targets = {}
        for i in range(self.keypoint_maps):
            targets[f'target{i}'] = target[i]

        out = self.augmentations(image=frame, keypoints=projected, **targets)

        centers = self._compute_centers(np.array(out['keypoints']))

        frame = torch.tensor(out['image'].transpose([2, 0, 1]))
        target = np.zeros((self.keypoint_maps, 360, 640), dtype=np.float32)
        for i in range(self.keypoint_maps):
            target[i] = out[f'target{i}']

        target = torch.tensor(target)
        if self.resize_target:
            target = F.interpolate(target[None], size=self.target_size, mode='bilinear', align_corners=False)[0]

        centers = torch.tensor(centers)

        target = torch.clamp(target / self.kernel_max, 0.0, 1.0)
        if not self.include_pose:
            return frame, target, centers
        else:
            return frame, target, centers, T_WC

    def _compute_centers(self, projected_keypoints):
        scaling_factor = float(self.target_size[0] / self.image_size[0])
        projected_keypoints = projected_keypoints * scaling_factor
        centers = np.zeros((2, *self.target_size), dtype=np.float32)
        for object_index in range(self.n_objects):
            center_keypoint = projected_keypoints[object_index * self.n_keypoints]
            center_vectors = (center_keypoint[:, None, None] - self.target_pixel_indices)
            for i in range(1, self.n_keypoints):
                keypoint = projected_keypoints[object_index * self.n_keypoints + i]
                distance_to_keypoint = np.linalg.norm(keypoint[:, None, None] - self.target_pixel_indices, axis=0)
                within_range = distance_to_keypoint < 6.25
                centers[:, within_range] = center_vectors[:, within_range]
        return centers

    @staticmethod
    def to_image(image):
        """
        Converts from torch.tensor to numpy array and reverses the image normalization process.
        Gives you a np.array with uint8 values in the range 0-255.
        image: numpy array 3 x H X W
        returns: np.uint8 array H x W x 3
        """
        image = image.transpose([1, 2, 0])
        return np.clip((image * RGB_STD + RGB_MEAN) * 255.0, 0.0, 255.0).astype(np.uint8)



