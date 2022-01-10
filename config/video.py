import os
import json
import h5py
import numpy as np
import yaml
import torch
import cv2
from numba import jit
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from perception.utils import camera_utils, linalg
from skvideo import io as video_io

from torch.nn import functional as F
import albumentations as A

heatmap_size = 64
center_radius = heatmap_size / 16.0
kernel_size = int(heatmap_size / 8.0)
default_length_scale = heatmap_size / 32.0

@jit(nopython=True)
def _gaussian_kernel(x, y, length_scale):
    norm = np.power(x - y, 2).sum(axis=-1)
    return np.exp(-norm / length_scale**2)

@jit(nopython=True)
def _compute_kernel(size, center, length_scale=default_length_scale):
    center = np.array([center, center], dtype=np.float32)
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x = np.array([i, j], dtype=np.float32)
            kernel[i, j] = _gaussian_kernel(center, x, length_scale=length_scale)
    return kernel / kernel.sum()

def _pixel_indices(height, width):
    out = np.zeros((2, height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            out[:, i, j] = np.array([j + 0.5, i + 0.5])
    return out

def _set_keypoints(heatmap, indices, length_scale=default_length_scale):
    for index in indices:
        int_x, int_y = index.astype(np.int32)
        start_x = np.maximum(int_x - kernel_size, 0)
        start_y = np.maximum(int_y - kernel_size, 0)
        end_x = np.minimum(int_x + kernel_size + 1, heatmap_size)
        end_y = np.minimum(int_y + kernel_size + 1, heatmap_size)
        for i in range(start_y, end_y):
            for j in range(start_x, end_x):
                heatmap[i, j] += _gaussian_kernel(index, np.array([j, i], dtype=index.dtype), length_scale=length_scale)

RGB_MEAN = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
RGB_STD = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)

class SceneDataset(IterableDataset):
    kernel_size = 50
    kernel_center = 25
    kernel = _compute_kernel(kernel_size, kernel_center)
    kernel_max = kernel.max()
    width = 1280
    height = 720
    width_resized = 511
    height_resized = 511
    prediction_size = np.array([heatmap_size, heatmap_size])
    # Offset x, y start point of cropped image.
    image_offset = np.array([(height_resized / height * width - 511.0) / 2.0, 0.0])

    def __init__(self, base_dir, keypoint_config, augment=False, augment_color=False, include_pose=False):
        self.base_dir = os.path.expanduser(base_dir)
        self.metadata_path = os.path.join(self.base_dir, "data.hdf5")
        self.augment = augment
        self.keypoint_config = [1] + keypoint_config['keypoint_config']
        self._init_points()
        self._load_calibration()
        self.target_size = self.prediction_size
        self.image_size = self.height_resized, self.width_resized
        self.include_pose = include_pose
        self.target_pixel_indices = _pixel_indices(*self.prediction_size)

        targets = {'image': 'image', 'keypoints': 'keypoints'}
        augmentations = []
        if augment:
            augmentations += [A.SmallestMaxSize(max_size=max(self.image_size)),
                    A.CenterCrop(height=self.image_size[0], width=self.image_size[1]),
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.CLAHE(p=0.1),
                    A.Cutout(max_h_size=25, max_w_size=25, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5)]
        else:
            augmentations += [A.SmallestMaxSize(max_size=max(self.image_size[0], self.image_size[1])),
                A.CenterCrop(height=self.image_size[0], width=self.image_size[1])]

        # if augment_color:
        #     augmentations += [A.RandomBrightnessContrast(p=0.25),
        #             A.ColorJitter(p=0.5),
        #             A.GaussNoise(p=0.25)]

        self.augmentations = A.Compose(augmentations, additional_targets=targets, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, check_each_transform=False))
        self.mean = RGB_MEAN
        self.std = RGB_STD

        with h5py.File(self.metadata_path, 'r') as f:
            self.poses = f['camera_transform'][:]

    def __len__(self):
        return self.poses.shape[0]

    def _load_calibration(self):
        calibration_file = os.path.join(self.base_dir, 'calibration.yaml')
        calibration = camera_utils.load_calibration_params(calibration_file)
        self.K = calibration['K']
        self.D = calibration['D']

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

    @classmethod
    def _add_kernel(cls, target, points):
        """
        target: height x width regression target
        points
        """
        center = np.array([cls.kernel_center, cls.kernel_center], dtype=np.float32)
        height = target.shape[0]
        width = target.shape[1]
        for i in range(points.shape[0]):
            point = points[i]
            x = round(point[0])
            y = round(point[1])
            x_start = max(x - cls.kernel_center, 0)
            x_end = max(min(x + cls.kernel_center, width), 0)
            y_start = max(y - cls.kernel_center, 0)
            y_end = max(min(y + cls.kernel_center, height), 0)
            y_range_start = 0
            y_range_end = cls.kernel_size
            x_range_start = 0
            x_range_end = cls.kernel_size
            if y_start == 0:
                y_range_start = abs(y - cls.kernel_center)
            if y + cls.kernel_center >= height:
                past_end = max(y + cls.kernel_center - height, 0)
                y_range_end = y_range_start + cls.kernel_size - past_end
            if x_start == 0:
                x_range_start = abs(x - cls.kernel_center)
            if x + cls.kernel_center > width:
                past_end = max(x + cls.kernel_center - width, 0)
                x_range_end = x_range_start + cls.kernel_size - past_end
            if (y_range_end - y_range_start) < 0 or (x_range_end - x_range_start) < 0:
                continue

            target[y_start:y_end, x_start:x_end] += cls.kernel[y_range_start:y_range_end, x_range_start:x_range_end]

    def __iter__(self):
        video_file = 'frames.mp4'
        video_file = os.path.join(self.base_dir, video_file)
        video = video_io.vreader(video_file)
        try:
            for i, frame in enumerate(video):
                yield self._extract_example(self.poses[i], frame)
        finally:
            video.close()

    def _extract_example(self, T_WC, frame):
        T_CW = linalg.inv_transform(T_WC)

        p_WK = self.world_points[:, :, None]

        R, _ = cv2.Rodrigues(T_CW[:3, :3])

        projected, _ = cv2.fisheye.projectPoints(self.world_points[:, None, :3], R, T_CW[:3, 3], self.K, self.D)
        projected = projected[:, 0, :]

        out = self.augmentations(image=frame, keypoints=projected)

        keypoints = np.array(out['keypoints'])

        scaling_factor = np.array([self.target_size]) / np.array([self.image_size])
        target = np.zeros((self.keypoint_maps, *self.target_size), dtype=np.float32)
        for object_index in range(self.n_objects):
            keypoint_start = object_index * self.n_keypoints
            keypoint_end = (object_index + 1) * self.n_keypoints
            points = keypoints[keypoint_start:keypoint_end] * scaling_factor
            for i, n_points in enumerate(self.keypoint_config):
                start = sum(self.keypoint_config[:i])
                end = start + n_points
                _set_keypoints(target[i], points[start:end])

        centers = self._compute_centers(keypoints)
        depth = self._compute_depth(keypoints, linalg.transform_points(T_CW, self.world_points))

        heatmap_max = np.maximum(target.max(axis=2).max(axis=1), 0.5)
        target = np.clip(target / heatmap_max[:, None, None], 0.0, 1.0)
        target = torch.tensor(target)
        centers = torch.tensor(centers)

        frame = torch.tensor((out['image'].astype(np.float32).transpose([2, 0, 1]) / 255.0 - self.mean[:, None, None]) / self.std[:, None, None])

        if not self.include_pose:
            return frame, target, depth, centers
        else:
            keypoints_out = np.zeros((self.n_keypoints * 4, 2))
            keypoints_out[:keypoints.shape[0], :] = keypoints
            keypoints_out = keypoints_out.reshape(4, self.n_keypoints, 2) * scaling_factor
            return frame, target, depth, centers, T_WC, keypoints_out

    def _compute_centers(self, projected_keypoints):
        scaling_factor = float(self.target_size[0] / self.image_size[0])
        projected_keypoints = projected_keypoints * scaling_factor
        center_map = np.zeros((self.keypoint_maps - 1, 2, *self.target_size), dtype=np.float32)

        keypoints = projected_keypoints.reshape(self.n_objects, self.n_keypoints, 2)
        for object_index in range(self.n_objects):
            center_keypoint = keypoints[object_index, 0]
            center_vectors = (center_keypoint[:, None, None] - self.target_pixel_indices)
            keypoint_index = 0
            for i, points_in_map in enumerate(self.keypoint_config[1:]):
                for j in range(points_in_map):
                    current_keypoint = keypoints[object_index, 1 + keypoint_index]
                    distance_to_keypoint = np.linalg.norm(current_keypoint[:, None, None] - self.target_pixel_indices, axis=0)
                    within_range = distance_to_keypoint < center_radius
                    center_map[i][:, within_range] = center_vectors[:, within_range]
                    keypoint_index += 1
        return center_map

    def _compute_depth(self, projected_keypoints, points_C):
        scaling_factor = float(self.target_size[0] / self.image_size[0])
        projected_keypoints = projected_keypoints * scaling_factor
        depth_map = np.zeros((self.keypoint_maps, *self.target_size), dtype=np.float32)

        keypoints = projected_keypoints.reshape(self.n_objects, self.n_keypoints, 2)
        points_3d = points_C.reshape(self.n_objects, self.n_keypoints, 3)

        for object_index in range(self.n_objects):
            keypoint_index = 0
            for i, points_in_map in enumerate(self.keypoint_config):
                for _ in range(points_in_map):
                    p_C = points_3d[object_index, keypoint_index]
                    current_keypoint = keypoints[object_index, keypoint_index]
                    distance_to_keypoint = np.linalg.norm(current_keypoint[:, None, None] - self.target_pixel_indices, axis=0)
                    within_range = distance_to_keypoint < center_radius
                    depth_map[i][within_range] = p_C[2]
                    keypoint_index += 1

        return depth_map

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



