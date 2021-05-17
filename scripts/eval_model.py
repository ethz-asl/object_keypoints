import argparse
import os
import hud
import time
import json
import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from perception.datasets.video import StereoVideoDataset
from perception.utils import Rate, camera_utils, clustering_utils, linalg
from perception.models import nms
from matplotlib import cm
from matplotlib import pyplot
from perception.pipeline import *
from rich.console import Console
from rich.table import Table

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="Path to dataset folder.")
    parser.add_argument('--model', '-m', type=str, required=True, help="Path to the model to evaluate.")
    parser.add_argument('--centers', action='store_true', help="Show object center predictions.")
    parser.add_argument('--ground-truth', action='store_true', help="Show labels instead of making predictions.")
    parser.add_argument('--keypoints', type=str, default='config/cups.json', help="The keypoint configuration file.")
    parser.add_argument('--write', type=str, help="Write frames to folder.")
    parser.add_argument('--cpu', action='store_true', help='Run model on cpu.')
    parser.add_argument('--world', action='store_true', help='Project points from world points.')
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    return parser.parse_args()

IMAGE_RECT = hud.Rect(0.0, 0.0, 511, 511.0)
class Sequence:
    def __init__(self, flags, sequence):
        self.flags = flags
        self.sequence_path = sequence
        self.prediction_size = StereoVideoDataset.prediction_size
        with open(flags.keypoints, 'rt') as f:
            self.keypoint_config = json.load(f)
        self.dataset_left = StereoVideoDataset(sequence, self.keypoint_config,
            camera=0, augment=False, include_pose=True)
        self.left_loader = self._loader(self.dataset_left)
        self.right_loader = self._loader(StereoVideoDataset(sequence, self.keypoint_config,
            camera=1, augment=False, include_pose=True))
        self.size_resized = np.array([StereoVideoDataset.height_resized, StereoVideoDataset.width_resized])
        self.scaling_factor = self.prediction_size / self.size_resized
        self.image_offset = StereoVideoDataset.image_offset
        self.scaled_size = np.array([1280.0 * (511.0 / 720.0), 511.0]) * self.scaling_factor
        self.scale_prediction_to_image = self.prediction_size / self.size_resized
        self.scale_full_to_resized = np.ones(2) / (511.0 / 720.0)

        self._load_calibration()
        self._read_keypoints()

    def _loader(self, dataset):
        return DataLoader(dataset, num_workers=0, batch_size=1, pin_memory=not self.flags.cpu and torch.cuda.is_available())

    def _read_keypoints(self):
        self.world_points = self.dataset_left.world_points.reshape(self.dataset_left.n_objects, -1, 3)
        filepath = os.path.join(self.sequence_path, 'keypoints.json')
        with open(filepath, 'rt') as f:
            self.keypoints = np.array(json.loads(f.read())['3d_points'])[:, :3]

    def _load_calibration(self):
        calibration_file = os.path.join(self.sequence_path, 'calibration.yaml')
        params = camera_utils.load_calibration_params(calibration_file)
        self.K = params['K']
        self.Kp = params['Kp']
        self.D = params['D']
        self.Dp = params['Dp']
        self.T_LR = params['T_LR']
        self.T_RL = params['T_RL']
        self.R_L, _ = cv2.Rodrigues(self.T_RL[:3, :3])
        self.R_R, _ = cv2.Rodrigues(np.eye(3))
        self.image_size = np.array(params['image_size'])

        self.K_small = camera_utils.shift_scale_camera_matrix(self.K, self.image_size[::-1], self.scaled_size, self.image_offset)
        self.Kp_small = camera_utils.shift_scale_camera_matrix(self.Kp, self.image_size[::-1], self.scaled_size, self.image_offset)

    def to_image_points(self, predictions):
        # predictions: points in 2d prediction space.
        return predictions / self.scale_prediction_to_image

    def project_points_left(self, p_LK):
        p_LK = np.concatenate(p_LK, axis=0)
        points, _ = cv2.fisheye.projectPoints(p_LK[:, None], self.R_L, self.T_RL[:3, 3], self.K_small, self.D)
        return points.reshape(-1, 2)

    def project_points_right(self, p_LK):
        p_LK = np.concatenate(p_LK, axis=0)
        points, _ = cv2.fisheye.projectPoints(p_LK[:, None], self.R_R, 2.0 * self.T_RL[:3, 3], self.Kp_small, self.Dp)
        return points.reshape(-1, 2)

object_color_maps = [cm.get_cmap('Reds'), cm.get_cmap('Purples'), cm.get_cmap('Greens'), cm.get_cmap('Blues'), cm.get_cmap('Oranges')]
n_colormaps = len(object_color_maps)
def _point_colors(object_points):
    # Color per object.
    colors = []
    for i, points in enumerate(object_points):
        colors.append(object_color_maps[i % n_colormaps](np.linspace(0.7, 0.9, points.shape[0])))
    return np.concatenate(colors)

class Visualizer:
    def __init__(self):
        self.done = False
        self.window = hud.AppWindow("Keypoints", 650, 360)
        self._create_views()

    def _create_views(self):
        self.left_image_pane = hud.ImagePane()
        self.right_image_pane = hud.ImagePane()
        self.left_points = hud.PointLayer([])
        self.right_points = hud.PointLayer([])

        left_view = hud.ZStack()
        left_view.add_view(self.left_image_pane)
        left_view.add_view(self.left_points)

        right_view = hud.ZStack()
        right_view.add_view(self.right_image_pane)
        right_view.add_view(self.right_points)

        h_stack = hud.HStack()
        h_stack.add_view(left_view)
        h_stack.add_view(right_view)
        self.window.set_view(h_stack)
        self.window.add_key_handler(self._key_callback)

    def _key_callback(self, event):
        if event.key == 'Q':
            self.done = True

    def set_left_image(self, image):
        self.left_image_pane.set_texture(image)

    def set_right_image(self, image):
        self.right_image_pane.set_texture(image)

    def set_left_points(self, points):
        colors = _point_colors(points)
        points = np.concatenate(points)
        points = self._to_hud_points(points)
        hud_points = [hud.utils.to_normalized_device_coordinates(p, IMAGE_RECT) for p in points]
        self.left_points.set_points(hud_points, colors)

    def set_right_points(self, points):
        colors = _point_colors(points)
        points = np.concatenate(points)
        points = self._to_hud_points(points)
        points = [hud.utils.to_normalized_device_coordinates(p, IMAGE_RECT) for p in points]
        self.right_points.set_points(points, colors)

    def _to_hud_points(self, points):
        return [hud.Point(p[0], p[1]) for p in points]

    def update(self):
        self.window.poll_events()
        if not self.window.update() or self.done:
            return True
        return False

class Results:
    def __init__(self):
        self.gt_keypoints = []
        self.predicted_keypoints = []
        self.K = self.Kp = self.D = self.Dp = None
        self.T_RL = None
        self.console = Console()
        self.screen = self.console.screen()

    def add(self, T_WL, T_WR, objects, scene_points):
        """
        T_WL: transform from left frame to world
        objects: object detections
        keypoints: n_objects X n_keypoints X 3 gt keypoints
        """
        centers = scene_points[:, 0, :]
        gt_keypoints = []
        keypoints = []
        T_LW = np.linalg.inv(T_WL)
        T_RW = np.linalg.inv(T_WR)
        T_RL = T_RW @ T_WL
        print(np.abs(T_RL - self.T_RL).round(3))
        R_L, _ = cv2.Rodrigues(T_LW[:3, :3])
        R_R, _ = cv2.Rodrigues(T_RW[:3, :3])
        for obj in objects:
            p_LK = obj['p_L']
            object_keypoints = []
            center_W = T_WL @ np.concatenate([p_LK[0].T, np.ones((1, 1))], axis=0)
            closest_object = np.linalg.norm(centers - center_W[:3, 0], axis=1).argmin()
            gt_points = []
            for i, point in enumerate(p_LK):
                gt_point = scene_points[closest_object, i]
                gt_point_left = cv2.fisheye.projectPoints(gt_point[None, None, :], R_L, T_LW[:3, 3], self.K, self.D)[0].ravel()
                gt_point_right = cv2.fisheye.projectPoints(gt_point[None, None, :], R_R, T_RW[:3, 3], self.Kp, self.Dp)[0].ravel()
                if (gt_point_left <= 0.0).any() or (gt_point_left >= self.frame_size).any():
                    print('point in left not in view')
                    continue
                if (gt_point_right <= 0.0).any() or (gt_point_right >= self.frame_size).any():
                    print('point in right not in view')
                    continue

                if point is not None and point.size > 0:
                    p_W = T_WL @ np.concatenate([point[:1].T, np.ones((1, 1))], axis=0)
                    point_left = cv2.fisheye.projectPoints(p_W[None, None, :3, 0], R_L, T_LW[:3, 3], self.K, self.D)[0].ravel()
                    print('image space: ', np.abs(point_left - gt_point_left).round(2))
                    print('world space: ', np.abs(gt_point - p_W[:3, 0]).round(2))
                    object_keypoints.append(p_W[:3, 0])
                else:
                    object_keypoints.append(None)
                gt_points.append(scene_points[closest_object, i])
            gt_keypoints.append(gt_points)
            keypoints.append(object_keypoints)
        self.gt_keypoints.append(gt_keypoints)
        self.predicted_keypoints.append(keypoints)

    def set_calibration(self, K, Kp, D, Dp, T_RL, frame_size):
        self.frame_size = frame_size
        self.T_RL = T_RL
        self.T_LR = np.linalg.inv(T_RL)
        self.K = K
        self.Kp = Kp
        self.D = D
        self.Dp = Dp

    def print_results(self):
        errors = []
        missing = 0
        n_points = 0
        for gt, predicted in zip(self.gt_keypoints, self.predicted_keypoints):
            assert len(gt) == len(predicted)
            for gt_points, p_points in zip(gt, predicted):
                assert len(gt_points) == len(p_points)
                for gt_point, p_point in zip(gt_points, p_points):
                    n_points += 1
                    if p_point is not None:
                        error = np.linalg.norm(gt_point - p_point, 2, axis=0)
                        errors.append(error)
                    else:
                        missing += 1
        table = Table(show_header=True)
        table.add_column("mean")
        table.add_column("std")
        table.add_column("missing")
        table.add_column("points")
        errors = np.array(errors) * 100.0 # convert to cm
        missing_percentage = (float(missing) / float(n_points)) * 100.0
        table.add_row(f"{errors.mean()}", f"{errors.std()}", f"{missing_percentage:.02f}%", f"{n_points}")
        self.screen.update(table)

class Runner:
    IMAGE_SIZE = (511, 511)
    def __init__(self):
        self.flags = read_args()
        if not self.flags.write:
            self.visualizer = Visualizer()
            self.figure = None
        else:
            self.visualizer = None
            self.figure = pyplot.figure(figsize=(14, 8))
        self.frame_number = 0
        self.pipeline = None
        self.results = Results()

    def _sequences(self):
        return sorted([os.path.join(self.flags.data, s) for s in os.listdir(self.flags.data)])

    def _to_image(self, frame):
        frame = StereoVideoDataset.to_image(frame)
        return cv2.resize(frame, self.IMAGE_SIZE)

    def _to_heatmap(self, target):
        target = np.clip(target, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)

    def _write_frames(self, left, left_points, right, right_points):
        x_scaling = left.shape[1] / 1280.0
        y_scaling = left.shape[0] / 720.0
        axis = pyplot.subplot2grid((1, 2), loc=(0, 0), fig=self.figure)
        axis.imshow(left)
        c = _point_colors(left_points.shape[0])
        axis.scatter(left_points[:, 0] * x_scaling, left_points[:, 1] * y_scaling, s=5.0, color=c)
        axis.axis('off')
        axis = pyplot.subplot2grid((1, 2), loc=(0, 1), fig=self.figure)
        axis.imshow(right)
        c = _point_colors(right_points.shape[0])
        axis.scatter(right_points[:, 0] * x_scaling, right_points[:, 1] * y_scaling, s=5.0, color=c)
        axis.axis('off')
        self.figure.savefig(os.path.join(self.flags.write, f'{self.frame_number:06}.jpg'), pil_kwargs={'quality': 85}, bbox_inches='tight')
        self.figure.clf()

    def _play_predictions(self, sequence):
        if self.flags.ground_truth:
            self.pipeline = ObjectKeypointPipeline(sequence.prediction_size, sequence.keypoints, sequence.keypoint_config)
        else:
            self.pipeline = LearnedKeypointTrackingPipeline(self.flags.model, not self.flags.cpu and torch.cuda.is_available(),
                    sequence.prediction_size, sequence.keypoints, sequence.keypoint_config)
        self.pipeline.reset(sequence.K, sequence.Kp, sequence.D, sequence.Dp, sequence.T_RL, sequence.image_size[::-1], sequence.scaled_size, sequence.image_offset)
        self.results.set_calibration(sequence.K_small, sequence.Kp_small, sequence.D, sequence.Dp, sequence.T_RL, sequence.prediction_size)

        rate = Rate(30)
        for i, ((left_frame, l_target, l_centers, T_WL, l_keypoints), (right_frame, r_target, r_centers, T_WR, r_keypoints)) in enumerate(zip(sequence.left_loader, sequence.right_loader)):
            N = left_frame.shape[0]
            if self.flags.ground_truth:
                l_target = nms(l_target)
                r_target = nms(r_target)
                objects = self.pipeline(l_target, l_centers, r_target, r_centers)
                p_left = l_target[0]
                p_right = r_target[0]
            else:
                objects = self.pipeline(left_frame, l_centers, right_frame, r_centers)

            self.results.add(T_WL[0].numpy(), T_WR[0].numpy(), objects, sequence.world_points)

            left_frame = left_frame.cpu().numpy()[0]
            right_frame = right_frame.cpu().numpy()[0]
            left_rgb = self._to_image(left_frame)
            right_rgb = self._to_image(right_frame)
            heatmap_left = self._to_heatmap(l_target[0].numpy())
            heatmap_right = self._to_heatmap(r_target[0].numpy())
            image_left = (0.3 * left_rgb + 0.7 * heatmap_left).astype(np.uint8)
            image_right = (0.3 * right_rgb + 0.7 * heatmap_right).astype(np.uint8)

            points_left = []
            points_right = []
            if self.flags.centers:
                for obj in objects:
                    if len(obj['centers_left']) > 0:
                        left = sequence.to_image_points(np.concatenate([c[None] for c in obj['centers_left']]))
                        points_left.append(left)
                    if len(obj['centers_right']) > 0:
                        right = sequence.to_image_points(np.concatenate([c[None] for c in obj['centers_right']]))
                        points_right.append(right)
            else:
                for obj in objects:
                    if self.flags.world:
                        p_LK = [p for p in obj['p_L'] if p is not None]
                        p_left = sequence.project_points_left(p_LK)
                        p_right = sequence.project_points_right(p_LK)
                    else:
                        p_left = np.concatenate([p for p in obj['keypoints_left'] if p.size != 0], axis=0)
                        p_right = np.concatenate([p for p in obj['keypoints_right'] if p.size != 0], axis=0)
                    p_left = sequence.to_image_points(p_left)
                    p_right = sequence.to_image_points(p_right)
                    points_left.append(p_left)
                    points_right.append(p_right)

            if self.flags.write is not None:
                self._write_frames(image_left, points_left, image_right, points_right)
            else:
                self.visualizer.set_left_image(image_left)
                self.visualizer.set_right_image(image_right)
                if len(points_left) > 0:
                    self.visualizer.set_left_points(points_left)
                if len(points_right) > 0:
                    self.visualizer.set_right_points(points_right)

                done = self.visualizer.update()
                if done:
                    exit()
            rate.sleep()

            self.frame_number += 1
        self.results.print_results()

    def run(self):
        random.seed(self.flags.seed)

        if self.flags.write:
            os.makedirs(self.flags.write, exist_ok=True)
        sequences = self._sequences()
        random.shuffle(sequences)
        for sequence in sequences:
            sequence = Sequence(self.flags, sequence)
            self._play_predictions(sequence)

if __name__ == "__main__":
    with torch.no_grad():
        Runner().run()

