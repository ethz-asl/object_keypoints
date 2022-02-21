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
from perception.datasets.video import SceneDataset
from perception.utils import Rate, camera_utils, clustering_utils, linalg
from perception.models import nms
from matplotlib import cm
from matplotlib import pyplot
from perception.pipeline import *
from rich.console import Console
from rich.table import Table

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
    parser.add_argument('--debug', action='store_true', help="Does not use background worker in dataloader.")
    return parser.parse_args()

IMAGE_RECT = hud.Rect(0.0, 0.0, 511, 511.0)
class Sequence:
    def __init__(self, flags, sequence):
        self.flags = flags
        self.sequence_path = sequence
        self.prediction_size = SceneDataset.prediction_size
        with open(flags.keypoints, 'rt') as f:
            self.keypoint_config = json.load(f)
        self.dataset = SceneDataset(sequence, self.keypoint_config, augment=False, include_pose=True)
        self.dataloader = self._loader(self.dataset)
        self.size_resized = np.array([SceneDataset.height_resized, SceneDataset.width_resized])
        self.scaling_factor = self.prediction_size / self.size_resized
        self.image_offset = SceneDataset.image_offset
        self.scale_prediction_to_image = self.prediction_size / self.size_resized

        self._load_calibration()
        self._read_keypoints()

    def _loader(self, dataset):
        return DataLoader(dataset, num_workers=0 if self.flags.debug else 1, batch_size=1, pin_memory=not self.flags.cpu and torch.cuda.is_available() and not self.flags.ground_truth)

    def _read_keypoints(self):
        self.world_points = self.dataset.world_points.reshape(self.dataset.n_objects, self.dataset.n_keypoints, 3)
        filepath = os.path.join(self.sequence_path, 'keypoints.json')
        with open(filepath, 'rt') as f:
            self.keypoints = np.array(json.loads(f.read())['3d_points'])[:, :3]

    def _load_calibration(self):
        calibration_file = os.path.join(self.sequence_path, 'calibration.yaml')
        camera = camera_utils.from_calibration(calibration_file)
        camera = camera.scale(SceneDataset.height_resized / SceneDataset.height)
        self.camera = camera.cut(self.image_offset)

        scale_small = self.prediction_size[0] / SceneDataset.height_resized
        self.camera_small = camera.cut(self.image_offset).scale(scale_small)

    def to_image_points(self, predictions):
        # predictions: points in 2d prediction space.
        return predictions / self.scale_prediction_to_image

    def project_points(self, p_CK):
        p_CK = np.concatenate(p_CK, axis=0)
        return self.camera.project(p_CK) + 0.5


object_color_maps = [cm.get_cmap('Reds'), cm.get_cmap('Purples'), cm.get_cmap('Greens'), cm.get_cmap('Blues'), cm.get_cmap('Oranges')]
n_colormaps = len(object_color_maps)
def _point_colors(object_points):
    # Color per object.
    colors = []
    for i, points in enumerate(object_points):
        colors.append(object_color_maps[i % n_colormaps](np.linspace(0.7, 1.0, points.shape[0])))
    return np.concatenate(colors)

class Visualizer:
    def __init__(self):
        self.done = False
        self.window = hud.AppWindow("Keypoints", 640, 640)
        self._create_views()

    def _create_views(self):
        self.image_pane = hud.ImagePane()
        self.points = hud.PointLayer([])

        stack = hud.ZStack()
        stack.add_view(self.image_pane)
        stack.add_view(self.points)

        self.window.set_view(stack)
        self.window.add_key_handler(self._key_callback)

    def _key_callback(self, event):
        if event.key == 'Q':
            self.done = True

    def set_image(self, image):
        self.image_pane.set_texture(image)

    def set_points(self, points):
        colors = _point_colors(points)
        points = np.concatenate(points)
        points = self._to_hud_points(points)
        hud_points = [hud.utils.to_normalized_device_coordinates(p, IMAGE_RECT) for p in points]
        self.points.set_points(hud_points, colors)

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
        self.console = Console()
        self.screen = self.console.screen()
        self.camera = None

    def add(self, T_WC, objects, scene_points):
        """
        T_WC: transform from camera frame to world
        objects: object detections
        keypoints: n_objects X n_keypoints X 3 gt keypoints
        """
        gt_keypoints = []
        keypoints = []
        T_CW = linalg.inv_transform(T_WC)

        scene_points_C = linalg.transform_points(T_CW, scene_points)
        centers_C = scene_points_C[:, 0]
        for obj in objects:
            p_CK = obj['p_C']
            # Disregard depth as that is likely off by a bit.
            object_distances = np.linalg.norm(centers_C[:, :2] - p_CK[0][0][:2], axis=1)
            closest_object = object_distances.argmin()
            object_points = scene_points_C[closest_object]
            # print("d: ", object_distances[closest_object])

            gt_center = self.camera.project(object_points[0:1])
            if not self.camera.in_frame(gt_center)[0]:
                # Object center is not in view. Skipping this object.
                print("Object center not in view.")
                continue

            gt_points = []
            object_keypoints = []
            for i, points in enumerate(p_CK):
                if points is None:
                    continue
                for point in points:
                    if point is not None and (point < 2.0).all():
                        closest_point = np.linalg.norm(object_points - point, axis=1).argmin()
                        gt_point_L = object_points[closest_point]
                        gt_point_projected = self.camera.project(gt_point_L[None])

                        if (self.camera.in_frame(gt_point_projected) == False).any():
                            print('point not in view')
                            continue

                        print('diff: ', (gt_point_L - point).round(2), end='\r')
                        object_keypoints.append(point)
                        gt_points.append(gt_point_L)
                    else:
                        object_keypoints.append(None)
                        gt_points.append(None)
            gt_keypoints.append(gt_points)
            keypoints.append(object_keypoints)
        self.gt_keypoints.append(gt_keypoints)
        self.predicted_keypoints.append(keypoints)

    def set_calibration(self, camera):
        self.camera = camera

    def print_results(self):
        errors = []
        errors_xy = []
        missing = 0
        n_points = 0
        small_error = 0
        for gt, predicted in zip(self.gt_keypoints, self.predicted_keypoints):
            assert len(gt) == len(predicted)
            for gt_points, p_points in zip(gt, predicted):
                assert len(gt_points) == len(p_points)
                for gt_point, p_point in zip(gt_points, p_points):
                    n_points += 1
                    if p_point is not None:
                        assert gt_point.shape == p_point.shape
                        error = np.linalg.norm(gt_point - p_point, 2, axis=0)
                        error_xy = np.linalg.norm(gt_point[:2] - p_point[:2], 2, axis=0)
                        assert error.size == 1
                        errors.append(error)
                        errors_xy.append(error_xy)
                        if error < 0.03:
                            small_error += 1
                    else:
                        missing += 1
        table = Table(show_header=True)
        table.add_column("mean")
        table.add_column("mean xy")
        table.add_column("std")
        table.add_column("< 3cm")
        table.add_column("25th percentile")
        table.add_column("75th percentile")
        table.add_column("missing")
        table.add_column("points")
        errors = np.array(errors) * 100.0 # convert to cm
        errors_xy = np.array(errors_xy) * 100.0
        small = float(small_error) / float(n_points)
        percentile25 = np.percentile(errors, 25)
        percentile75 = np.percentile(errors, 75)
        missing_percentage = (float(missing) / float(n_points)) * 100.0
        table.add_row(f"{errors.mean()}", f"{errors_xy.mean()}", f"{errors.std()}",
                f"{small}", f"{percentile25}", f"{percentile75}", f"{missing_percentage:.02f}%", f"{n_points}")
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
        frame = SceneDataset.to_image(frame)
        return cv2.resize(frame, self.IMAGE_SIZE)

    def _to_heatmap(self, target):
        target = np.clip(target, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)

    def _write_frames(self, left, left_points):
        x_scaling = left.shape[1] / 1280.0
        y_scaling = left.shape[0] / 720.0
        axis = pyplot.subplot2grid((1, 1), loc=(0, 0), fig=self.figure)
        axis.imshow(left)
        if len(left_points):
            c = _point_colors(left_points)
            left_points = np.concatenate(left_points, axis=0)
            axis.scatter(left_points[:, 0], left_points[:, 1], s=5.0, color=c)
        axis.axis('off')
        self.figure.savefig(os.path.join(self.flags.write, f'{self.frame_number:06}.jpg'), pil_kwargs={'quality': 85}, bbox_inches='tight')
        self.figure.clf()

    def _play_predictions(self, sequence):
        if self.flags.ground_truth:
            self.pipeline = ObjectKeypointPipeline(sequence.prediction_size, sequence.keypoints, sequence.keypoint_config)
        else:
            self.pipeline = LearnedKeypointTrackingPipeline(self.flags.model, not self.flags.cpu and torch.cuda.is_available(),
                    sequence.prediction_size, sequence.keypoint_config)
        self.pipeline.reset(sequence.camera_small)
        self.results.set_calibration(sequence.camera_small)

        rate = Rate(30)
        for i, (frame, target, depth, centers, T_WC, keypoints) in enumerate(sequence.dataloader):
            N = frame.shape[0]
            if self.flags.ground_truth:
                objects = self.pipeline(target, depth, centers)
                heatmap = self._to_heatmap(target[0].numpy())
            else:
                objects, heatmap = self.pipeline(frame)
                heatmap = self._to_heatmap(heatmap[0].numpy())

            self.results.add(T_WC[0].numpy(), objects, sequence.world_points)

            frame = frame.cpu().numpy()[0]
            rgb = self._to_image(frame)
            image = (0.3 * rgb + 0.7 * heatmap).astype(np.uint8)

            image_points = []
            if self.flags.centers:
                for obj in objects:
                    if len(obj['p_centers']) > 0:
                        left = sequence.to_image_points(np.concatenate([c[None] for c in obj['p_centers']]))
                        image_points.append(left)
            else:
                for obj in objects:
                    if self.flags.world:
                        p_CK = [[a for a in p if a is not None] for p in obj['p_C'] if p is not None]
                        p_points = sequence.project_points(p_CK)
                    else:
                        p_points = np.concatenate([p + 1.0 for p in obj['keypoints'] if p.size != 0], axis=0)
                        p_points = sequence.to_image_points(p_points)
                    image_points.append(p_points)

            if self.flags.write is not None:
                self._write_frames(image, image_points)
            else:
                self.visualizer.set_image(image)

                if len(image_points) > 0:
                    self.visualizer.set_points(image_points)

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

