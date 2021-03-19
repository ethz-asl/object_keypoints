import argparse
import os
import hud
import time
import cv2
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from perception.datasets.video import StereoVideoDataset
from perception.utils import Rate, camera_utils, clustering_utils, linalg
from matplotlib import cm
from matplotlib import pyplot
from perception.pipeline import *

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="Path to dataset folder.")
    parser.add_argument('--model', '-m', type=str, required=True, help="Path to the model to evaluate.")
    parser.add_argument('--batch-size', '-b', type=int, default=1, help="Batch size used in data loader and inference.")
    parser.add_argument('--ground-truth', action='store_true', help="Show labels instead of making predictions.")
    parser.add_argument('--write', type=str, help="Write frames to folder.")
    return parser.parse_args()

class Sequence:
    def __init__(self, flags, sequence, prediction_size=(90, 160)):
        self.flags = flags
        self.sequence_path = sequence
        self.prediction_size = (90, 160)
        self.left_loader = self._loader(StereoVideoDataset(sequence, camera=0, augment=False, include_pose=True))
        self.right_loader = self._loader(StereoVideoDataset(sequence, camera=1, augment=False, include_pose=True))
        self._load_calibration()

        self.scaling_factor = np.array(self.image_size[::-1]) / np.array(self.prediction_size[::-1])

    def _loader(self, dataset):
        return DataLoader(dataset, num_workers=1, batch_size=self.flags.batch_size, pin_memory=True)

    def _load_calibration(self):
        calibration_file = os.path.join(self.sequence_path, 'calibration.yaml')
        with open(calibration_file, 'rt') as f:
            calibration = yaml.load(f.read(), Loader=yaml.SafeLoader)

        left = calibration['cam0']
        self.K = camera_utils.camera_matrix(left['intrinsics'])
        right = calibration['cam1']
        self.Kp = camera_utils.camera_matrix(right['intrinsics'])

        self.T_RL = np.array(calibration['cam1']['T_cn_cnm1'])
        self.image_size = calibration['cam1']['resolution'][::-1]

    def to_frame_points(self, p_WK, T_CW):
        """
        p_WK: N x 4 homogenous world coordinates
        Returns np.array N x 2 normalized device coordinate points
        """
        p_CK = T_CW @ p_WK[:, :, None]
        p_CK = (self.K @ np.eye(3, 4) @ p_CK)[:, :, 0]
        p_CK = p_CK / p_CK[:, 2:3]
        normalized = p_CK[:, :2] / (np.array([self.image_size[1], self.image_size[0]])[None, :] * 0.5) - 1.0
        normalized[:, 1] *= -1.0
        return normalized

    def project_points(self, p_WK, T_CW, K):
        image_points = K @ np.eye(3, 4) @ T_CW @ p_WK[:, :, None]
        image_points = image_points[:, :, 0]
        return (image_points / image_points[:, 2:3])[:, :2]


def _point_colors(n):
    return cm.jet(np.linspace(0.0, 1.0, n))

class Visualizer:
    def __init__(self):
        self.done = False
        self.window = hud.AppWindow("Keypoints", 1280, 360)
        self._create_views()
        self._point_colors = _point_colors(4)

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
        self.left_points.set_points(self._to_hud_points(points), self._point_colors)

    def set_right_points(self, points):
        self.right_points.set_points(self._to_hud_points(points), self._point_colors)

    def _to_hud_points(self, points):
        points = [hud.Point(p[0], p[1]) for p in points]
        return sorted(points, key=lambda p: p.y)

    def update(self):
        self.window.poll_events()
        if not self.window.update() or self.done:
            return True
        return False

class Runner:
    IMAGE_SIZE = (640, 360)
    def __init__(self):
        self.flags = read_args()
        if not self.flags.write:
            self.visualizer = Visualizer()
            self.figure = None
        else:
            self.visualizer = None
            self.figure = pyplot.figure(figsize=(16, 4.5))
        self.frame_number = 0
        self._setup_pipeline()

    def _setup_pipeline(self):
        self.pipeline = KeypointPipeline(self.flags.model, 3)

    def _sequences(self):
        return sorted([os.path.join(self.flags.data, s) for s in os.listdir(self.flags.data)])

    def _to_image(self, frame):
        frame = StereoVideoDataset.to_image(frame)
        return cv2.resize(frame, self.IMAGE_SIZE)

    def _to_heatmap(self, target):
        target = np.clip((target + 1.0) / 2.0, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)

    def _write_frames(self, left, left_points, right, right_points):
        x_scaling = left.shape[1] / 1280.0
        y_scaling = left.shape[0] / 720.0
        axis = pyplot.subplot2grid((1, 2), loc=(0, 0), fig=self.figure)
        axis.imshow(left)
        axis.scatter(left_points[:, 0] * x_scaling, left_points[:, 1] * y_scaling, s=5.0, color='y')
        axis.axis('off')
        axis = pyplot.subplot2grid((1, 2), loc=(0, 1), fig=self.figure)
        axis.imshow(right)
        axis.scatter(right_points[:, 0] * x_scaling, right_points[:, 1] * y_scaling, s=5.0, color='y')
        axis.axis('off')
        self.figure.savefig(os.path.join(self.flags.write, f'{self.frame_number:06}.jpg'), pil_kwargs={'quality': 85}, bbox_inches='tight')
        self.figure.clf()

    def _play_predictions(self, sequence):
        self.pipeline.reset(sequence.K, sequence.Kp, sequence.T_RL, sequence.scaling_factor)

        rate = Rate(60)
        for i, ((left_frame, l_target, T_WL), (right_frame, r_target, T_WR)) in enumerate(zip(sequence.left_loader, sequence.right_loader)):
            N = left_frame.shape[0]
            T_LW = np.linalg.inv(T_WL.numpy())
            T_RW = np.linalg.inv(T_WR.numpy())
            pipeline_out = self.pipeline(left_frame.cuda(), T_LW, right_frame.cuda(), T_WR)

            left_frame = left_frame.cpu().numpy()
            right_frame = right_frame.cpu().numpy()

            if self.flags.ground_truth:
                predictions_left = l_target
                predictions_right = r_target
            else:
                predictions_left = pipeline_out['heatmap_left']
                predictions_right = pipeline_out['heatmap_right']

            for i in range(min(left_frame.shape[0], right_frame.shape[0])):
                print(f"Frame {self.frame_number:06}", end='\r')
                left_rgb = self._to_image(left_frame[i])
                right_rgb = self._to_image(right_frame[i])
                heatmap_left = self._to_heatmap(predictions_left[i])
                heatmap_right = self._to_heatmap(predictions_right[i])

                image_left = (0.3 * left_rgb + 0.7 * heatmap_left).astype(np.uint8)
                image_right = (0.3 * right_rgb + 0.7 * heatmap_right).astype(np.uint8)

                p_WK = pipeline_out['keypoints_world'][i]

                if self.flags.write:
                    left_points = sequence.project_points(p_WK, T_LW, sequence.K)
                    right_points = sequence.project_points(p_WK, T_RW, sequence.Kp)
                    self._write_frames(image_left, left_points, image_right, right_points)
                else:
                    left_points = sequence.to_frame_points(p_WK, T_LW)
                    right_points = sequence.to_frame_points(p_WK, T_RW)

                    self.visualizer.set_left_image(image_left)
                    self.visualizer.set_left_points(left_points)
                    self.visualizer.set_right_image(image_right)
                    self.visualizer.set_right_points(right_points)

                    done = self.visualizer.update()
                    if done:
                        exit()

                    rate.sleep()

                self.frame_number += 1

    def run(self):
        if self.flags.write:
            os.makedirs(self.flags.write, exist_ok=True)
        sequences = self._sequences()
        import random
        random.shuffle(sequences)
        for sequence in sequences:
            sequence = Sequence(self.flags, sequence)
            self._play_predictions(sequence)

if __name__ == "__main__":
    with torch.no_grad():
        Runner().run()

