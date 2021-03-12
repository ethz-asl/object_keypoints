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
from train import KeypointModule
from matplotlib import cm
from matplotlib import pyplot

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="Path to dataset folder.")
    parser.add_argument('--model', '-m', type=str, required=True, help="Path to the model to evaluate.")
    parser.add_argument('--batch-size', '-b', type=int, default=1, help="Batch size used in data loader and inference.")
    parser.add_argument('--ground-truth', action='store_true', help="Show labels instead of making predictions.")
    parser.add_argument('--write', type=str, help="Write frames to folder.")
    return parser.parse_args()

PROBABILITY_CUTOFF = 0.1

class Sequence:
    def __init__(self, flags, sequence, prediction_size=(90, 160)):
        self.flags = flags
        self.sequence_path = sequence
        self.prediction_size = (90, 160)
        self.left_loader = self._loader(StereoVideoDataset(sequence, camera=0, augment=False, include_pose=True))
        self.right_loader = self._loader(StereoVideoDataset(sequence, camera=1, augment=False, include_pose=True))
        self._load_calibration()
        self.indices = np.zeros((*prediction_size, 2), dtype=np.float32)
        for y in range(prediction_size[0]):
            for x in range(prediction_size[1]):
                self.indices[y, x, 0] = np.float32(x) + 1.0
                self.indices[y, x, 1] = np.float32(y) + 1.0

        self.prediction_to_image_scaling_factor = np.array(self.image_size[::-1]) / np.array(self.prediction_size[::-1])
        self.clustering_left = clustering_utils.KeypointClustering()
        self.clustering_right = clustering_utils.KeypointClustering()

    def _loader(self, dataset):
        return DataLoader(dataset, num_workers=1, batch_size=self.flags.batch_size)

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

    def _find_keypoints(self, predictions, left=True):
        predictions += 1.0
        predictions *= 0.5
        keypoints = np.zeros((4, 2), dtype=predictions.dtype)
        # Center point
        center_point = predictions[0]
        where_larger = center_point > PROBABILITY_CUTOFF
        left_center = self.indices[where_larger, :]
        probabilities_center = center_point[where_larger]
        probabilities_center /= probabilities_center.sum()
        keypoints[0, :] = (left_center * probabilities_center[:, None]).sum(axis=0)

        # Three spoke points.
        spokes = predictions[1]
        where_larger = spokes > PROBABILITY_CUTOFF
        spoke_indices = self.indices[where_larger, :]
        probabilities_spoke = spokes[where_larger]
        if left:
            clusters = self.clustering_left(spoke_indices, probabilities_spoke)
        else:
            clusters = self.clustering_right(spoke_indices, probabilities_spoke)
        keypoints[1:1+clusters.shape[0], :] = clusters
        return keypoints

    def _associate_keypoints(self, left_keypoints, right_keypoints):
        R = self.T_RL[:3, :3]
        t = self.T_RL[:3, 3]
        Kp_inv = np.linalg.inv(self.Kp)

        C = linalg.skew_matrix(self.K @ R.T @ t)
        F = Kp_inv.T @ R @ self.K.T @ C

        distances = np.zeros((left_keypoints.shape[0], right_keypoints.shape[0]), dtype=left_keypoints.dtype)
        one = np.ones((1,), dtype=left_keypoints.dtype)
        for i in range(left_keypoints.shape[0]):
            v1 = np.concatenate([left_keypoints[i, :], one], axis=0)[:, None]
            for j in range(right_keypoints.shape[0]):
                v2 = np.concatenate([right_keypoints[j, :], one], axis=0)[:, None]
                distances[i, j] = np.abs(v2.T @ F @ v1)
        return distances.argmin(axis=1)

    def triangulate(self, prediction_left, prediction_right, T_LW, T_RW):
        left_keypoints = self._find_keypoints(prediction_left, True)
        right_keypoints = self._find_keypoints(prediction_right, False)
        left_keypoints = self._scale_points(left_keypoints)
        right_keypoints = self._scale_points(right_keypoints)
        associations = self._associate_keypoints(left_keypoints, right_keypoints)
        # Permute keypoints into the same order as left.
        right_keypoints = right_keypoints[associations]

        P1 = camera_utils.projection_matrix(self.K, np.eye(4))
        P2 = self.Kp @ np.eye(3, 4) @ self.T_RL
        p_LK = cv2.triangulatePoints(P1, P2, left_keypoints.T, right_keypoints.T).T # N x 4
        p_LK = p_LK / p_LK[:, 3:4]
        p_WK = (np.linalg.inv(T_LW) @ p_LK[:, :, None])[:, :, 0]
        p_WK /= p_WK[:, 3:4]
        return p_WK

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

    def _scale_points(self, points):
        # points: N x 2 image points on the prediction heatmap.
        # returns: N x 2 image points scaled to the full image size.
        return points * self.prediction_to_image_scaling_factor



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
        self._load_model()
        self.frame_number = 0

    def _load_model(self):
        self.model = KeypointModule.load_from_checkpoint(self.flags.model)

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

    def _predict(self, frame):
        return torch.tanh(self.model(frame)).cpu().numpy()

    def _write_frames(self, left, right):
        axis = pyplot.subplot2grid((1, 2), loc=(0, 0), fig=self.figure)
        axis.imshow(left)
        axis.axis('off')
        axis = pyplot.subplot2grid((1, 2), loc=(0, 1), fig=self.figure)
        axis.imshow(right)
        axis.axis('off')
        self.figure.savefig(os.path.join(self.flags.write, f'{self.frame_number:06}.jpg'), pil_kwargs={'quality': 85}, bbox_inches='tight')
        self.figure.clf()

    def _play_predictions(self, sequence):
        for i, ((left_frame, l_target, T_WL), (right_frame, r_target, T_WR)) in enumerate(zip(sequence.left_loader, sequence.right_loader)):
            if self.flags.ground_truth:
                predictions_left = l_target
                predictions_right = r_target
            else:
                predictions_left = self._predict(left_frame)
                predictions_right = self._predict(right_frame)
            rate = Rate(60)
            left_frame = left_frame.cpu().numpy()
            right_frame = right_frame.cpu().numpy()
            for i in range(min(left_frame.shape[0], right_frame.shape[0])):
                print(f"Frame {self.frame_number:06}", end='\r')
                left_rgb = self._to_image(left_frame[i])
                right_rgb = self._to_image(right_frame[i])
                heatmap_left = self._to_heatmap(predictions_left[i])
                heatmap_right = self._to_heatmap(predictions_right[i])

                image_left = (0.3 * left_rgb + 0.7 * heatmap_left).astype(np.uint8)
                image_right = (0.3 * right_rgb + 0.7 * heatmap_right).astype(np.uint8)

                if self.flags.write:
                    self._write_frames(image_left, image_right)
                else:
                    T_LW = np.linalg.inv(T_WL[i].numpy())
                    T_RW = np.linalg.inv(T_WR[i].numpy())
                    p_WK = sequence.triangulate(predictions_left[i], predictions_right[i],
                            T_LW,
                            T_RW)
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
        sequences = self._sequences()
        for sequence in sequences:
            sequence = Sequence(self.flags, sequence)
            self._play_predictions(sequence)

if __name__ == "__main__":
    with torch.no_grad():
        Runner().run()

