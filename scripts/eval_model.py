import argparse
import os
import hud
import time
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from perception.datasets.video import StereoVideoDataset
from train import KeypointModule
from matplotlib import cm

hud.set_data_directory(os.path.dirname(hud.__file__))

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="Path to dataset folder.")
    parser.add_argument('--model', '-m', type=str, required=True, help="Path to the model to evaluate.")
    parser.add_argument('--ground-truth', action='store_true', help="Show labels instead of making predictions.")
    return parser.parse_args()

class Visualizer:
    def __init__(self):
        self.done = False
        self.window = hud.AppWindow("Keypoints", 1280, 360)
        self._create_views()

    def _create_views(self):
        self.left_image_pane = hud.ImagePane()
        self.right_image_pane = hud.ImagePane()

        h_stack = hud.HStack()
        h_stack.add_view(self.left_image_pane)
        h_stack.add_view(self.right_image_pane)
        self.window.set_view(h_stack)
        self.window.add_key_handler(self._key_callback)

    def _key_callback(self, event):
        if event.key == 'Q':
            self.done = True

    def set_left_image(self, image):
        self.left_image_pane.set_texture(image)

    def set_right_image(self, image):
        self.right_image_pane.set_texture(image)

    def update(self):
        self.window.poll_events()
        if not self.window.update() or self.done:
            return True
        return False

class Runner:
    IMAGE_SIZE = (360, 640)
    def __init__(self):
        self.flags = read_args()
        self.visualizer = Visualizer()
        self._load_model()

    def _load_model(self):
        self.model = KeypointModule.load_from_checkpoint(self.flags.model).cuda()

    def _sequences(self):
        return sorted([os.path.join(self.flags.data, s) for s in os.listdir(self.flags.data)])

    def _loader(self, dataset):
        return DataLoader(dataset, num_workers=0, batch_size=1)

    def _to_image(self, frame):
        frame = StereoVideoDataset.to_image(frame[0].numpy())
        return cv2.resize(frame, self.IMAGE_SIZE)

    def _to_heatmap(self, target):
        target = np.clip((target + 1.0) / 2.0, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)
        return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)

    def _predict(self, frame):
        frame = frame.cuda()
        prediction = torch.tanh(self.model(frame)).cpu().numpy()
        return self._to_heatmap(prediction[0])

    def _play_predictions(self, left_dataset, right_dataset):
        for (left_frame, l_target), (right_frame, r_target) in zip(left_dataset, right_dataset):
            if self.flags.ground_truth:
                heatmap_left = self._to_heatmap(l_target[0])
                heatmap_right = self._to_heatmap(r_target[0])
            else:
                heatmap_left = self._predict(left_frame)
                heatmap_right = self._predict(right_frame)
            left_frame = self._to_image(left_frame)
            right_frame = self._to_image(right_frame)

            self.visualizer.set_left_image(0.3 * left_frame + 0.7 * heatmap_left)
            self.visualizer.set_right_image(0.3 * right_frame + 0.7 * heatmap_right)
            done = self.visualizer.update()
            if done:
                exit()

    def run(self):
        sequences = self._sequences()
        for sequence in sequences:
            left_video = self._loader(StereoVideoDataset(sequence, camera=0, augment=False))
            right_video = self._loader(StereoVideoDataset(sequence, camera=1, augment=False))
            self._play_predictions(left_video, right_video)

if __name__ == "__main__":
    with torch.no_grad():
        Runner().run()

