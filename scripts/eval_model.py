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
from perception.utils import Rate
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
        self.model = KeypointModule.load_from_checkpoint(self.flags.model).cuda()

    def _sequences(self):
        return sorted([os.path.join(self.flags.data, s) for s in os.listdir(self.flags.data)])

    def _loader(self, dataset):
        return DataLoader(dataset, num_workers=1, batch_size=self.flags.batch_size)

    def _to_image(self, frame):
        frame = StereoVideoDataset.to_image(frame)
        return cv2.resize(frame, self.IMAGE_SIZE)

    def _to_heatmap(self, target):
        target = np.clip((target + 1.0) / 2.0, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)

    def _predict(self, frame):
        frame = frame.cuda()
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

    def _play_predictions(self, left_dataset, right_dataset):
        for i, ((left_frame, l_target), (right_frame, r_target)) in enumerate(zip(left_dataset, right_dataset)):
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
                    self.visualizer.set_left_image(image_left)
                    self.visualizer.set_right_image(image_right)

                    done = self.visualizer.update()
                    if done:
                        exit()

                    rate.sleep()

                self.frame_number += 1

    def run(self):
        sequences = self._sequences()
        for sequence in sequences:
            left_video = self._loader(StereoVideoDataset(sequence, camera=0, augment=False))
            right_video = self._loader(StereoVideoDataset(sequence, camera=1, augment=False))
            self._play_predictions(left_video, right_video)

if __name__ == "__main__":
    with torch.no_grad():
        Runner().run()

