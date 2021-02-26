import argparse
import os
import torch
from perception.datasets.video import StereoVideoDataset
from perception.datasets.utils import RoundRobin, SamplingPool
import numpy as np
from matplotlib import pyplot as plt
from albumentations.augmentations import transforms
import albumentations as A

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--workers', '-w', type=int, default=4, help="How many workers to use in data loader.")
    return parser.parse_args()

def _to_image(image):
    image = image.transpose([1, 2, 0])
    image = image * np.array([0.25, 0.25, 0.25])
    image = image + np.array([0.5, 0.5, 0.5])
    return np.clip((image * 255.0).round(), 0.0, 255.0).astype(np.uint8)

class Runner:
    def __init__(self):
        self.flags = read_args()
        self._load_datasets()

    def _load_datasets(self):
        datasets = []
        directories = os.listdir(self.flags.data)
        for directory in directories:
            path = os.path.join(self.flags.data, directory)
            dataset = StereoVideoDataset(path)
            datasets.append(dataset)
        self.train = SamplingPool(RoundRobin(datasets), 10)

    def run(self):
        dataloader = torch.utils.data.DataLoader(self.train,
                num_workers=self.flags.workers)
        for (left_frame, left_target), (right_frame, right_target) in dataloader:
            plt.figure(figsize=(14, 5))
            axis = plt.subplot2grid((1, 2), loc=(0, 0))
            axis.imshow(_to_image(left_frame[0].numpy()))
            axis.imshow(left_target[0].sum(axis=0).numpy(), alpha=0.7)
            plt.axis('off')
            axis = plt.subplot2grid((1, 2), loc=(0, 1))
            axis.imshow(_to_image(right_frame[0].numpy()))
            axis.imshow(right_target[0].sum(axis=0).numpy(), alpha=0.7)
            plt.axis('off')
            plt.tight_layout()
            plt.show()



if __name__ == "__main__":
    Runner().run()

