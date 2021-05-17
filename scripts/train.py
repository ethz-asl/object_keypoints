import argparse
import os
import torch
import numpy as np
import json
from matplotlib import pyplot as plt
from albumentations.augmentations import transforms
import albumentations as A
from torch.utils.data import DataLoader
from perception.loss import KeypointLoss
from perception.datasets.video import StereoVideoDataset
from perception.datasets.utils import RoundRobin, SamplingPool, Chain
from perception.models import KeypointNet
import pytorch_lightning as pl

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', '-w', type=int, default=8, help="How many workers to use in data loader.")
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--train', type=str, required=True, help="Path to training dataset.")
    parser.add_argument('--val', type=str, required=True, help="Path to validation dataset.")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help="Use half-precision.")
    parser.add_argument('--pool', default=1000, type=int, help="How many examples to use in shuffle pool")
    parser.add_argument('--keypoints', default="config/cups.json", help="Keypoint configuration file.")
    return parser.parse_args()

def _to_image(image):
    image = image.transpose([1, 2, 0])
    image = image * np.array([0.25, 0.25, 0.25])
    image = image + np.array([0.5, 0.5, 0.5])
    return np.clip((image * 255.0).round(), 0.0, 255.0).astype(np.uint8)

def _init_worker(worker_id):
    np.random.seed(worker_id)

class KeypointModule(pl.LightningModule):
    def __init__(self, keypoint_config):
        super().__init__()
        self.keypoint_config = keypoint_config
        self._load_model()
        self.loss = KeypointLoss(keypoint_config['keypoint_config'])

    def _load_model(self):
        self.model = KeypointNet([180, 320], heatmaps_out=len(self.keypoint_config["keypoint_config"]) + 1)

    def forward(self, frame, *args, **kwargs):
        return self.model(frame, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        frame, target, gt_centers = batch
        heatmaps, p_centers = self(frame)

        loss, heatmap_losses, center_losses = self.loss(heatmaps, target, p_centers, gt_centers)

        self.log('train_loss', loss)
        self.log('heatmap_loss1', heatmap_losses[0])
        self.log('heatmap_loss2', heatmap_losses[1])
        self.log('center_loss1', center_losses[0])
        self.log('center_loss2', center_losses[1])

        return loss

    def validation_step(self, batch, batch_idx):
        frame, target, gt_centers, _, keypoints = batch
        heatmaps, p_centers = self(frame, train=False)

        loss = self._validation_loss(heatmaps, keypoints)
        loss, heatmap_losses, center_losses = self.loss(heatmaps, target, p_centers, gt_centers)

        self.log('val_loss', loss)
        self.log('val_heatmap_loss1', heatmap_losses[0])
        self.log('val_heatmap_loss2', heatmap_losses[1])
        self.log('val_center_loss1', center_losses[0])
        self.log('val_center_loss2', center_losses[1])

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def _validation_loss(self, heatmaps, keypoints):
        # heatmaps: N x K x H x W
        # target: N x n_objects x K x 2
        heatmaps = heatmaps[-1]
        N, K, H, W = heatmaps.shape

        true_positives = 0
        false_positives = 0
        total = 0
        for i in range(N):
            for k in range(K):
                heatmap = heatmaps[i, k].detach().cpu().numpy()
                actual = keypoints[i, :, k].detach().cpu().numpy() # O x 2
                predicted = np.argwhere(heatmap > 0.5) # D x 2
                if predicted.shape[0] == 0:
                    continue
                dima = np.linalg.norm(actual[:, None] - predicted[None], 2, axis=2)
                # O x D
                detected = dima.min(axis=1) <= 1.0
                detected = detected.sum()
                true_positives += detected
                false_positives += (dima.min(axis=1) > 1.0).sum()
                total += actual.shape[0]
        if total == 0:
            return 1e7
        true_positive = float(true_positives) / float(total)
        false_positive = float(false_positives) / float(total)
        self.log('val_true_positive', true_positive)
        self.log('val_false_positive', false_positive)
        return np.abs(1.0 - true_positive) + false_positive

def _build_datasets(sequences, **kwargs):
    datasets = []
    for sequence in sequences:
        dataset = StereoVideoDataset(sequence, **kwargs)
        datasets.append(dataset)
    return datasets

class DataModule(pl.LightningDataModule):
    def __init__(self, flags, keypoint_config):
        super().__init__()
        self.keypoint_config = keypoint_config
        datasets = []
        train_directories = os.listdir(flags.train)
        train_sequences = sorted([os.path.join(flags.train, d) for d in train_directories])
        val_directories = os.listdir(flags.val)
        val_sequences = sorted([os.path.join(flags.val, d) for d in val_directories])
        self.flags = flags
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences

    def setup(self, stage):
        if stage == 'fit':
            train_datasets = []
            for camera in [0, 1]:
                train_datasets += _build_datasets(self.train_sequences, keypoint_config=self.keypoint_config, augment=True, camera=camera)
            val_datasets = (_build_datasets(self.val_sequences, keypoint_config=self.keypoint_config, augment=False, include_pose=True) +
                    _build_datasets(self.val_sequences, keypoint_config=self.keypoint_config, augment=False, camera=1, include_pose=True))
            self.train = SamplingPool(Chain(train_datasets, shuffle=True, infinite=True), self.flags.pool)
            self.val = Chain(val_datasets, shuffle=True)
        else:
            raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.flags.batch_size, num_workers=self.flags.workers,
                worker_init_fn=_init_worker,
                persistent_workers=self.flags.workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.flags.batch_size * 2, num_workers=self.flags.workers)

def main():
    flags = read_args()
    with open(flags.keypoints) as f:
        keypoint_config = json.load(f)
    data_module = DataModule(flags, keypoint_config)
    module = KeypointModule(keypoint_config)

    trainer = pl.Trainer(
            gpus=flags.gpus,
            reload_dataloaders_every_epoch=False,
            precision=16 if flags.fp16 else 32)

    trainer.fit(module, data_module)

if __name__ == "__main__":
    main()

