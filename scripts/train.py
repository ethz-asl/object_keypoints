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

    def forward(self, frame):
        return self.model(frame)

    def training_step(self, batch, batch_idx):
        frame, target, gt_depth, gt_centers = batch
        heatmaps, p_depth, centers = self(frame)

        loss, losses = self.loss(heatmaps, target, p_depth, gt_depth, centers, gt_centers)

        self.log('train_loss', loss)
        self.log('heatmap_loss', losses[0])
        self.log('depth_loss', losses[1])
        self.log('center_loss', losses[2])

        return loss

    def validation_step(self, batch, batch_idx):
        frame, target, gt_depth, gt_centers = batch
        y_hat, p_depth, centers = self(frame)

        loss, losses = self.loss(y_hat, target, p_depth, gt_depth, centers, gt_centers)

        self.log('val_loss', loss)
        self.log('heatmap_loss_val', losses[0])
        self.log('depth_loss_val', losses[1])
        self.log('center_loss_val', losses[2])

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)


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
            val_datasets = (_build_datasets(self.val_sequences, keypoint_config=self.keypoint_config, augment=False) +
                    _build_datasets(self.val_sequences, keypoint_config=self.keypoint_config, augment=False, camera=1))
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

