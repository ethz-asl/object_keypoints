import argparse
import os
import torch
import numpy as np
import json
import random
from matplotlib import pyplot as plt
from albumentations.augmentations import transforms
from perception.models import nms
import albumentations as A
from torch.utils.data import DataLoader
from perception.loss import KeypointLoss
from perception.datasets.video import StereoVideoDataset
from perception.models import KeypointNet
import pytorch_lightning as pl

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', '-w', type=int, default=8, help="How many workers to use in data loader.")
    parser.add_argument('--train', type=str, required=True, help="Path to training dataset.")
    parser.add_argument('--val', type=str, required=True, help="Path to validation dataset.")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help="Use half-precision.")
    parser.add_argument('--pool', default=1000, type=int, help="How many examples to use in shuffle pool")
    parser.add_argument('--keypoints', default="config/cups.json", help="Keypoint configuration file.")
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--features', default=128, type=int, help="Intermediate features in network.")
    parser.add_argument('--center-weight', default=1.0, help="Weight for center loss vs. heatmap loss.")
    parser.add_argument('--lr', default=4e-3, type=float, help="Learning rate.")
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--resume', default=None)
    return parser.parse_args()

def _to_image(image):
    image = image.transpose([1, 2, 0])
    image = image * np.array([0.25, 0.25, 0.25])
    image = image + np.array([0.5, 0.5, 0.5])
    return np.clip((image * 255.0).round(), 0.0, 255.0).astype(np.uint8)

def _init_worker(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)

class KeypointModule(pl.LightningModule):
    def __init__(self, keypoint_config, lr=3e-4, features=128, dropout=0.1, weight_decay=0.01, center_weight=10.0):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.keypoint_config = keypoint_config
        self._load_model(features, dropout)
        self.loss = KeypointLoss(keypoint_config['keypoint_config'], center_weight=center_weight)
        self.save_hyperparameters()

    def _load_model(self, features, dropout):
        self.model = KeypointNet([180, 320], features=features, dropout=dropout, heatmaps_out=len(self.keypoint_config["keypoint_config"]) + 1)

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
        heatmaps, p_centers = self(frame)

        loss = self._validation_loss(heatmaps, target, keypoints)
        val_loss, heatmap_losses, center_losses = self.loss(heatmaps, target, p_centers, gt_centers)

        self.log('val_loss', loss)
        self.log('total_heatmap_loss', val_loss)
        self.log('val_heatmap_loss1', heatmap_losses[0])
        self.log('val_heatmap_loss2', heatmap_losses[1])
        self.log('val_center_loss1', center_losses[0])
        self.log('val_center_loss2', center_losses[1])

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
        return {
            'scheduler': schedule,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'train_loss',
            'optimizer': optimizer
        }

    def _validation_loss(self, p_heatmaps, gt_heatmap, keypoints):
        # heatmaps: N x K x H x W
        # target: N x n_objects x K x 2
        p_heatmap = torch.sigmoid(p_heatmaps[-1])
        return torch.nn.functional.l1_loss(p_heatmap, gt_heatmap)

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
                train_datasets += _build_datasets(self.train_sequences, keypoint_config=self.keypoint_config, augment=True, augment_color=True, camera=camera)
            val_datasets = (_build_datasets(self.val_sequences, camera=0, keypoint_config=self.keypoint_config, augment=False, include_pose=True) +
                    _build_datasets(self.val_sequences, keypoint_config=self.keypoint_config, augment=False, camera=1, include_pose=True))
            train = torch.utils.data.ChainDataset(train_datasets)
            self.train = torch.utils.data.BufferedShuffleDataset(train, self.flags.pool)
            self.val = torch.utils.data.ChainDataset(val_datasets)
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
    if flags.resume is None:
        module = KeypointModule(keypoint_config,
                lr=flags.lr,
                center_weight=flags.center_weight,
                features=flags.features,
                dropout=flags.dropout,
                weight_decay=flags.weight_decay)
    else:
        module = KeypointModule.load_from_checkpoint(flags.resume,
                lr=flags.lr,
                center_weight=flags.center_weight,
                dropout=flags.dropout,
                weight_decay=flags.weight_decay)

    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(monitor='val_loss',
            save_top_k=1)
    trainer = pl.Trainer(
            callbacks=[checkpoint_cb],
            gpus=flags.gpus,
            reload_dataloaders_every_epoch=False,
            precision=16 if flags.fp16 else 32)

    trainer.fit(module, data_module)

if __name__ == "__main__":
    main()

