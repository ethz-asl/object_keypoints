import argparse
import os
import torch
import json
from train import KeypointModule
import yaml
from pathlib import Path

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

def load_hparams(path):
    version_dir = Path(path).parent.parent.absolute()
    with open(os.path.join(version_dir, 'hparams.yaml'), 'rt') as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return params

class Model(torch.nn.Module):
    def __init__(self, flags, hparams):
        super().__init__()
        self.model = KeypointModule.load_from_checkpoint(flags.model, **hparams).model

    def forward(self, x):
        heatmap, centers = self.model(x)
        N, D, two, H, W = centers[-1].shape
        return torch.sigmoid(heatmap[-1]), centers[-1]

def main():
    flags = read_args()
    hparams = load_hparams(flags.model)
    model = Model(flags, hparams).eval().cuda()

    dummy_input = torch.randn(2, 3, 511, 511).cuda()
    input_names = ["frames"]
    output_names = ["out"]

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

        traced.save(flags.out)


if __name__ == "__main__":
    main()
