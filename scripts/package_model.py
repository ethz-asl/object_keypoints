import argparse
import os
import torch
import json
from train import KeypointModule

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--keypoint-config', type=str, default='./config/cups.json')
    return parser.parse_args()

class Model(torch.nn.Module):
    def __init__(self, flags):
        super().__init__()
        with open(flags.keypoint_config) as f:
            keypoint_config = json.load(f)
        self.model = KeypointModule.load_from_checkpoint(flags.model,
                keypoint_config=keypoint_config).model

    def forward(self, x):
        heatmap, depth, centers = self.model(x)
        return torch.sigmoid(heatmap), depth, centers

def main():
    flags = read_args()
    model = Model(flags).eval().half().cuda()

    dummy_input = torch.randn(2, 3, 360, 640).half().cuda()
    input_names = ["frames"]
    output_names = ["out"]

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

        traced.save(flags.out)


if __name__ == "__main__":
    main()
