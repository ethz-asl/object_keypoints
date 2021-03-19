import argparse
import os
import torch
from train import KeypointModule

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--out', type=str)
    return parser.parse_args()

class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = KeypointModule.load_from_checkpoint(model).model
        self.model.backbone.set_swish(memory_efficient=False)

    def forward(self, x):
        return torch.tanh(self.model(x))

def main():
    flags = read_args()
    model = Model(flags.model).eval().half().cuda()

    dummy_input = torch.randn(2, 3, 360, 640).half().cuda()
    input_names = ["frames"]
    output_names = ["out"]

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

        traced.save(flags.out)


if __name__ == "__main__":
    main()
