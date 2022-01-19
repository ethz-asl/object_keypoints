import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from perception.corner_net_lite.core.models import CornerNet_Squeeze
from perception.corner_net_lite.core.models.py_utils.utils import convolution
from perception.corner_net_lite.core.base import load_nnet, load_cfg
from perception.corner_net_lite.core.config import SystemConfig
from perception.corner_net_lite.core.nnet.py_factory import NetworkFactory
import timm


def prediction_module(int_features, features_out):
    return nn.Sequential(
        convolution(1, 256, int_features, with_bn=True),
        convolution(1, int_features, 32, with_bn=True),
        nn.Conv2d(32, features_out, (1, 1), bias=True)
    )

class HeatmapHead(nn.Module):
    def __init__(self, features, heatmaps):
        super().__init__()
        self.output_head1 = prediction_module(features, heatmaps)
        self.output_head2 = prediction_module(features, heatmaps)
        self.output_head1[-1].bias.data.fill_(0.01/0.99)
        self.output_head2[-1].bias.data.fill_(0.01/0.99)

    def forward(self, heatmaps):
        return self.output_head1(heatmaps[0]), self.output_head2(heatmaps[1])

class DepthHead(nn.Module):
    def __init__(self, features, heatmaps):
        super().__init__()
        self.output_head1 = prediction_module(features, heatmaps)
        self.output_head2 = prediction_module(features, heatmaps)

    def forward(self, x):
        out1 = self.output_head1(x[0])
        out2 = self.output_head2(x[1])
        return out1, out2

class CenterHead(nn.Module):
    def __init__(self, features, heatmaps):
        super().__init__()
        self.outputs = heatmaps - 1
        self.output_head1 = prediction_module(features, self.outputs * 2)
        self.output_head2 = prediction_module(features, self.outputs * 2)

    def forward(self, x):
        N, C, H, W = x[1].shape
        out1 = self.output_head1(x[0])
        out2 = self.output_head2(x[1])
        return out1.reshape(N, self.outputs, 2, H, W), out2.reshape(N, self.outputs, 2, H, W)

def nms(x, size=5):
    hmax = nn.functional.max_pool2d(x, (size, size), padding=size // 2, stride=1)
    keep = (x == hmax).to(x.dtype)
    return x * keep

class KeypointNet(nn.Module):
    def __init__(self, output_size, features=128, heatmaps_out=2, dropout=0.1):
        super().__init__()
        self.backbone = self._build_hourglass()
        self.heatmap_head = HeatmapHead(features, heatmaps_out)
        self.depth_head = DepthHead(features, heatmaps_out)
        self.center_head = CenterHead(features, heatmaps_out)
        self.dropout = nn.Dropout(p=dropout)

    def _build_hourglass(self):
        corner_net = CornerNet_Squeeze.model()
        config, _ = load_cfg("./perception/corner_net_lite/configs/CornerNet_Squeeze.json")
        sys_cfg = SystemConfig().update_config(config)
        net = load_nnet(sys_cfg, corner_net)
        if torch.cuda.is_available():
            net.load_pretrained_params('./models/corner_net.pkl')
        else:
            print('Cuda not available. Will not load pretrained params')
        return net.model.module.hg

    def forward(self, x):
        features = [self.dropout(f) for f in self.backbone(x)]
        heatmaps_out = self.heatmap_head(features)
        depth_out = self.depth_head(features)
        centers_out = self.center_head(features)
        return heatmaps_out, depth_out, centers_out

