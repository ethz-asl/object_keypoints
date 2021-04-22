import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import segmentation
from torchvision.models import mobilenetv3

class CenterHead(nn.Module):
    def __init__(self, output_size, in_channels, features_out):
        super().__init__()
        self.output_size = output_size
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True))
        self.output_conv = nn.Conv2d(32, features_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        vectors = self.layers(x)
        center_vectors = F.interpolate(vectors, self.output_size, mode='bilinear', align_corners=False)
        return self.output_conv(center_vectors)

class HeatmapHead(nn.Module):
    def __init__(self, output_size, in_channels, features_out):
        super().__init__()
        self.output_size = output_size
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        self.output_conv = nn.Conv2d(64, features_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.layers(x)
        x = F.interpolate(x, self.output_size, mode='bilinear', align_corners=False)
        return self.output_conv(x)

class KeypointNet(nn.Module):
    def __init__(self, output_size, heatmaps_out=2, regression_features=2):
        super().__init__()
        self.output_size = output_size
        backbone = mobilenetv3.mobilenet_v3_large(pretrained=True).features
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        low_pos = stage_indices[-3]
        low_channels = backbone[low_pos].out_channels

        self.backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): 'out'})
        self.heatmap_head = HeatmapHead(output_size, low_channels, heatmaps_out)
        self.center_head = CenterHead(output_size, low_channels, regression_features)

        # Initialize bias to account for class imbalance.
        self.heatmap_head.output_conv.bias.data.fill_(np.log(0.01/0.99))
        self.heatmap_head.output_conv.bias.data.fill_(np.log(0.01/0.99))

    def forward(self, x):
        backbone_out = self.backbone(x)['out']
        heatmaps = self.heatmap_head(backbone_out)
        center_vectors = self.center_head(backbone_out)
        return heatmaps, center_vectors


