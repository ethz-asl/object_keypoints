import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import segmentation
from torchvision.models import mobilenetv3
class CenterHead(nn.Module):
    def __init__(self, output_size, in_channels, features_out, extra_channels):
        super().__init__()
        self.output_size = output_size
        self.layers = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels, 32, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True))
        intermediate_channels = 32 + extra_channels
        self.intermediate_layers = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(intermediate_channels),
                nn.ReLU(inplace=True))
        self.output_conv = nn.Conv2d(intermediate_channels, features_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x, large):
        x = self.layers(x)
        x = F.interpolate(x, [45, 80], mode='bilinear', align_corners=False)
        x = self.intermediate_layers(torch.cat([x, large], dim=1))
        x = F.interpolate(x, self.output_size, mode='bilinear', align_corners=False)
        return self.output_conv(x)

class HeatmapHead(nn.Module):
    def __init__(self, output_size, in_channels, features_out, extra_channels):
        super().__init__()
        self.output_size = output_size
        self.layers = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channels, 128, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True))
        intermediate_channels = 128 + extra_channels
        self.intermediate_layers = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, dilation=2,
                    padding=2, bias=False),
                nn.BatchNorm2d(intermediate_channels),
                nn.ReLU(inplace=True))
        self.output_conv = nn.Sequential(
                nn.Conv2d(intermediate_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, features_out * 2, kernel_size=1, padding=0, bias=True))
        self.n_features = features_out

    def forward(self, x, large):
        x = self.layers(x)
        x = F.interpolate(x, [45, 80], mode='bilinear', align_corners=False)
        x = self.intermediate_layers(torch.cat([x, large], dim=1))
        x = F.interpolate(x, self.output_size, mode='bilinear', align_corners=False)
        x = self.output_conv(x)
        heatmap = x[:, :self.n_features]
        depth = F.relu(x[:, self.n_features:])
        return heatmap, depth

class KeypointNet(nn.Module):
    def __init__(self, output_size, heatmaps_out=2, regression_features=2):
        super().__init__()
        self.output_size = output_size
        backbone = mobilenetv3.mobilenet_v3_large(pretrained=True).features
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        backbone_last = stage_indices[-1]
        backbone_channels = backbone[backbone_last].out_channels
        intermediate_channels = backbone[stage_indices[-4]].out_channels

        self.backbone = IntermediateLayerGetter(backbone, return_layers={str(backbone_last): 'out',
            str(stage_indices[-4]): 'large'})
        self.heatmap_head = HeatmapHead(output_size, backbone_channels, heatmaps_out, intermediate_channels)
        self.center_head = CenterHead(output_size, backbone_channels, regression_features * (heatmaps_out-1), intermediate_channels)

    def forward(self, x):
        backbone_out = self.backbone(x)
        heatmaps, depth = self.heatmap_head(backbone_out['out'], backbone_out['large'])
        center_vectors = self.center_head(backbone_out['out'], backbone_out['large'])
        return heatmaps, depth, center_vectors


