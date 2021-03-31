import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import segmentation
from torchvision.models import mobilenetv3
from torchvision.models.segmentation.segmentation import _segm_lraspp_mobilenetv3

class KeypointNet(nn.Module):
    def __init__(self, out_features=2):
        super().__init__()
        self.model = _segm_lraspp_mobilenetv3('mobilenet_v3_large', num_classes=out_features,
                pretrained_backbone=True)

    def forward(self, x):
        return self.model(x)['out']

