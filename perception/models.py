import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from efficientnet_pytorch import EfficientNet

class KeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b2', include_top=False)
        self.features = 1408
        self._deconvs = self._build_deconv_layers([512, 256, 128])
        self.out_conv = nn.Conv2d(in_channels=128,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=True,
                padding=0)

    def _build_deconv_layers(self, features):
        layers = []
        in_features = self.features
        for i in range(len(features)):
            out_features = features[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(out_features))
            layers.append(nn.ReLU(inplace=True))

            if i == 1:
                # With input 360x640 we get output 48 x 160 at this point.
                # Correct to get proper output aspect ratio.
                layers.append(nn.UpsamplingBilinear2d(size=(45, 80)))
            in_features = out_features

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self._deconvs(x)
        x = self.out_conv(x)
        return x
