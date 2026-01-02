from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import resnet18


@dataclass
class EncoderOut:
    feat_dim: int = 512


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 without the classification head.
    Returns features of shape [B, 512].
    """

    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool
        )
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.pool = m.avgpool
        self.feat_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)               # [B, 512, 1, 1]
        x = torch.flatten(x, 1)        # [B, 512]
        return x
