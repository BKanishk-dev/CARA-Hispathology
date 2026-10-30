"""ResNet18 implemented from scratch (BasicBlock x [2, 2, 2, 2])."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import NUM_CLASSES, BaseClassifier, init_weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResNet18(BaseClassifier):
    key = "resnet18"

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        cfg = [(64, 1), (128, 2), (256, 2), (512, 2)]
        layers, in_ch = [], 64
        for out_ch, stride in cfg:
            layers += [BasicBlock(in_ch, out_ch, stride), BasicBlock(out_ch, out_ch, 1)]
            in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(self.stem(x))
        x = torch.flatten(self.pool(x), 1)
        return self.fc(self.drop(x))


def build_model(num_classes: int = NUM_CLASSES) -> ResNet18:
    return ResNet18(num_classes)
