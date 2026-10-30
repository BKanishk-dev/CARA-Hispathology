"""MobileNetV3-Small implemented from scratch (inverted residuals + SE + h-swish)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import NUM_CLASSES, BaseClassifier, init_weights


def make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        sq = make_divisible(ch // reduction)
        self.fc1 = nn.Conv2d(ch, sq, 1)
        self.fc2 = nn.Conv2d(sq, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3), keepdim=True)
        s = torch.relu(self.fc1(s))
        s = torch.nn.functional.hardsigmoid(self.fc2(s))
        return x * s


def conv_bn_act(in_ch, out_ch, k, stride=1, groups=1, act="hs"):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, (k - 1) // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act == "hs":
        layers.append(nn.Hardswish(inplace=True))
    elif act == "re":
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, k, exp_ch, out_ch, use_se, act, stride):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if exp_ch != in_ch:
            layers.append(conv_bn_act(in_ch, exp_ch, 1, act=act))
        layers.append(conv_bn_act(exp_ch, exp_ch, k, stride, groups=exp_ch, act=act))
        if use_se:
            layers.append(SqueezeExcite(exp_ch))
        layers.append(conv_bn_act(exp_ch, out_ch, 1, act=None))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_res else out


# (in, kernel, expanded, out, SE, activation, stride) - MobileNetV3-Small table
_SMALL_CFG = [
    (16, 3, 16, 16, True, "re", 2),
    (16, 3, 72, 24, False, "re", 2),
    (24, 3, 88, 24, False, "re", 1),
    (24, 5, 96, 40, True, "hs", 2),
    (40, 5, 240, 40, True, "hs", 1),
    (40, 5, 240, 40, True, "hs", 1),
    (40, 5, 120, 48, True, "hs", 1),
    (48, 5, 144, 48, True, "hs", 1),
    (48, 5, 288, 96, True, "hs", 2),
    (96, 5, 576, 96, True, "hs", 1),
    (96, 5, 576, 96, True, "hs", 1),
]


class MobileNetV3Small(BaseClassifier):
    key = "mobilenetv3"

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2):
        super().__init__()
        self.stem = conv_bn_act(3, 16, 3, 2, act="hs")
        self.blocks = nn.Sequential(*[InvertedResidual(*c) for c in _SMALL_CFG])
        self.head_conv = conv_bn_act(96, 576, 1, act="hs")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024), nn.Hardswish(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, num_classes))
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head_conv(self.blocks(self.stem(x)))
        return self.classifier(torch.flatten(self.pool(x), 1))


def build_model(num_classes: int = NUM_CLASSES) -> MobileNetV3Small:
    return MobileNetV3Small(num_classes)
