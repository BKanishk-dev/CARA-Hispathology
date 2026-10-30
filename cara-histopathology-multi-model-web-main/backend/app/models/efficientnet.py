"""EfficientNet-B0 implemented from scratch (MBConv + squeeze-excitation + SiLU)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import NUM_CLASSES, BaseClassifier, init_weights


class SE(nn.Module):
    def __init__(self, ch: int, se_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(ch, se_ch, 1)
        self.expand = nn.Conv2d(se_ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3), keepdim=True)
        s = torch.sigmoid(self.expand(nn.functional.silu(self.reduce(s))))
        return x * s


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand, k, stride, drop_path=0.0):
        super().__init__()
        mid = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.drop_path = drop_path
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True)]
        layers += [nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
                   SE(mid, max(1, in_ch // 4)),
                   nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if not self.use_res:
            return out
        if self.training and self.drop_path > 0:                 # stochastic depth
            keep = 1.0 - self.drop_path
            mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep).to(x.dtype) / keep
            out = out * mask
        return x + out


# (expand, out_channels, repeats, stride, kernel) - EfficientNet-B0
_B0_CFG = [(1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5), (6, 80, 3, 2, 3),
           (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]


class EfficientNetB0(BaseClassifier):
    key = "efficientnet"

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2, drop_path: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU(inplace=True))
        total = sum(c[2] for c in _B0_CFG)
        blocks, in_ch, idx = [], 32, 0
        for expand, out_ch, repeats, stride, k in _B0_CFG:
            for r in range(repeats):
                dp = drop_path * idx / max(total - 1, 1)
                blocks.append(MBConv(in_ch, out_ch, expand, k, stride if r == 0 else 1, dp))
                in_ch, idx = out_ch, idx + 1
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(320, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.SiLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.blocks(self.stem(x)))
        return self.fc(self.drop(torch.flatten(self.pool(x), 1)))


def build_model(num_classes: int = NUM_CLASSES) -> EfficientNetB0:
    return EfficientNetB0(num_classes)
