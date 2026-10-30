"""Capsule Network with primary capsules and dynamic routing-by-agreement (Sabour et al., 2017).

Adapted to 256x256 histopathology:
  conv stem (256 -> 16x16, 128 ch)  ->  PrimaryCaps (16 types x 8-D, 8x8 grid = 1024 capsules)
  -> DigitCaps: 19 class capsules x 16-D, 3 routing iterations.
The prediction-vector transform W_ij is shared across grid positions of the same capsule type
(a standard trick that keeps the parameter count reasonable for large inputs).

forward() returns logits = scale * ||v_k||  (capsule lengths scaled by a learnable temperature) so the
model plugs into the same softmax / CARA pipeline as every other model.  The margin loss of the
original paper is exposed through `extra_loss` and used by train_all.py in addition to CE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import NUM_CLASSES, BaseClassifier, init_weights


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    n2 = (s * s).sum(dim, keepdim=True)
    return (n2 / (1.0 + n2)) * s / torch.sqrt(n2 + eps)


class CapsNet(BaseClassifier):
    key = "capsnet"

    def __init__(self, num_classes: int = NUM_CLASSES, n_types: int = 16, prim_dim: int = 8,
                 out_dim: int = 16, routing_iters: int = 3):
        super().__init__()
        self.num_classes, self.n_types, self.prim_dim = num_classes, n_types, prim_dim
        self.routing_iters = routing_iters

        def cb(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.stem = nn.Sequential(cb(3, 32, 5, 2), cb(32, 64, 3, 2), cb(64, 128, 3, 2), cb(128, 128, 3, 2))  # 256->16
        self.primary = nn.Conv2d(128, n_types * prim_dim, 3, 2, 1)                                           # 16 -> 8
        # W[type, class, out_dim, prim_dim], shared across the 8x8 positions of a type
        self.W = nn.Parameter(torch.randn(n_types, num_classes, out_dim, prim_dim) * 0.05)
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        init_weights(self.stem)

    def route(self, u_hat: torch.Tensor) -> torch.Tensor:
        """u_hat: (B, N_in, K, D) -> class capsules v: (B, K, D) by dynamic routing."""
        b = torch.zeros(u_hat.shape[:3], device=u_hat.device, dtype=u_hat.dtype)       # B, N, K
        v = None
        for it in range(self.routing_iters):
            c = F.softmax(b, dim=2)                                                      # coupling over classes
            s = (c.unsqueeze(-1) * u_hat).sum(1)                                         # B, K, D
            v = squash(s)
            if it < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(-1)                                 # agreement
        return v

    def capsules(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        p = self.primary(self.stem(x))                                                   # B, T*8, 8, 8
        H, W_ = p.shape[2:]
        p = p.view(B, self.n_types, self.prim_dim, H, W_).permute(0, 1, 3, 4, 2)         # B, T, H, W, 8
        u = squash(p)
        u_hat = torch.einsum("tkdp,bthwp->bthwkd", self.W, u)                            # prediction vectors
        u_hat = u_hat.reshape(B, self.n_types * H * W_, self.num_classes, -1)
        return self.route(u_hat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.capsules(x)
        lengths = torch.sqrt((v * v).sum(-1) + 1e-8)                                     # B, K in [0, 1)
        return lengths * self.logit_scale

    def extra_loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Margin loss of the original paper, computed from the capsule lengths (logits / scale)."""
        lengths = logits / self.logit_scale
        t = F.one_hot(y, self.num_classes).to(lengths.dtype)
        loss = t * F.relu(0.9 - lengths) ** 2 + 0.5 * (1 - t) * F.relu(lengths - 0.1) ** 2
        return loss.sum(1).mean()


def build_model(num_classes: int = NUM_CLASSES) -> CapsNet:
    return CapsNet(num_classes)
