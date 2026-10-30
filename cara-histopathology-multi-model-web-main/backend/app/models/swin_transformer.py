"""Swin-Tiny (Liu et al., 2021) implemented from scratch for 256x256 inputs.

patch size 4, embed dim 96, depths (2, 2, 6, 2), heads (3, 6, 12, 24), window 8.
Token grids are 64 -> 32 -> 16 -> 8, so every grid is divisible by the window size.
At the last stage the grid equals the window, so shifting is disabled there (as in the paper).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import NUM_CLASSES, BaseClassifier


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(w: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    B = w.shape[0] // ((H // ws) * (W // ws))
    x = w.view(B, H // ws, W // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) < keep).to(x.dtype) / keep
        return x * mask


class WindowAttention(nn.Module):
    def __init__(self, dim: int, ws: int, heads: int):
        super().__init__()
        self.heads, self.ws = heads, ws
        self.scale = (dim // heads) ** -0.5
        self.rel_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)
        attn = attn.softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(Bw, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, drop_path):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.drop_path = DropPath(drop_path)
        self._mask_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _mask(self, H: int, W: int, device) -> torch.Tensor | None:
        if self.shift == 0:
            return None
        key = (H, W, str(device))
        if key not in self._mask_cache:
            img = torch.zeros(1, H, W, 1)
            cnt = 0
            for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                    img[:, hs, wsl, :] = cnt
                    cnt += 1
            win = window_partition(img, self.ws).squeeze(-1)
            m = win.unsqueeze(1) - win.unsqueeze(2)
            self._mask_cache[key] = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0).to(device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: (B, H, W, C)
        B, H, W, C = x.shape
        h = self.norm1(x)
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), (1, 2))
        w = self.attn(window_partition(h, self.ws), self._mask(H, W, x.device))
        h = window_reverse(w, self.ws, H, W)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), (1, 2))
        x = x + self.drop_path(h)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x))


class SwinTransformer(BaseClassifier):
    key = "swin_transformer"

    def __init__(self, num_classes: int = NUM_CLASSES, embed_dim: int = 96, depths=(2, 2, 6, 2),
                 heads=(3, 6, 12, 24), window: int = 8, drop_path: float = 0.1, img_size: int = 256):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, 4, 4)
        self.embed_norm = nn.LayerNorm(embed_dim)
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        stages, merges, dim, res, k = [], [], embed_dim, img_size // 4, 0
        for i, (d, h) in enumerate(zip(depths, heads)):
            ws = min(window, res)
            blocks = []
            for j in range(d):
                shift = 0 if (j % 2 == 0 or res <= ws) else ws // 2
                blocks.append(SwinBlock(dim, h, ws, shift, dpr[k]))
                k += 1
            stages.append(nn.Sequential(*blocks))
            if i < len(depths) - 1:
                merges.append(PatchMerging(dim))
                dim, res = dim * 2, res // 2
        self.stages = nn.ModuleList(stages)
        self.merges = nn.ModuleList(merges)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm(self.patch_embed(x).permute(0, 2, 3, 1))     # B, 64, 64, 96
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.merges):
                x = self.merges[i](x)
        x = self.norm(x).mean((1, 2))
        return self.head(x)


def build_model(num_classes: int = NUM_CLASSES) -> SwinTransformer:
    return SwinTransformer(num_classes)
