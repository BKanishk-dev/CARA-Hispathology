"""VMamba - Vision Mamba (2D selective state-space model) classifier, pure PyTorch.

Implements the core of VMamba (Liu et al., 2024):
  * VSS block  = LN -> SS2D -> residual, then LN -> MLP -> residual
  * SS2D       = in-projection, depth-wise conv, SiLU, 4-direction cross-scan
                 (row-major, column-major and their reverses), a selective SSM (S6) per direction,
                 cross-merge, output gate with SiLU(z), out-projection
  * hierarchy  = 4 stages with patch-merging downsampling

The selective scan  h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t ,  y_t = C_t . h_t + D * x_t
is evaluated with a chunked closed-form scan (parallel inside 16-token chunks, chained across chunks)
in plain PyTorch - no CUDA kernels needed, so the model runs on CPU and GPU alike.  For speed the network is a "lite" VMamba: 8x8 patch
embedding (32x32 = 1024 tokens at stage 1), dims (64, 128, 256, 384), depths (2, 2, 4, 2), d_state 8.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..utils import NUM_CLASSES, BaseClassifier


SCAN_CHUNK = 16
MIN_LOG_DECAY = -5.0       # per-step decay factor is kept >= exp(-5); keeps exp(-cumsum) finite inside a chunk


def chunked_scan(a_log: torch.Tensor, b: torch.Tensor, chunk: int = SCAN_CHUNK) -> torch.Tensor:
    """Solve h_t = exp(a_log_t) * h_{t-1} + b_t  (h_{-1} = 0) along dim=2.

    a_log, b: (B, K, L, D, N) with a_log <= 0.  Inside a chunk the recurrence has the closed form
        h_t = exp(S_t) * cumsum_s( exp(-S_s) * b_s ),   S = cumsum(a_log)
    which is evaluated fully in parallel; chunks are chained sequentially through a carried state.
    """
    B, K, L, D, N = b.shape
    pad = (-L) % chunk
    if pad:                                                    # identity extension: decay 1, input 0
        a_log = F.pad(a_log, (0, 0, 0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, 0, 0, pad))
    nC = (L + pad) // chunk
    a_log = a_log.reshape(B, K, nC, chunk, D, N)
    b = b.reshape(B, K, nC, chunk, D, N)
    S = torch.cumsum(a_log, dim=3)
    decay = torch.exp(S)
    h_loc = decay * torch.cumsum(torch.exp(-S) * b, dim=3)     # zero-initial-state solution per chunk
    carry = b.new_zeros(B, K, 1, D, N)
    outs = []
    for i in range(nC):
        h = h_loc[:, :, i] + decay[:, :, i] * carry            # add the contribution of the previous chunks
        outs.append(h)
        carry = h[:, :, -1:]
    h = torch.stack(outs, dim=2).reshape(B, K, nC * chunk, D, N)
    return h[:, :, :L]


def selective_scan(u, delta, A, B, C, D):
    """u, delta: (B, K, L, d); A: (K, d, N); B, C: (B, K, L, N); D: (K, d) -> y: (B, K, L, d)."""
    a_log = (delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)).clamp(min=MIN_LOG_DECAY, max=0.0)   # B,K,L,d,N
    dBu = delta.unsqueeze(-1) * B.unsqueeze(3) * u.unsqueeze(-1)                                      # B,K,L,d,N
    h = chunked_scan(a_log, dBu)
    y = (h * C.unsqueeze(3)).sum(-1)
    return y + u * D.unsqueeze(0).unsqueeze(2)


class SS2D(nn.Module):
    def __init__(self, dim: int, d_state: int = 8, expand: int = 2, dt_rank: int | None = None, d_conv: int = 3):
        super().__init__()
        self.d_inner = d_inner = expand * dim
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(dim / 16)
        K = 4
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.conv = nn.Conv2d(d_inner, d_inner, d_conv, padding=d_conv // 2, groups=d_inner)
        self.x_proj = nn.Parameter(torch.randn(K, self.dt_rank + 2 * d_state, d_inner) * d_inner ** -0.5)
        self.dt_proj_w = nn.Parameter(torch.randn(K, d_inner, self.dt_rank) * self.dt_rank ** -0.5)
        # dt bias initialised so that softplus(bias) in [1e-3, 1e-1]
        dt = torch.exp(torch.rand(K, d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        self.dt_proj_b = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(K, d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(K, d_inner))
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # x: (B, H, W, C)
        Bsz, H, W, _ = x.shape
        L = H * W
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = F.silu(self.conv(xi.permute(0, 3, 1, 2)))                    # B, d, H, W

        # cross-scan: 4 sequences (row-major, column-major, and their reverses)
        x_row = xi.flatten(2)                                              # B, d, L
        x_col = xi.transpose(2, 3).flatten(2)
        xs = torch.stack([x_row, x_col, x_row.flip(-1), x_col.flip(-1)], dim=1)   # B, 4, d, L
        xs = xs.transpose(2, 3)                                            # B, 4, L, d

        proj = torch.einsum("bkld,kcd->bklc", xs, self.x_proj)             # B,4,L, R+2N
        dt, Bm, Cm = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(torch.einsum("bklr,kdr->bkld", dt, self.dt_proj_w) + self.dt_proj_b[None, :, None, :])
        A = -torch.exp(self.A_log.float())
        y = selective_scan(xs, delta, A, Bm, Cm, self.D)                   # B,4,L,d

        # cross-merge
        y = y.transpose(2, 3)                                              # B,4,d,L
        y_row = y[:, 0] + y[:, 2].flip(-1)
        y_col = (y[:, 1] + y[:, 3].flip(-1)).view(Bsz, -1, W, H).transpose(2, 3).reshape(Bsz, -1, L)
        y = (y_row + y_col).transpose(1, 2).reshape(Bsz, H, W, -1)
        y = self.out_norm(y) * F.silu(z)
        return self.out_proj(y)


class VSSBlock(nn.Module):
    def __init__(self, dim: int, d_state: int, mlp_ratio: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.drop_path = drop_path

    def _dp(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path == 0.0:
            return x
        keep = 1.0 - self.drop_path
        return x * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep).to(x.dtype) / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._dp(self.ss2d(self.norm1(x)))
        return x + self._dp(self.mlp(self.norm2(x)))


class Downsample(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))


class VMamba(BaseClassifier):
    key = "vmamba"

    def __init__(self, num_classes: int = NUM_CLASSES, dims=(64, 128, 256, 384), depths=(2, 2, 4, 2),
                 d_state: int = 8, patch: int = 8, drop_path: float = 0.1, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.patch_embed = nn.Sequential(nn.Conv2d(3, dims[0], patch, patch))
        self.embed_norm = nn.LayerNorm(dims[0])
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        k = 0
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            self.stages.append(nn.ModuleList([VSSBlock(dim, d_state, drop_path=dpr[k + j]) for j in range(depth)]))
            k += depth
            if i < len(dims) - 1:
                self.downs.append(Downsample(dim, dims[i + 1]))
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
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
        x = self.embed_norm(self.patch_embed(x).permute(0, 2, 3, 1))     # B, 32, 32, C
        for i, blocks in enumerate(self.stages):
            for blk in blocks:
                if self.use_checkpoint and self.training and x.requires_grad:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return self.head(self.norm(x).mean((1, 2)))


def build_model(num_classes: int = NUM_CLASSES) -> VMamba:
    return VMamba(num_classes)
