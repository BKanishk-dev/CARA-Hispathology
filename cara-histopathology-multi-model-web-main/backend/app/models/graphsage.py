"""GraphSAGE classifier.

Each 256x256 image is turned into a graph:
  * nodes   : the 8x8 = 64 non-overlapping 32x32 patches
  * features: a small CNN (shared across patches) encodes every patch into a 128-d vector,
              plus a learned positional embedding
  * edges   : 8-neighbourhood of the patch grid (spatial adjacency)
Three GraphSAGE layers (mean aggregator, dense adjacency) are applied, followed by a
mean+max graph read-out and an MLP classification head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import NUM_CLASSES, BaseClassifier, init_weights

PATCH = 32
GRID = 256 // PATCH  # 8


def grid_adjacency(grid: int = GRID, eight_neighbour: bool = True) -> torch.Tensor:
    """Row-normalised neighbour matrix (without self loops) for a grid x grid patch graph."""
    n = grid * grid
    adj = torch.zeros(n, n)
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr == 0 and dc == 0) or (not eight_neighbour and dr != 0 and dc != 0):
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid and 0 <= cc < grid:
                        adj[i, rr * grid + cc] = 1.0
    return adj / adj.sum(1, keepdim=True)


class PatchEncoder(nn.Module):
    """CNN applied to every 32x32 patch independently -> feature vector."""

    def __init__(self, out_dim: int = 128):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128))   # 32 -> 4
        self.proj = nn.Linear(128 * 2, out_dim)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        f = self.net(p)
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)
        return F.relu(self.proj(f))


class SAGELayer(nn.Module):
    """h_v' = ReLU( W_self h_v + W_neigh mean_{u in N(v)} h_u ), followed by L2 normalisation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.matmul(adj, h)                    # (B, N, D) mean over neighbours
        out = F.relu(self.w_self(h) + self.w_neigh(agg))
        return self.norm(out)


class GraphSAGE(BaseClassifier):
    key = "graphsage"

    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 128, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = PatchEncoder(hidden)
        self.pos = nn.Parameter(torch.zeros(1, GRID * GRID, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.sage = nn.ModuleList([SAGELayer(hidden, hidden) for _ in range(layers)])
        self.register_buffer("adj", grid_adjacency(), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(256, num_classes))
        init_weights(self.encoder)

    def image_to_graph(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 256, 256) -> node features (B, 64, hidden)."""
        B = x.shape[0]
        patches = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)        # B,3,8,8,32,32
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * GRID * GRID, 3, PATCH, PATCH)
        return self.encoder(patches).view(B, GRID * GRID, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.image_to_graph(x) + self.pos
        for layer in self.sage:
            h = h + layer(self.drop(h), self.adj)                              # residual SAGE layer
        g = torch.cat([h.mean(1), h.amax(1)], dim=1)
        return self.head(g)


def build_model(num_classes: int = NUM_CLASSES) -> GraphSAGE:
    return GraphSAGE(num_classes)
