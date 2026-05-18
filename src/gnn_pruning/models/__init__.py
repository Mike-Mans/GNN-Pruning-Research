"""Model definitions for the M1 pipelines.

Two-layer GCN, GAT, and GraphSAGE backbones with hidden dim 256 (per
`proposal.tex`). For graph-classification datasets the backbone is wrapped
with a global-mean readout and a final linear head.

`prunable_layers(model)` returns the conv layers whose `lin` weight matrix
is the per-layer target of every method in the M1 milestone.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, cached=False, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, cached=False, add_self_loops=True)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 heads: int = 8, dropout: float = 0.5):
        super().__init__()
        # Layer 1: hidden_dim per head, concat across heads -> hidden_dim * heads
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout,
                             concat=True, add_self_loops=True)
        # Layer 2: average heads -> out_dim
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1,
                             concat=False, dropout=dropout, add_self_loops=True)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x


ARCHITECTURES = {"gcn": GCN, "gat": GAT, "graphsage": GraphSAGE, "sage": GraphSAGE}


def build_model(architecture: str, in_dim: int, hidden_dim: int, out_dim: int,
                **kwargs) -> nn.Module:
    key = architecture.lower()
    if key not in ARCHITECTURES:
        raise KeyError(
            f"Unknown architecture {architecture!r}. "
            f"Valid: {sorted(ARCHITECTURES)}"
        )
    return ARCHITECTURES[key](in_dim, hidden_dim, out_dim, **kwargs)


def prunable_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return the conv layers whose weight matrix is the per-layer target.

    The shared `apply_mask` utility scores and masks the `.lin.weight` (for
    GCN/GAT) or `.lin_l.weight` / `.lin_r.weight` (for SAGE) of each entry.
    Returning the conv module itself keeps the architecture-specific weight
    discovery localized in `pruning/__init__.py`.
    """
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, (GCNConv, GATConv, SAGEConv))]


def conv_prunable_linears(conv: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Return `(suffix, linear)` for every prunable Linear inside one conv.

    Wanda-family scoring needs the *actual input* to each Linear, not the
    input to its parent conv. For SAGE in particular, `lin_l` sees the
    aggregated neighbor features while `lin_r` sees the original self-features
    — so each Linear must be hooked separately to capture the right `X`.

    - GCN: a single `lin`.
    - GAT: a single `lin` in PyG 2.7 (when src/dst share weights, the standard
      non-bipartite case); falls back to `lin_src`/`lin_dst` for bipartite.
    - SAGE: `lin_l` (aggregated neighbors) and `lin_r` (self) — both are
      pruned independently and **see different inputs**.
    """
    linears: list[tuple[str, nn.Linear]] = []
    if isinstance(conv, GCNConv):
        linears.append(("lin", conv.lin))
    elif isinstance(conv, GATConv):
        if getattr(conv, "lin", None) is not None:
            linears.append(("lin", conv.lin))
        else:
            if getattr(conv, "lin_src", None) is not None:
                linears.append(("lin_src", conv.lin_src))
            if getattr(conv, "lin_dst", None) is not None and \
                    conv.lin_dst is not getattr(conv, "lin_src", None):
                linears.append(("lin_dst", conv.lin_dst))
    elif isinstance(conv, SAGEConv):
        linears.append(("lin_l", conv.lin_l))
        if hasattr(conv, "lin_r") and conv.lin_r is not None:
            linears.append(("lin_r", conv.lin_r))
    return linears


def named_prunable_linears(model: nn.Module
                           ) -> list[tuple[str, nn.Linear]]:
    """Flatten prunable Linears as `(conv_path.suffix, linear)`."""
    out: list[tuple[str, nn.Linear]] = []
    for conv_name, conv in prunable_layers(model):
        for suffix, lin in conv_prunable_linears(conv):
            out.append((f"{conv_name}.{suffix}", lin))
    return out


def named_prunable_weights(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Flatten prunable weights as `(conv_path.suffix.weight, W)`."""
    return [(f"{name}.weight", lin.weight)
            for name, lin in named_prunable_linears(model)]
