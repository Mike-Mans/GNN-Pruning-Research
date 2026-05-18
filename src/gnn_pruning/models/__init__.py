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


def conv_weight_tensors(conv: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Return (name, weight) for the prunable weight matrices inside one conv.

    GCN: a single `lin.weight` (shape: [out, in]).
    GAT: a single `lin.weight` or `lin_src.weight` (shape: [out*heads, in]).
    SAGE: `lin_l.weight` and `lin_r.weight` — both are pruned independently.
    """
    weights: list[tuple[str, torch.Tensor]] = []
    if isinstance(conv, GCNConv):
        weights.append(("lin.weight", conv.lin.weight))
    elif isinstance(conv, GATConv):
        # In PyG 2.7, GATConv uses a single `lin` (Linear(in, out*heads)) when
        # src and dst share weights (the standard non-bipartite case used here).
        # `lin_src` / `lin_dst` are only populated for bipartite inputs.
        if getattr(conv, "lin", None) is not None:
            weights.append(("lin.weight", conv.lin.weight))
        else:
            if getattr(conv, "lin_src", None) is not None:
                weights.append(("lin_src.weight", conv.lin_src.weight))
            if getattr(conv, "lin_dst", None) is not None and \
                    conv.lin_dst is not getattr(conv, "lin_src", None):
                weights.append(("lin_dst.weight", conv.lin_dst.weight))
    elif isinstance(conv, SAGEConv):
        weights.append(("lin_l.weight", conv.lin_l.weight))
        if hasattr(conv, "lin_r") and conv.lin_r is not None:
            weights.append(("lin_r.weight", conv.lin_r.weight))
    return weights


def named_prunable_weights(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Flatten prunable weights across the whole model as `(full.name, W)`."""
    out: list[tuple[str, torch.Tensor]] = []
    for conv_name, conv in prunable_layers(model):
        for w_name, w in conv_weight_tensors(conv):
            out.append((f"{conv_name}.{w_name}", w))
    return out
