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
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter


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


class GPRGNN(nn.Module):
    """Generalized PageRank GNN (Chien et al., ICLR 2021) — a heterophily-capable
    backbone whose prunable weights are still plain `nn.Linear` layers.

    Architecture: a 2-layer MLP feature transform, then K steps of generalized
    PageRank propagation with **learnable** per-hop coefficients ``gamma`` (the
    K+1 scalars are learned, not pruned). The learnable coefficients can go
    negative — that high-pass behaviour is what lets it model heterophily where
    GCN/GAT fail.

    Prunable layers: ``lin1`` / ``lin2`` (exposed via :meth:`prunable_linears`).
    Like GCNConv (which applies its weight *before* propagation), both Linears
    see pre-propagation inputs, so Wanda / degree / per-class scoring applies
    exactly as it does for GCN.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 K: int = 10, alpha: float = 0.1, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.K = K
        self.dropout = dropout
        # PPR initialisation of the per-hop coefficients (learnable thereafter).
        gamma = alpha * (1.0 - alpha) ** torch.arange(K + 1, dtype=torch.float32)
        gamma[-1] = (1.0 - alpha) ** K
        self.gamma = nn.Parameter(gamma)

    def prunable_linears(self) -> list[tuple[str, nn.Linear]]:
        return [("lin1", self.lin1), ("lin2", self.lin2)]

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin2(x)
        # Symmetric-normalised adjacency with self-loops, then GPR propagation.
        ei, ew = gcn_norm(edge_index, num_nodes=h.size(0), add_self_loops=True)
        src, dst = ei[0], ei[1]
        z = self.gamma[0] * h
        for k in range(1, self.K + 1):
            h = scatter(ew.unsqueeze(-1) * h[src], dst, dim=0,
                        dim_size=h.size(0), reduce="sum")
            z = z + self.gamma[k] * h
        if batch is not None:
            z = global_mean_pool(z, batch)
        return z


ARCHITECTURES = {"gcn": GCN, "gat": GAT, "graphsage": GraphSAGE, "sage": GraphSAGE,
                 "gprgnn": GPRGNN}


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
    """Flatten prunable Linears as `(conv_path.suffix, linear)`.

    Models that aren't built from PyG conv layers (e.g. `GPRGNN`) can expose
    their prunable Linears directly via a `prunable_linears()` method; this is
    the single integration point shared by the activation hooks and every
    pruning method, so a new backbone only needs that one method.
    """
    if hasattr(model, "prunable_linears"):
        return list(model.prunable_linears())
    out: list[tuple[str, nn.Linear]] = []
    for conv_name, conv in prunable_layers(model):
        for suffix, lin in conv_prunable_linears(conv):
            out.append((f"{conv_name}.{suffix}", lin))
    return out


def named_prunable_weights(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Flatten prunable weights as `(conv_path.suffix.weight, W)`."""
    return [(f"{name}.weight", lin.weight)
            for name, lin in named_prunable_linears(model)]
