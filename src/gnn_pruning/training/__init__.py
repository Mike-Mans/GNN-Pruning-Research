"""Training utilities for the M1 pipelines.

`train_node_classification` and `train_graph_classification` are the two
top-level entry points. Both accept a per-dataset hyperparameter dict and
return `(best_state_dict, best_metric, epoch_of_best)`. They early-stop on
the validation metric where a val split is available.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_pruning.eval import evaluate_node_classification, loss_fn


# Large undirected NC graphs that OOM full-batch dense on a single GPU (issue
# #6). For these we feed the convs a sparse CSR adjacency so aggregation runs
# as SpMM (memory O(N·F) instead of O(E·F)). Restricted to GCN/SAGE on these
# three because: (a) only they OOM, (b) all three are undirected so the sparse
# path is numerically identical to the dense edge_index path (verified on
# cora/reddit-like graphs; directed graphs like ogbn-arxiv would differ under
# gcn_norm and are deliberately kept dense — they fit anyway), and (c) GATConv
# does not support the native torch.sparse layout (and is infeasible full-batch
# on Reddit-scale regardless).
SPARSE_NC_DATASETS = {"reddit", "ogbn-products", "yelp"}
SPARSE_ARCHS = {"gcn", "graphsage", "sage"}


def use_sparse_adj(dataset: str, architecture: str) -> bool:
    return (dataset.lower() in SPARSE_NC_DATASETS
            and architecture.lower() in SPARSE_ARCHS)


def build_sparse_adj(data: Data):
    """Transposed CSR adjacency (`adj_t`) for memory-efficient SpMM.

    `to_torch_csr_tensor(edge_index.flip(0))` puts targets on rows and sources
    on cols, matching PyG's `adj_t` convention so `conv(x, adj_t)` aggregates
    each node's in-neighbours exactly as `conv(x, edge_index)` does.
    """
    from torch_geometric.utils import to_torch_csr_tensor

    ei = data.edge_index
    n = data.num_nodes
    return to_torch_csr_tensor(ei.flip(0), size=(n, n))


def _forward_adj(data: Data, use_sparse: bool):
    """The adjacency to feed the model: sparse CSR if requested, else COO
    edge_index. `data` must already be on the target device."""
    return build_sparse_adj(data) if use_sparse else data.edge_index


def _resolve_mask(data: Data, attr: str, split: int = 0) -> Optional[torch.Tensor]:
    mask = getattr(data, attr, None)
    if mask is None:
        return None
    # WebKB / Actor ship 10 split columns (Geom-GCN convention) — select the
    # requested split. 1-D masks ignore `split`, so non-multi-split datasets
    # are unaffected.
    if mask.dim() == 2:
        mask = mask[:, split]
    return mask.bool()


def _ogb_masks(data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = data.split_idx
    n = data.num_nodes
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    train[idx["train"]] = True
    val[idx["valid"]] = True
    test[idx["test"]] = True
    return train, val, test


def get_node_masks(data: Data, split: int = 0) -> tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor
]:
    """Return (train_mask, val_mask, test_mask) for node-classification data.

    OGB datasets use `split_idx`. Other PyG datasets use boolean mask
    attributes. WebKB-style datasets ship multiple split columns; `split`
    selects which one (default 0).
    """
    if hasattr(data, "split_idx"):
        return _ogb_masks(data)
    train = _resolve_mask(data, "train_mask", split)
    val = _resolve_mask(data, "val_mask", split)
    test = _resolve_mask(data, "test_mask", split)
    if train is None or test is None:
        # Fall back: random 60/20/20 split.
        n = data.num_nodes
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=gen)
        ntr, nv = int(0.6 * n), int(0.2 * n)
        train = torch.zeros(n, dtype=torch.bool)
        val = torch.zeros(n, dtype=torch.bool)
        test = torch.zeros(n, dtype=torch.bool)
        train[perm[:ntr]] = True
        val[perm[ntr:ntr + nv]] = True
        test[perm[ntr + nv:]] = True
    return train, val, test


def _flatten_y(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 2 and y.shape[1] == 1:
        return y.view(-1)
    return y


def train_node_classification(
    model: nn.Module,
    data: Data,
    *,
    device: torch.device,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 50,
    metric_name: str = "accuracy",
    seed: int = 0,
    split: int = 0,
    use_sparse: bool = False,
) -> tuple[dict, float, int]:
    torch.manual_seed(seed)
    model = model.to(device)
    data = data.to(device)
    fwd_adj = _forward_adj(data, use_sparse)

    train_mask, val_mask, test_mask = get_node_masks(data, split=split)
    train_mask = train_mask.to(device)
    if val_mask is not None:
        val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    y = _flatten_y(data.y)
    loss = loss_fn(metric_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, fwd_adj)
        if metric_name == "micro_f1":
            target = y[train_mask].float()
        else:
            target = y[train_mask]
        loss_value = loss(out[train_mask], target)
        loss_value.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, fwd_adj)
            val_target = val_mask if val_mask is not None else test_mask
            current = evaluate_node_classification(
                out, y, val_target, metric_name
            )
        if current > best_metric:
            best_metric = current
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    return best_state, best_metric, best_epoch


def evaluate_test(
    model: nn.Module, data: Data, device: torch.device, metric_name: str,
    split: int = 0, use_sparse: bool = False,
) -> float:
    model = model.to(device).eval()
    data = data.to(device)
    fwd_adj = _forward_adj(data, use_sparse)
    _, _, test_mask = get_node_masks(data, split=split)
    test_mask = test_mask.to(device)
    y = _flatten_y(data.y)
    with torch.no_grad():
        out = model(data.x, fwd_adj)
    return evaluate_node_classification(out, y, test_mask, metric_name)


def _split_graphs(dataset, seed: int = 0):
    n = len(dataset)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    ntr, nv = int(0.8 * n), int(0.1 * n)
    return (
        dataset[perm[:ntr]],
        dataset[perm[ntr:ntr + nv]],
        dataset[perm[ntr + nv:]],
    )


def train_graph_classification(
    model: nn.Module,
    dataset,
    *,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 20,
    metric_name: str = "accuracy",
    seed: int = 0,
) -> tuple[dict, float, int]:
    torch.manual_seed(seed)
    model = model.to(device)

    train_set, val_set, test_set = _split_graphs(dataset, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    loss = loss_fn(metric_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x.float(), batch.edge_index, batch=batch.batch)
            y = _flatten_y(batch.y)
            if metric_name == "micro_f1":
                target = y.float()
            else:
                target = y.long()
            loss_value = loss(out, target)
            loss_value.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x.float(), batch.edge_index, batch=batch.batch)
                preds = out.argmax(dim=-1)
                y = _flatten_y(batch.y).long()
                correct += int((preds == y).sum().item())
                total += int(y.numel())
        current = correct / max(total, 1)
        if current > best_metric:
            best_metric = current
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    # Also stash the test split into the model so the caller can rerun eval
    # against a stable held-out set without redoing the random split.
    model._test_split = test_set  # type: ignore[attr-defined]
    return best_state, best_metric, best_epoch


def evaluate_test_graphs(model: nn.Module, device: torch.device,
                         batch_size: int = 32) -> float:
    test_set = getattr(model, "_test_split", None)
    if test_set is None:
        raise RuntimeError(
            "Graph-classification model has no `_test_split` — call "
            "train_graph_classification first."
        )
    loader = DataLoader(test_set, batch_size=batch_size)
    model = model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x.float(), batch.edge_index, batch=batch.batch)
            preds = out.argmax(dim=-1)
            y = _flatten_y(batch.y).long()
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)
