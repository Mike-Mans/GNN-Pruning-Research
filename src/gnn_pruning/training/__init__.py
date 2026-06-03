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


def _csr_by_dst(edge_index: torch.Tensor, num_nodes: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR over destinations: `col[rowptr[n]:rowptr[n+1]]` are the in-neighbours
    of node `n` (sources of edges →n) — i.e. the nodes `n` aggregates from."""
    dst = edge_index[1]
    order = torch.argsort(dst)
    col = edge_index[0][order].contiguous()
    counts = torch.bincount(dst, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    torch.cumsum(counts, 0, out=rowptr[1:])
    return rowptr, col


def _neighbor_sample(seeds: torch.Tensor, rowptr: torch.Tensor, col: torch.Tensor,
                     fanouts: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised GraphSAGE-style neighbour sampling (pure PyTorch — no
    pyg-lib/torch-sparse). Returns `(sub_nodes, sub_edge_index)` for an L-hop
    subgraph around `seeds`, with the seeds at local indices `0..len(seeds)-1`.
    Edges point neighbour→target (message direction). Sampling is with
    replacement (a small, standard training-time approximation)."""
    num_nodes = rowptr.numel() - 1
    e_src, e_dst = [], []
    frontier = seeds
    for f in fanouts:
        deg = rowptr[frontier + 1] - rowptr[frontier]
        fr = frontier[deg > 0]
        d = deg[deg > 0]
        if fr.numel() == 0:
            break
        rnd = (torch.rand(fr.numel(), f) * d.unsqueeze(1)).long()   # [F, f]
        nbr = col[rowptr[fr].unsqueeze(1) + rnd]                    # [F, f]
        e_src.append(nbr.reshape(-1))
        e_dst.append(fr.repeat_interleave(f))
        frontier = torch.unique(nbr.reshape(-1))
    src = torch.cat(e_src) if e_src else seeds.new_empty(0)
    dst = torch.cat(e_dst) if e_dst else seeds.new_empty(0)
    # Relabel to a compact subgraph with seeds first.
    n_seed = seeds.numel()
    remap = torch.full((num_nodes,), -1, dtype=torch.long)
    remap[seeds] = torch.arange(n_seed)
    involved = torch.cat([seeds, src, dst]).unique()
    new = involved[remap[involved] < 0]
    remap[new] = torch.arange(n_seed, n_seed + new.numel())
    sub_nodes = torch.empty(n_seed + new.numel(), dtype=torch.long)
    sub_nodes[remap[seeds]] = seeds
    sub_nodes[remap[new]] = new
    sub_edge_index = torch.stack([remap[src], remap[dst]], dim=0)
    return sub_nodes, sub_edge_index


def _train_node_minibatch(
    model: nn.Module, data: Data, *, device: torch.device,
    lr: float, weight_decay: float, epochs: int, patience: int,
    metric_name: str, split: int, use_sparse: bool,
    fanout: list, batch_size: int,
) -> tuple[dict, float, int]:
    """Minibatch training for large graphs — many gradient steps per epoch via
    neighbour-sampled subgraphs. Only training is minibatched; validation is a
    full-graph forward (sparse path when `use_sparse`), matching the full-batch
    eval used everywhere else. Sampling is on CPU; only the small subgraph's
    features move to `device`, so GPU memory stays bounded."""
    train_mask, val_mask, test_mask = get_node_masks(data, split=split)
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    rowptr, col = _csr_by_dst(data.edge_index, data.num_nodes)  # CPU
    x_cpu, y_cpu = data.x, _flatten_y(data.y)

    # Full graph on-device for validation (fits via the sparse path).
    data_dev = data.to(device)
    fwd_adj = _forward_adj(data_dev, use_sparse)
    val_target = (val_mask if val_mask is not None else test_mask).to(device)
    y_dev = _flatten_y(data_dev.y)

    loss = loss_fn(metric_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel())]
        for i in range(0, perm.numel(), batch_size):
            seeds = perm[i:i + batch_size]
            sub_nodes, sub_ei = _neighbor_sample(seeds, rowptr, col, fanout)
            x = x_cpu[sub_nodes].to(device)
            ei = sub_ei.to(device)
            opt.zero_grad()
            out = model(x, ei)[:seeds.numel()]      # seeds lead the subgraph
            seed_y = y_cpu[seeds].to(device)
            target = seed_y.float() if metric_name == "micro_f1" else seed_y
            loss_value = loss(out, target)
            loss_value.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data_dev.x, fwd_adj)
            current = evaluate_node_classification(
                out, y_dev, val_target, metric_name)
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
    mini_batch: bool = False,
    fanout: Optional[list] = None,
    batch_size: int = 1024,
) -> tuple[dict, float, int]:
    torch.manual_seed(seed)
    model = model.to(device)
    if mini_batch:
        # NeighborLoader training for large graphs (many gradient steps/epoch).
        # Validation/eval stay full-batch — see `_train_node_minibatch`.
        return _train_node_minibatch(
            model, data, device=device, lr=lr, weight_decay=weight_decay,
            epochs=epochs, patience=patience, metric_name=metric_name,
            split=split, use_sparse=use_sparse,
            fanout=fanout or [15, 10], batch_size=batch_size)
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
