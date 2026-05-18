"""Issue #5: Wanda-Per-Class — class-balanced activation norms.

For each layer ℓ:

    a^(ℓ,c)_q = || X^(ℓ)_{i ∈ class c, q} ||_2          (per-class column norm)
    ā^(ℓ)_q   = (1/C) Σ_c a^(ℓ,c)_q                     (equal class weights)
    S^(ℓ)_pq  = |W^(ℓ)_pq| · ā^(ℓ)_q                    (score)

Equal-weighting across classes prevents majority-class nodes from dominating
the activation norm — the documented motivation in `proposal.tex` for the
expected lift on heterophilic / class-imbalanced graphs.

Per-task class partitioning:
- Single-label NC: partition by training-mask labels `y` ∈ [0, C).
- Multi-label NC (Yelp, ogbn-proteins): each of the L binary labels is split
  into positive/negative → 2L "classes". `multi_label_strategy =
  "binary-per-label"` is logged in metrics.json.
- Graph classification (BBBP, Proteins): each node's class is its **parent
  graph's** label, so per-node-class partitioning is correct after batching.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from gnn_pruning.data import DATASET_META, load_dataset
from gnn_pruning.models import build_model, named_prunable_linears
from gnn_pruning.plotting import plot_accuracy_vs_sparsity
from gnn_pruning.pruning import masked_weights
from gnn_pruning.training import (
    _split_graphs,  # noqa: PLC2701
    evaluate_test,
    evaluate_test_graphs,
    get_node_masks,
)
from gnn_pruning.training.activation_hooks import collect_activations


RESULTS_ROOT = Path("results/wanda-per-class")
DENSE_ROOT = Path("results/no-pruning")


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_dense(dataset: str, architecture: str,
                checkpoint_dir: str | None) -> tuple[dict, str]:
    root = Path(checkpoint_dir) if checkpoint_dir else DENSE_ROOT
    ck_path = root / dataset / architecture / "checkpoint.pt"
    if not ck_path.exists():
        raise FileNotFoundError(
            f"Dense checkpoint missing at {ck_path}. Run issue #1 first."
        )
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    return ck, str(ck_path)


def _build_from_checkpoint(architecture: str, ck: dict) -> torch.nn.Module:
    model = build_model(architecture, ck["in_dim"], ck["hidden_dim"], ck["out_dim"])
    model.load_state_dict(ck["state_dict"])
    return model


def _per_class_norm_single_label(
    x: torch.Tensor, labels: torch.Tensor, n_classes: int,
) -> tuple[torch.Tensor, list[int]]:
    """Average of per-class column L2 norms, plus per-class counts.

    Classes with zero training nodes contribute zero (and are excluded from
    the averaging denominator to avoid biasing toward sparse classes).
    """
    in_dim = x.shape[1]
    norms = torch.zeros(in_dim, dtype=x.dtype, device=x.device)
    counts: list[int] = [0] * n_classes
    n_nonempty = 0
    for c in range(n_classes):
        mask = labels == c
        n_c = int(mask.sum().item())
        counts[c] = n_c
        if n_c == 0:
            continue
        norms += x[mask].norm(dim=0, p=2)
        n_nonempty += 1
    if n_nonempty > 0:
        norms = norms / n_nonempty
    return norms, counts


def _per_class_norm_multi_label(
    x: torch.Tensor, labels: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Multi-label: each of L binary labels has a (pos, neg) partition.

    Score: average over 2L "classes" of the per-class column L2 norm of x.
    Counts are returned as a flat list of length 2L.
    """
    in_dim = x.shape[1]
    n_labels = labels.shape[1]
    norms = torch.zeros(in_dim, dtype=x.dtype, device=x.device)
    counts: list[int] = []
    n_nonempty = 0
    for c in range(n_labels):
        col = labels[:, c]
        for sign, mask in (("pos", col > 0.5), ("neg", col <= 0.5)):
            n_c = int(mask.sum().item())
            counts.append(n_c)
            if n_c == 0:
                continue
            norms += x[mask].norm(dim=0, p=2)
            n_nonempty += 1
    if n_nonempty > 0:
        norms = norms / n_nonempty
    return norms, counts


def _collect_node_classification(
    model: torch.nn.Module, data, device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, bool]:
    data = data.to(device)
    model = model.to(device).eval()
    acts = collect_activations(model, data.x, data.edge_index)
    train_mask, _, _ = get_node_masks(data)
    train_mask = train_mask.to(device)
    n_used = int(train_mask.sum().item())
    acts_used = {k: v[train_mask] for k, v in acts.items()}
    y = data.y
    if y.dim() == 2 and y.shape[1] == 1:
        y = y.view(-1)
    if y.dim() == 2:
        # Multi-label target shape (n, L).
        labels = y[train_mask].float()
        multi_label = True
    else:
        labels = y[train_mask]
        multi_label = False
    return acts_used, labels, n_used, multi_label


def _collect_graph_classification(
    model: torch.nn.Module, dataset, device: torch.device, seed: int,
    batch_size: int = 64,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, bool]:
    """Activations are per-node; each node inherits its parent graph's label."""
    train_set, _, _ = _split_graphs(dataset, seed=seed)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    buckets: dict[str, list[torch.Tensor]] = {}
    node_labels: list[torch.Tensor] = []
    n_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        acts = collect_activations(
            model, batch.x.float(), batch.edge_index, batch=batch.batch
        )
        for k, v in acts.items():
            buckets.setdefault(k, []).append(v)
        y_g = batch.y.view(-1).long()
        per_node_labels = y_g[batch.batch]
        node_labels.append(per_node_labels)
        n_nodes += int(batch.x.shape[0])
    merged_acts = {k: torch.cat(v, dim=0) for k, v in buckets.items()}
    merged_labels = torch.cat(node_labels, dim=0)
    return merged_acts, merged_labels, n_nodes, False


def _class_summary(labels: torch.Tensor, multi_label: bool) -> dict:
    if multi_label:
        n_labels = labels.shape[1]
        counts = []
        for c in range(n_labels):
            counts.append(int((labels[:, c] > 0.5).sum().item()))
            counts.append(int((labels[:, c] <= 0.5).sum().item()))
        return {
            "n_classes": 2 * n_labels,
            "min_count": int(min(counts)),
            "max_count": int(max(counts)),
            "imbalance_ratio": (max(counts) / max(min(counts), 1)),
        }
    counts: dict[int, int] = {}
    for v in labels.view(-1).tolist():
        counts[int(v)] = counts.get(int(v), 0) + 1
    vals = list(counts.values())
    if not vals:
        return {"n_classes": 0, "min_count": 0, "max_count": 0,
                "imbalance_ratio": 1.0}
    return {
        "n_classes": int(len(counts)),
        "min_count": int(min(vals)),
        "max_count": int(max(vals)),
        "imbalance_ratio": float(max(vals) / max(min(vals), 1)),
    }


def _wanda_per_class_score_pairs(
    model: torch.nn.Module,
    activations: dict[str, torch.Tensor],
    labels: torch.Tensor,
    multi_label: bool,
    n_classes_hint: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Score each prunable Linear by |W| times the class-averaged column norm."""
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name, lin in named_prunable_linears(model):
        if name not in activations:
            raise RuntimeError(f"Missing activation for layer {name!r}")
        x = activations[name]
        if multi_label:
            a, _ = _per_class_norm_multi_label(x, labels.to(x.device))
        else:
            n_classes = n_classes_hint or (int(labels.max().item()) + 1)
            a, _ = _per_class_norm_single_label(
                x, labels.to(x.device).long(), n_classes
            )
        a = a.to(lin.weight.device, dtype=lin.weight.dtype)
        score = lin.weight.detach().abs() * a.unsqueeze(0)
        pairs.append((lin.weight, score))
    return pairs


def run_cell(
    *,
    dataset: str,
    architecture: str,
    metric_name: str,
    sparsity_grid: list[float],
    checkpoint_dir: str | None = None,
    seed: int = 0,
    **_unused_hp,
) -> list[dict]:
    torch.manual_seed(seed)
    device = _device()

    ck, ck_path = _load_dense(dataset, architecture, checkpoint_dir)
    model = _build_from_checkpoint(architecture, ck).to(device)

    meta = DATASET_META.get(dataset.lower())
    task = meta.task if meta else "node-classification"
    is_multi_label = (meta is not None and meta.multi_label) \
        or metric_name == "micro_f1"

    if task == "graph-classification":
        ds = load_dataset(dataset)
        _, _, test_set = _split_graphs(ds, seed=int(ck.get("seed", 0)))
        model._test_split = test_set  # type: ignore[attr-defined]
        activations, labels, n_used, _ = _collect_graph_classification(
            model, ds, device, seed=int(ck.get("seed", 0))
        )
        multi_label = False
        n_classes_hint = int(ck["out_dim"])
    else:
        ds = load_dataset(dataset)
        activations, labels, n_used, multi_label = \
            _collect_node_classification(model, ds, device)
        n_classes_hint = int(ck["out_dim"])
        if is_multi_label:
            multi_label = True

    class_summary = _class_summary(labels, multi_label)
    pairs = _wanda_per_class_score_pairs(
        model, activations, labels, multi_label, n_classes_hint=n_classes_hint
    )

    metric_values: list[float] = []
    for s in sparsity_grid:
        with masked_weights(pairs, sparsity=s):
            if task == "graph-classification":
                v = evaluate_test_graphs(model, device)
            else:
                v = evaluate_test(model, ds, device, metric_name)
        metric_values.append(float(v))

    cell_dir = RESULTS_ROOT / dataset / architecture
    cell_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "sparsity_grid": list(map(float, sparsity_grid)),
        "metric_name": metric_name,
        "metric_values": metric_values,
        "class_summary": class_summary,
        "multi_label_strategy": "binary-per-label" if multi_label else None,
        "n_nodes_used": int(n_used),
        "checkpoint": ck_path,
        "seed": int(seed),
    }
    (cell_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_path = RESULTS_ROOT / "plots" / \
        f"accuracy_vs_sparsity_{dataset}_{architecture}.png"
    plot_accuracy_vs_sparsity(
        dataset, architecture,
        sparsities=sparsity_grid, values=metric_values,
        out_path=plot_path,
        metric_name=metric_name,
        method_label="wanda-per-class",
        reference_paths={
            "dense (#1)": DENSE_ROOT / "summary.csv",
            "magnitude (#2)": Path("results/magnitude/summary.csv"),
            "wanda-uniform (#3)": Path("results/wanda-uniform/summary.csv"),
            "wanda-degree (#4)": Path("results/wanda-degree/summary.csv"),
        },
    )

    return [
        {
            "dataset": dataset,
            "architecture": architecture,
            "sparsity": float(s),
            "metric_name": metric_name,
            "metric_value": float(v),
        }
        for s, v in zip(sparsity_grid, metric_values)
    ]
