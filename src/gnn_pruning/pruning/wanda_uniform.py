"""Issue #3: Wanda-Uniform — `S_pq = |W_pq| · ‖X̄_q‖_2` (column L2, uniform).

Per cell: load dense checkpoint → single forward pass under hooks to capture
each prunable Linear's input tensor `X^(ℓ)` → compute per-feature L2 norm
across nodes → for each sparsity `s ∈ {0.1, …, 0.9}` apply a per-layer
top-k mask using `|W| * a_q` and re-evaluate.

Calibration set: training-mask nodes for transductive NC (matches issue
#3 wording "training subgraph"); all nodes when no mask exists (e.g.,
graph-classification, where every training-graph node is calibration).

Architecture correctness notes are documented in
`gnn_pruning/training/activation_hooks.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

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
from torch_geometric.loader import DataLoader


RESULTS_ROOT = Path("results/wanda-uniform")
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


def _collect_node_classification_activations(
    model: torch.nn.Module, data, device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    """Forward over the full graph; row-slice each X to training-mask nodes."""
    data = data.to(device)
    model = model.to(device).eval()
    acts = collect_activations(model, data.x, data.edge_index)
    train_mask, _, _ = get_node_masks(data)
    train_mask = train_mask.to(device)
    n_used = int(train_mask.sum().item())
    sliced = {k: v[train_mask] for k, v in acts.items()}
    return sliced, n_used


def _collect_graph_classification_activations(
    model: torch.nn.Module, dataset, device: torch.device, seed: int,
    batch_size: int = 64,
) -> tuple[dict[str, torch.Tensor], int]:
    """Concatenate activations across all training-graph nodes."""
    train_set, _, _ = _split_graphs(dataset, seed=seed)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    buckets: dict[str, list[torch.Tensor]] = {}
    n_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        acts = collect_activations(
            model, batch.x.float(), batch.edge_index, batch=batch.batch
        )
        for k, v in acts.items():
            buckets.setdefault(k, []).append(v)
        n_nodes += int(batch.x.shape[0])
    merged = {k: torch.cat(v, dim=0) for k, v in buckets.items()}
    return merged, n_nodes


def _wanda_score_pairs(
    model: torch.nn.Module,
    activations: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return `[(W, |W| * a_q), ...]` for every prunable Linear in the model.

    `a_q = ||X[:, q]||_2`. Score broadcasts to W's shape (out, in).
    """
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name, lin in named_prunable_linears(model):
        if name not in activations:
            raise RuntimeError(
                f"Activation for prunable layer {name!r} was not captured."
            )
        x = activations[name]
        # Per-feature L2 norm across nodes — shape (in_dim,).
        a = x.norm(dim=0, p=2).to(lin.weight.device, dtype=lin.weight.dtype)
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

    if task == "graph-classification":
        ds = load_dataset(dataset)
        # Replay #1's train/test split so eval is consistent.
        _, _, test_set = _split_graphs(ds, seed=int(ck.get("seed", 0)))
        model._test_split = test_set  # type: ignore[attr-defined]
        activations, n_used = _collect_graph_classification_activations(
            model, ds, device, seed=int(ck.get("seed", 0))
        )
    else:
        ds = load_dataset(dataset)
        activations, n_used = _collect_node_classification_activations(
            model, ds, device
        )

    pairs = _wanda_score_pairs(model, activations)

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
        "activation_collection_seed": int(seed),
        "n_nodes_used": int(n_used),
        "checkpoint": ck_path,
    }
    (cell_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_path = RESULTS_ROOT / "plots" / \
        f"accuracy_vs_sparsity_{dataset}_{architecture}.png"
    plot_accuracy_vs_sparsity(
        dataset, architecture,
        sparsities=sparsity_grid, values=metric_values,
        out_path=plot_path,
        metric_name=metric_name,
        method_label="wanda-uniform",
        reference_paths={
            "dense (#1)": DENSE_ROOT / "summary.csv",
            "magnitude (#2)": Path("results/magnitude/summary.csv"),
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
