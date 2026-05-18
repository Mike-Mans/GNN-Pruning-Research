"""Issue #1: dense baseline pipeline.

Per (dataset, architecture) cell: load data → build model → train to
convergence → eval → save checkpoint + metrics + trivial plot.

Outputs match issue #1's "Output paths" section exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from gnn_pruning.data import DATASET_META, load_dataset
from gnn_pruning.eval import evaluate_node_classification
from gnn_pruning.models import build_model
from gnn_pruning.plotting import plot_accuracy_vs_sparsity
from gnn_pruning.training import (
    evaluate_test,
    evaluate_test_graphs,
    train_graph_classification,
    train_node_classification,
)


RESULTS_ROOT = Path("results/no-pruning")


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _flatten_y(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 2 and y.shape[1] == 1:
        return y.view(-1)
    return y


def _infer_dims(data_or_dataset, task: str) -> tuple[int, int]:
    if task == "graph-classification":
        ds = data_or_dataset
        in_dim = ds[0].num_node_features
        y = ds.y if hasattr(ds, "y") else torch.cat([g.y.view(-1) for g in ds])
        out_dim = int(y.max().item()) + 1
        return in_dim, out_dim
    data = data_or_dataset
    in_dim = int(data.num_node_features)
    if hasattr(data, "num_classes") and data.num_classes is not None:
        out_dim = int(data.num_classes)
    else:
        y = _flatten_y(data.y)
        if y.dtype.is_floating_point:
            # multi-label: target shape is (n, C)
            out_dim = int(data.y.shape[1])
        else:
            out_dim = int(y.max().item()) + 1
    return in_dim, out_dim


def run_cell(
    *,
    dataset: str,
    architecture: str,
    hidden_dim: int = 256,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 50,
    metric_name: str = "accuracy",
    batch_size: int = 32,
    seed: int = 0,
) -> dict:
    """Train one (dataset, architecture) cell to convergence, save artifacts.

    Returns a dict suitable for the per-method summary CSV row.
    """
    torch.manual_seed(seed)
    device = _device()

    meta = DATASET_META.get(dataset.lower())
    task = meta.task if meta else "node-classification"

    ds = load_dataset(dataset)
    in_dim, out_dim = _infer_dims(ds, task)
    # GAT default heads=8; allow override later if needed.
    model = build_model(architecture, in_dim, hidden_dim, out_dim)

    if task == "graph-classification":
        best_state, best_val, best_epoch = train_graph_classification(
            model, ds, device=device, lr=lr, weight_decay=weight_decay,
            epochs=epochs, batch_size=batch_size, patience=patience,
            metric_name=metric_name, seed=seed,
        )
        model.load_state_dict(best_state)
        test_metric = evaluate_test_graphs(model, device, batch_size=batch_size)
    else:
        best_state, best_val, best_epoch = train_node_classification(
            model, ds, device=device, lr=lr, weight_decay=weight_decay,
            epochs=epochs, patience=patience, metric_name=metric_name,
            seed=seed,
        )
        model.load_state_dict(best_state)
        test_metric = evaluate_test(model, ds, device, metric_name)

    # Persist artifacts.
    cell_dir = RESULTS_ROOT / dataset / architecture
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cell_dir / "checkpoint.pt"
    torch.save({
        "state_dict": best_state,
        "architecture": architecture,
        "dataset": dataset,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden_dim": hidden_dim,
        "seed": seed,
    }, ckpt_path)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        "sparsity": 0.0,
        "metric_name": metric_name,
        "metric_value": float(test_metric),
        "n_params": int(n_params),
        "epoch_of_best_val": int(best_epoch),
        "seed": int(seed),
    }
    (cell_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_path = RESULTS_ROOT / "plots" / \
        f"accuracy_vs_sparsity_{dataset}_{architecture}.png"
    plot_accuracy_vs_sparsity(
        dataset, architecture,
        sparsities=[0.0], values=[float(test_metric)],
        out_path=plot_path,
        metric_name=metric_name,
        method_label="no-pruning (dense)",
    )

    return {
        "dataset": dataset,
        "architecture": architecture,
        "sparsity": 0.0,
        "metric_name": metric_name,
        "metric_value": float(test_metric),
        "checkpoint_path": str(ckpt_path),
    }
