"""Issue #2: magnitude pruning (|W| baseline).

Per (dataset, architecture) cell: load dense checkpoint → score each layer
by `|W|` → for each `s ∈ {0.1, …, 0.9}` apply a per-layer top-k mask and
re-evaluate on the test split. No retraining (Wanda protocol).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from gnn_pruning.data import DATASET_META, load_dataset
from gnn_pruning.models import build_model, named_prunable_weights
from gnn_pruning.plotting import plot_accuracy_vs_sparsity
from gnn_pruning.pruning import masked_weights
from gnn_pruning.pruning.no_pruning import _infer_dims  # noqa: PLC2701
from gnn_pruning.training import evaluate_test, evaluate_test_graphs


RESULTS_ROOT = Path("results/magnitude")
DENSE_ROOT = Path("results/no-pruning")


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_dense(dataset: str, architecture: str,
                checkpoint_dir: str | None) -> tuple[dict, dict]:
    root = Path(checkpoint_dir) if checkpoint_dir else DENSE_ROOT
    ck_path = root / dataset / architecture / "checkpoint.pt"
    if not ck_path.exists():
        raise FileNotFoundError(
            f"Dense checkpoint missing at {ck_path}. Run issue-#1 first."
        )
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    return ck, {"checkpoint_path": str(ck_path)}


def _build_from_checkpoint(architecture: str, ck: dict) -> torch.nn.Module:
    model = build_model(architecture, ck["in_dim"], ck["hidden_dim"], ck["out_dim"])
    model.load_state_dict(ck["state_dict"])
    return model


def _score_magnitude(model: torch.nn.Module
                     ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return `[(W, |W|), ...]` for every prunable weight in the model."""
    pairs = []
    for _name, w in named_prunable_weights(model):
        pairs.append((w, w.detach().abs()))
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

    ck, prov = _load_dense(dataset, architecture, checkpoint_dir)
    model = _build_from_checkpoint(architecture, ck).to(device)
    pairs = _score_magnitude(model)

    meta = DATASET_META.get(dataset.lower())
    task = meta.task if meta else "node-classification"

    if task == "graph-classification":
        # Replay the same train/val/test split so eval is consistent.
        # We re-split with the same seed; the model's `_test_split` is then
        # the held-out set used by `evaluate_test_graphs`.
        from gnn_pruning.training import _split_graphs  # noqa: PLC2701
        ds = load_dataset(dataset)
        _, _, test_set = _split_graphs(ds, seed=int(ck.get("seed", 0)))
        model._test_split = test_set  # type: ignore[attr-defined]
    else:
        ds = load_dataset(dataset)

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
    layer_info = [
        {"name": name, "shape": list(w.shape), "numel": int(w.numel())}
        for name, w in named_prunable_weights(model)
    ]
    metrics = {
        "sparsity_grid": list(map(float, sparsity_grid)),
        "metric_name": metric_name,
        "metric_values": metric_values,
        # Pruning is per-layer uniform: every prunable weight is pruned to the
        # same target ratio `s` for each row of the grid. Record the
        # structural layer info so downstream tooling can verify the cell.
        "per_layer_sparsity": layer_info,
        "checkpoint": prov["checkpoint_path"],
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
        method_label="magnitude",
        reference_paths={"dense (#1)": DENSE_ROOT / "summary.csv"},
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
