"""Issue #4: Wanda-Degree-Weighted — `S_pq = |W_pq| · ‖X̃_q‖_2`, X̃=√deg·X.

Same activation collection as Wanda-Uniform (one forward pass under hooks),
but each row of X is scaled by √deg before the column-wise L2 norm. Hub
nodes get up-weighted because they aggregate more strongly during message
passing.

Degree is computed from the *original* (pre-self-loop) edge_index of the
training subgraph — self-loops are excluded as a graph property (not a model
implementation detail). The default is logged in `metrics.json` so it's
verifiable post-hoc.

For graph-classification (BBBP, Proteins): degree is per-graph (PyG's
`degree(...)` on the batched `edge_index` is already correct because node
ids are re-indexed per batch); activation rows and degree rows align.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.utils import degree, remove_self_loops
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


RESULTS_ROOT = Path("results/wanda-degree")
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


def _node_degree(edge_index: torch.Tensor, num_nodes: int,
                 include_self_loops: bool = False) -> torch.Tensor:
    ei = edge_index
    if not include_self_loops:
        ei, _ = remove_self_loops(ei)
    return degree(ei[0], num_nodes=num_nodes).to(torch.float32)


def _degree_summary(deg: torch.Tensor) -> dict:
    return {
        "mean": float(deg.mean().item()),
        "median": float(deg.median().item()),
        "max": float(deg.max().item()),
        "p99": float(torch.quantile(deg.float(), 0.99).item()),
    }


def _collect_node_classification(
    model: torch.nn.Module, data, device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    data = data.to(device)
    model = model.to(device).eval()
    acts = collect_activations(model, data.x, data.edge_index)
    train_mask, _, _ = get_node_masks(data)
    train_mask = train_mask.to(device)
    n_used = int(train_mask.sum().item())
    deg = _node_degree(data.edge_index, num_nodes=data.num_nodes).to(device)
    # Restrict to training-mask rows for both activations and degree weighting.
    deg_used = deg[train_mask]
    acts_used = {k: v[train_mask] for k, v in acts.items()}
    return acts_used, deg_used, n_used


def _collect_graph_classification(
    model: torch.nn.Module, dataset, device: torch.device, seed: int,
    batch_size: int = 64,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    train_set, _, _ = _split_graphs(dataset, seed=seed)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    buckets: dict[str, list[torch.Tensor]] = {}
    deg_chunks: list[torch.Tensor] = []
    n_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        acts = collect_activations(
            model, batch.x.float(), batch.edge_index, batch=batch.batch
        )
        for k, v in acts.items():
            buckets.setdefault(k, []).append(v)
        deg = _node_degree(batch.edge_index, num_nodes=batch.x.shape[0]).to(device)
        deg_chunks.append(deg)
        n_nodes += int(batch.x.shape[0])
    merged_acts = {k: torch.cat(v, dim=0) for k, v in buckets.items()}
    merged_deg = torch.cat(deg_chunks, dim=0)
    return merged_acts, merged_deg, n_nodes


def _wanda_degree_score_pairs(
    model: torch.nn.Module,
    activations: dict[str, torch.Tensor],
    deg: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Score each prunable Linear by |W| * ||X̃_q||_2 where X̃ = √deg · X."""
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    sqrt_deg = deg.clamp(min=0).sqrt()
    for name, lin in named_prunable_linears(model):
        if name not in activations:
            raise RuntimeError(f"Missing activation for layer {name!r}")
        x = activations[name]
        # Row-scale by sqrt(deg); column L2 norm gives shape (in_dim,).
        x_tilde = x * sqrt_deg.to(x.device, dtype=x.dtype).unsqueeze(1)
        a = x_tilde.norm(dim=0, p=2).to(lin.weight.device, dtype=lin.weight.dtype)
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
        _, _, test_set = _split_graphs(ds, seed=int(ck.get("seed", 0)))
        model._test_split = test_set  # type: ignore[attr-defined]
        activations, deg, n_used = _collect_graph_classification(
            model, ds, device, seed=int(ck.get("seed", 0))
        )
    else:
        ds = load_dataset(dataset)
        activations, deg, n_used = _collect_node_classification(
            model, ds, device
        )

    deg_summary = _degree_summary(deg)
    pairs = _wanda_degree_score_pairs(model, activations, deg)

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
        "degree_summary": deg_summary,
        "self_loops_included": False,
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
        method_label="wanda-degree",
        reference_paths={
            "dense (#1)": DENSE_ROOT / "summary.csv",
            "magnitude (#2)": Path("results/magnitude/summary.csv"),
            "wanda-uniform (#3)": Path("results/wanda-uniform/summary.csv"),
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
