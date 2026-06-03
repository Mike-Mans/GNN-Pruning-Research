"""CLI driver for the M1 pipelines.

Two subcommands:

    python -m gnn_pruning.cli run-cell --method <m> --dataset <d> --architecture <a>
        Runs a single (dataset, architecture) cell in this process — used
        by the orchestrator as the body of a `subprocess.run(...)` call.

    python -m gnn_pruning.cli sweep --config <yaml>
        Reads the config, enumerates the grid, and launches one fresh
        subprocess per (dataset, architecture) cell. **Subprocess-per-cell
        isolation is mandatory** to avoid the MPS driver-memory creep that
        caused a segfault in pre-flight when looping architectures in one
        process.

Optional `--datasets` / `--architectures` flags on `sweep` restrict the
grid at runtime — that's how the smoke-test slice is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


METHODS = {
    "no-pruning": "results/no-pruning",
    "magnitude": "results/magnitude",
    "wanda-uniform": "results/wanda-uniform",
    "wanda-degree": "results/wanda-degree",
    "wanda-per-class": "results/wanda-per-class",
}

# Large NC datasets run full-batch and may OOM on 24 GB MPS (issue #6). We
# order them LAST in the sweep so an OOM there never costs the already-finished
# small / heterophilic cells (the publishable core).
LARGE_DATASETS = {"reddit", "ogbn-products", "ogbn-arxiv", "yelp", "flickr"}


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _metric_for(dataset: str, cfg: dict) -> str:
    overrides = cfg.get("metric_overrides") or {}
    return overrides.get(dataset, cfg.get("default_metric", "accuracy"))


def _hyperparams_for(dataset: str, cfg: dict) -> dict:
    overrides = (cfg.get("hyperparam_overrides") or {}).get(dataset, {})
    base = dict(cfg.get("default_hyperparams") or {})
    base.update(overrides)
    return base


def _cells_from_config(cfg: dict) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for arch, datasets in (cfg.get("cells") or {}).items():
        for d in datasets:
            cells.append((d, arch))
    return cells


def _resolve_seeds(dataset: str, cfg: dict) -> list[int]:
    """Seed list for `dataset`. Large NC datasets stay single-seed (their
    minibatch training is run once); everything else uses the config `seeds`
    list (multi-seed for error bars, issue #8)."""
    if dataset in LARGE_DATASETS:
        return [0]
    return [int(s) for s in (cfg.get("seeds") or [0])]


def _resolve_splits(dataset: str, cfg: dict) -> list[int]:
    """Split columns to run for `dataset` (issue #9). WebKB / Actor ship 10
    Geom-GCN splits; `splits: { <dataset>: all }` runs and averages over all of
    them. Datasets not listed run the single default split [0]."""
    spec = (cfg.get("splits") or {}).get(dataset)
    if spec is None:
        return [0]
    if spec == "all":
        return list(range(10))
    return [int(s) for s in spec]


def _expand_runs(cfg: dict, cells: list[tuple[str, str]]
                 ) -> list[tuple[str, str, int, int]]:
    """Expand each (dataset, arch) cell into (dataset, arch, seed, split) runs,
    ordering large datasets last (issue #6 guard)."""
    runs: list[tuple[str, str, int, int]] = []
    for dataset, arch in cells:
        for seed in _resolve_seeds(dataset, cfg):
            for split in _resolve_splits(dataset, cfg):
                runs.append((dataset, arch, seed, split))
    # Stable sort: False (0) before True (1) → large datasets last, original
    # order preserved within each group.
    runs.sort(key=lambda r: r[0] in LARGE_DATASETS)
    return runs


def run_cell(method: str, dataset: str, architecture: str, cfg_path: Path,
             seed: int = 0, split: int = 0,
             checkpoint_dir: Optional[str] = None) -> dict:
    """Execute one (dataset, arch, seed, split) run; write its metrics.json."""
    cfg = _load_config(cfg_path)
    metric_name = _metric_for(dataset, cfg)
    hp = _hyperparams_for(dataset, cfg)
    hp.pop("seed", None)  # seed comes from the sweep dimension, not the hp dict
    sparsity_grid = list(cfg.get("sparsity_grid") or [])

    method_root = Path(METHODS[method])
    method_root.mkdir(parents=True, exist_ok=True)
    summary_path = method_root / "summary.csv"

    if method == "no-pruning":
        from gnn_pruning.pruning.no_pruning import run_cell as _run
        row = _run(dataset=dataset, architecture=architecture,
                   metric_name=metric_name, seed=seed, split=split, **hp)
        rows = [row]
    elif method == "magnitude":
        from gnn_pruning.pruning.magnitude import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, seed=seed, split=split, **hp)
    elif method == "wanda-uniform":
        from gnn_pruning.pruning.wanda_uniform import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, seed=seed, split=split, **hp)
    elif method == "wanda-degree":
        from gnn_pruning.pruning.wanda_degree import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, seed=seed, split=split, **hp)
    elif method == "wanda-per-class":
        from gnn_pruning.pruning.wanda_per_class import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, seed=seed, split=split, **hp)
    else:
        raise ValueError(f"Unknown method {method!r}")

    return {"rows": rows, "summary_path": str(summary_path)}


def _filter_cells(cells, datasets_filter, archs_filter):
    if datasets_filter:
        keep_d = set(d.lower() for d in datasets_filter)
        cells = [(d, a) for (d, a) in cells if d.lower() in keep_d]
    if archs_filter:
        keep_a = set(a.lower() for a in archs_filter)
        cells = [(d, a) for (d, a) in cells if a.lower() in keep_a]
    return cells


def _expected_outputs_present(method: str, dataset: str, architecture: str,
                              seed: int, split: int,
                              n_sparsities: int) -> bool:
    root = (Path(METHODS[method]) / dataset / architecture
            / f"seed-{seed}" / f"split-{split}")
    metrics = root / "metrics.json"
    if not metrics.exists():
        return False
    if method == "no-pruning":
        return (root / "checkpoint.pt").exists()
    # Pruning methods: confirm metrics.json has the full sparsity grid.
    try:
        m = json.loads(metrics.read_text())
        return len(m.get("metric_values", [])) == n_sparsities
    except Exception:
        return False


def sweep(method: str, cfg_path: Path,
          datasets_filter: Optional[list[str]] = None,
          archs_filter: Optional[list[str]] = None,
          force: bool = False,
          checkpoint_dir: Optional[str] = None) -> None:
    cfg = _load_config(cfg_path)
    cells = _filter_cells(_cells_from_config(cfg), datasets_filter, archs_filter)
    runs = _expand_runs(cfg, cells)
    n_sparsities = len(cfg.get("sparsity_grid") or [])
    method_root = Path(METHODS[method])
    method_root.mkdir(parents=True, exist_ok=True)
    log_path = method_root / "run.log"

    # Append-only run.log for the orchestrator.
    with log_path.open("a") as logf:
        logf.write(f"\n=== sweep start: method={method} runs={len(runs)} "
                   f"datasets_filter={datasets_filter} "
                   f"archs_filter={archs_filter} ===\n")
        for i, (dataset, arch, seed, split) in enumerate(runs, start=1):
            tag = f"{dataset}/{arch}/seed-{seed}/split-{split}"
            if not force and _expected_outputs_present(
                method, dataset, arch, seed, split, n_sparsities
            ):
                logf.write(f"[{i}/{len(runs)}] {tag}: SKIP "
                           f"(outputs already present)\n")
                logf.flush()
                continue
            t0 = time.time()
            logf.write(f"[{i}/{len(runs)}] {tag}: launching\n")
            logf.flush()
            cmd = [
                sys.executable, "-m", "gnn_pruning.cli", "run-cell",
                "--method", method,
                "--dataset", dataset,
                "--architecture", arch,
                "--seed", str(seed),
                "--split", str(split),
                "--config", str(cfg_path),
            ]
            if checkpoint_dir:
                cmd += ["--checkpoint-dir", checkpoint_dir]
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                proc = subprocess.run(cmd, env=env, check=False,
                                      capture_output=True, text=True)
            except Exception as e:
                logf.write(f"  ERROR: subprocess raised {e!r}\n")
                logf.flush()
                continue
            dt = time.time() - t0
            if proc.returncode != 0:
                logf.write(
                    f"  FAILED in {dt:.1f}s (rc={proc.returncode})\n"
                    f"  stdout-tail:\n{proc.stdout[-2000:]}\n"
                    f"  stderr-tail:\n{proc.stderr[-2000:]}\n"
                )
                logf.flush()
                continue
            logf.write(f"  done in {dt:.1f}s\n")
            logf.flush()

    # Write summary.csv (rebuild from disk to be idempotent across runs).
    _rebuild_summary(method, n_sparsities)


def _rebuild_summary(method: str, n_sparsities: int) -> None:
    """Scan per-(seed, split) metrics and rewrite summary.csv as the MEAN over
    all seeds × splits per (dataset, arch, sparsity) row (issues #8, #9)."""
    method_root = Path(METHODS[method])
    # (dataset, arch, sparsity) -> list of metric values across seeds/splits.
    agg: dict[tuple[str, str, float], list[float]] = {}
    metric_names: dict[tuple[str, str], str] = {}
    for cell_metrics in sorted(
        method_root.glob("*/*/seed-*/split-*/metrics.json")
    ):
        m = json.loads(cell_metrics.read_text())
        arch = cell_metrics.parent.parent.parent.name
        dataset = cell_metrics.parent.parent.parent.parent.name
        metric_names[(dataset, arch)] = m["metric_name"]
        if method == "no-pruning":
            agg.setdefault((dataset, arch, 0.0), []).append(
                float(m["metric_value"]))
        else:
            for s, v in zip(m["sparsity_grid"], m["metric_values"]):
                agg.setdefault((dataset, arch, float(s)), []).append(float(v))
    if not agg:
        return
    rows: list[dict] = []
    for (dataset, arch, sparsity), vals in sorted(agg.items()):
        mean = sum(vals) / len(vals)
        # Population std across seeds × splits (0 for single-run cells).
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        rows.append({
            "dataset": dataset,
            "architecture": arch,
            "sparsity": sparsity,
            "metric_name": metric_names[(dataset, arch)],
            "metric_value": mean,
            "metric_std": std,
            "n_runs": len(vals),
        })
    summary_path = method_root / "summary.csv"
    fieldnames = ["dataset", "architecture", "sparsity",
                  "metric_name", "metric_value", "metric_std", "n_runs"]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="gnn_pruning.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-cell")
    p_run.add_argument("--method", required=True, choices=list(METHODS))
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--architecture", required=True)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--split", type=int, default=0)
    p_run.add_argument("--config", required=True, type=Path)
    p_run.add_argument("--checkpoint-dir", default=None,
                       help="Override the dense-checkpoint directory for "
                            "pruning methods (default: results/no-pruning).")

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--config", required=True, type=Path)
    p_sweep.add_argument("--method", required=True, choices=list(METHODS))
    p_sweep.add_argument("--datasets", default=None,
                         help="Comma-separated subset of datasets to run.")
    p_sweep.add_argument("--architectures", default=None,
                         help="Comma-separated subset of archs to run.")
    p_sweep.add_argument("--force", action="store_true",
                         help="Re-run even if outputs already exist.")
    p_sweep.add_argument("--checkpoint-dir", default=None)

    args = parser.parse_args(argv)

    if args.command == "run-cell":
        run_cell(args.method, args.dataset, args.architecture, args.config,
                 seed=args.seed, split=args.split,
                 checkpoint_dir=args.checkpoint_dir)
        return 0
    if args.command == "sweep":
        ds_filter = args.datasets.split(",") if args.datasets else None
        a_filter = args.architectures.split(",") if args.architectures else None
        sweep(args.method, args.config,
              datasets_filter=ds_filter, archs_filter=a_filter,
              force=args.force, checkpoint_dir=args.checkpoint_dir)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
