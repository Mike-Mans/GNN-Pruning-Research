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


def run_cell(method: str, dataset: str, architecture: str, cfg_path: Path,
             checkpoint_dir: Optional[str] = None) -> dict:
    """Execute one cell and append its row(s) to `<method>/summary.csv`."""
    cfg = _load_config(cfg_path)
    metric_name = _metric_for(dataset, cfg)
    hp = _hyperparams_for(dataset, cfg)
    sparsity_grid = list(cfg.get("sparsity_grid") or [])

    method_root = Path(METHODS[method])
    method_root.mkdir(parents=True, exist_ok=True)
    summary_path = method_root / "summary.csv"

    if method == "no-pruning":
        from gnn_pruning.pruning.no_pruning import run_cell as _run
        row = _run(dataset=dataset, architecture=architecture,
                   metric_name=metric_name, **hp)
        rows = [row]
    elif method == "magnitude":
        from gnn_pruning.pruning.magnitude import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, **hp)
    elif method == "wanda-uniform":
        from gnn_pruning.pruning.wanda_uniform import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, **hp)
    elif method == "wanda-degree":
        from gnn_pruning.pruning.wanda_degree import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, **hp)
    elif method == "wanda-per-class":
        from gnn_pruning.pruning.wanda_per_class import run_cell as _run
        rows = _run(dataset=dataset, architecture=architecture,
                    metric_name=metric_name, sparsity_grid=sparsity_grid,
                    checkpoint_dir=checkpoint_dir, **hp)
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
                              n_sparsities: int) -> bool:
    root = Path(METHODS[method]) / dataset / architecture
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
    n_sparsities = len(cfg.get("sparsity_grid") or [])
    method_root = Path(METHODS[method])
    method_root.mkdir(parents=True, exist_ok=True)
    log_path = method_root / "run.log"

    summary_rows: list[dict] = []

    # Append-only run.log for the orchestrator.
    with log_path.open("a") as logf:
        logf.write(f"\n=== sweep start: method={method} cells={len(cells)} "
                   f"datasets_filter={datasets_filter} "
                   f"archs_filter={archs_filter} ===\n")
        for i, (dataset, arch) in enumerate(cells, start=1):
            if not force and _expected_outputs_present(
                method, dataset, arch, n_sparsities
            ):
                logf.write(f"[{i}/{len(cells)}] {dataset}/{arch}: SKIP "
                           f"(outputs already present)\n")
                logf.flush()
                continue
            t0 = time.time()
            logf.write(f"[{i}/{len(cells)}] {dataset}/{arch}: launching\n")
            logf.flush()
            cmd = [
                sys.executable, "-m", "gnn_pruning.cli", "run-cell",
                "--method", method,
                "--dataset", dataset,
                "--architecture", arch,
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
            # Collect rows from the cell's metrics.json.
            cell_metrics = method_root / dataset / arch / "metrics.json"
            if not cell_metrics.exists():
                continue
            m = json.loads(cell_metrics.read_text())
            if method == "no-pruning":
                ckpt = method_root / dataset / arch / "checkpoint.pt"
                summary_rows.append({
                    "dataset": dataset,
                    "architecture": arch,
                    "sparsity": m["sparsity"],
                    "metric_name": m["metric_name"],
                    "metric_value": m["metric_value"],
                    "checkpoint_path": str(ckpt),
                })
            else:
                for s, v in zip(m["sparsity_grid"], m["metric_values"]):
                    summary_rows.append({
                        "dataset": dataset,
                        "architecture": arch,
                        "sparsity": float(s),
                        "metric_name": m["metric_name"],
                        "metric_value": float(v),
                    })

    # Write summary.csv (rebuild from disk to be idempotent across runs).
    _rebuild_summary(method, n_sparsities)


def _rebuild_summary(method: str, n_sparsities: int) -> None:
    """Scan `results/<method>/*/*/metrics.json` and rewrite summary.csv."""
    method_root = Path(METHODS[method])
    rows: list[dict] = []
    for cell_metrics in sorted(method_root.glob("*/*/metrics.json")):
        m = json.loads(cell_metrics.read_text())
        dataset = cell_metrics.parent.parent.name
        arch = cell_metrics.parent.name
        if method == "no-pruning":
            ckpt = method_root / dataset / arch / "checkpoint.pt"
            rows.append({
                "dataset": dataset,
                "architecture": arch,
                "sparsity": m["sparsity"],
                "metric_name": m["metric_name"],
                "metric_value": m["metric_value"],
                "checkpoint_path": str(ckpt),
            })
        else:
            for s, v in zip(m["sparsity_grid"], m["metric_values"]):
                rows.append({
                    "dataset": dataset,
                    "architecture": arch,
                    "sparsity": float(s),
                    "metric_name": m["metric_name"],
                    "metric_value": float(v),
                })
    if not rows:
        return
    summary_path = method_root / "summary.csv"
    fieldnames = list(rows[0].keys())
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
