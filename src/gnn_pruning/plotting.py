"""Shared plotting helpers for the M1 pipelines.

`plot_accuracy_vs_sparsity(...)` writes a PNG with the method's 9-point
curve plus dashed reference lines for whichever upstream methods' summary
CSVs exist on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
import pandas as pd


def _read_summary(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df


def _series_for(df: pd.DataFrame, dataset: str, architecture: str
                ) -> Optional[pd.DataFrame]:
    sub = df[(df["dataset"] == dataset) & (df["architecture"] == architecture)]
    if sub.empty:
        return None
    return sub.sort_values("sparsity")


def plot_accuracy_vs_sparsity(
    dataset: str,
    architecture: str,
    sparsities: Sequence[float],
    values: Sequence[float],
    out_path: Path,
    *,
    metric_name: str = "accuracy",
    method_label: str = "this method",
    reference_paths: Optional[dict[str, Path]] = None,
) -> None:
    """Write a single PNG to `out_path`.

    `reference_paths` maps `{label: Path-to-summary.csv}` for methods whose
    curves should be overlaid as dashed/dotted reference lines when their
    summary CSV exists. Missing files are silently skipped.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    # Reference overlays first so the focal curve renders on top.
    style_cycle = ["--", ":", "-.", (0, (3, 1, 1, 1))]
    if reference_paths:
        for i, (label, path) in enumerate(reference_paths.items()):
            df = _read_summary(Path(path))
            if df is None:
                continue
            series = _series_for(df, dataset, architecture)
            if series is None:
                continue
            ax.plot(
                series["sparsity"].to_numpy(),
                series["metric_value"].to_numpy(),
                linestyle=style_cycle[i % len(style_cycle)],
                linewidth=1.2,
                label=label,
                alpha=0.7,
            )

    ax.plot(list(sparsities), list(values), marker="o", linewidth=1.6,
            label=method_label, color="#1f77b4")
    ax.set_xlabel("Sparsity")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{architecture.upper()} on {dataset}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
