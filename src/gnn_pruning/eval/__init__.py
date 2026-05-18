"""Metric helpers for the M1 pipelines.

`evaluate(model, data, ...)` returns a single scalar that matches the
per-dataset metric called out in the issues:
- accuracy: most single-label NC datasets and graph-classification
- macro_f1: heterophilic single-label NC (Cornell, Texas, Wisconsin, Actor)
- micro_f1: multi-label NC (Yelp, ogbn-proteins) with BCE-derived predictions
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()


def macro_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    return float(f1_score(y.cpu().numpy(), preds, average="macro",
                          zero_division=0))


def micro_f1_multilabel(logits: torch.Tensor, y: torch.Tensor,
                        threshold: float = 0.0) -> float:
    """Micro-F1 for multi-label targets. `logits` are pre-sigmoid scores.

    A label is predicted positive when logit > `threshold` (sigmoid > 0.5).
    """
    preds = (logits > threshold).int().cpu().numpy()
    return float(f1_score(y.int().cpu().numpy(), preds, average="micro",
                          zero_division=0))


def evaluate_node_classification(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor],
    metric_name: str,
) -> float:
    """Compute the configured metric over `mask` (or all nodes if None)."""
    if mask is not None:
        logits = logits[mask]
        y = y[mask]
    if metric_name == "accuracy":
        return accuracy(logits, y)
    if metric_name == "macro_f1":
        return macro_f1(logits, y)
    if metric_name == "micro_f1":
        return micro_f1_multilabel(logits, y)
    raise ValueError(f"Unknown metric {metric_name!r}")


def loss_fn(metric_name: str):
    """Pick the training loss to match the eval metric / label structure."""
    if metric_name == "micro_f1":
        # Multi-label: BCE on raw logits with float targets.
        def _bce(logits, y):
            return F.binary_cross_entropy_with_logits(logits, y.float())
        return _bce
    # Default: cross-entropy on logits.
    return F.cross_entropy
