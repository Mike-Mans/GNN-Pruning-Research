"""Shared masking utility used by every pruning method (#2–#5).

`apply_per_layer_mask(weights_scores, sparsity)` is the canonical top-k
zeroing primitive. It mutates each weight tensor in-place by zeroing the
lowest-scoring `sparsity` fraction. Callers must save and restore the
original weights themselves (or work on clones) — this function does not
manage that lifecycle.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch


def per_layer_topk_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Return a boolean mask of `score`'s shape: True where to KEEP.

    The lowest-`sparsity` fraction of entries (by absolute score) is dropped.
    """
    if sparsity <= 0.0:
        return torch.ones_like(score, dtype=torch.bool)
    if sparsity >= 1.0:
        return torch.zeros_like(score, dtype=torch.bool)
    flat = score.flatten()
    k = int(round(sparsity * flat.numel()))
    if k == 0:
        return torch.ones_like(score, dtype=torch.bool)
    # `kthvalue` gives the k-th smallest value; entries strictly above the
    # threshold survive. Ties are broken arbitrarily (deterministic given
    # the same input order).
    threshold = flat.kthvalue(k).values
    keep = score > threshold
    return keep


def apply_mask_inplace(weight: torch.Tensor, keep: torch.Tensor) -> None:
    with torch.no_grad():
        weight.mul_(keep.to(weight.dtype))


@contextmanager
def masked_weights(
    weights_and_scores: Iterable[tuple[torch.Tensor, torch.Tensor]],
    sparsity: float,
):
    """Temporarily zero out the lowest-scoring `sparsity` fraction per layer.

    `weights_and_scores` yields `(W, score)` pairs where `W` is the live
    nn.Parameter and `score` has the same shape (e.g. `|W|` for magnitude,
    `|W| * a_q` for Wanda variants). Original weights are restored on exit.
    """
    pairs = list(weights_and_scores)
    originals = [w.detach().clone() for w, _ in pairs]
    for w, s in pairs:
        keep = per_layer_topk_mask(s, sparsity)
        apply_mask_inplace(w, keep)
    try:
        yield
    finally:
        with torch.no_grad():
            for (w, _), original in zip(pairs, originals):
                w.copy_(original)


def measure_per_layer_sparsity(weights: Iterable[torch.Tensor]) -> list[float]:
    """Fraction of zeros in each weight tensor — for the `metrics.json` log."""
    out = []
    for w in weights:
        total = w.numel()
        zeros = int((w == 0).sum().item())
        out.append(zeros / max(total, 1))
    return out
