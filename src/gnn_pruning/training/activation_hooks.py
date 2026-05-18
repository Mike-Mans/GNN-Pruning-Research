"""Forward-hook utility shared by every Wanda-family pipeline (#3, #4, #5).

Wanda-style scoring needs the *input* activation matrix `X^(ℓ)` at each
prunable Linear ℓ. We hook the Linears themselves — not their parent conv —
because PyG SAGEConv routes different tensors into `lin_l` (aggregated
neighbor features) vs. `lin_r` (raw self features). Hooking the conv would
collapse those into one tensor and silently miscount for SAGE.

Important architecture notes (from PREFLIGHT_REPORT.md §6 / issue #3):

- **GCN / SAGE-lin_r**: `inputs[0]` is the pre-conv feature matrix — exactly
  the matrix the layer's weight is multiplied against.
- **SAGE-lin_l**: `inputs[0]` is the *aggregated neighbor features* (the
  conv runs propagate first, then applies `lin_l`). This is the right
  input for Wanda scoring of `lin_l.weight`.
- **GAT**: `GATConv` applies `W` *before* aggregation. So `inputs[0]` to the
  Linear is the raw, pre-aggregation feature matrix at every layer. At
  layer 1 this is the original input X; at layer 2 it is the aggregated
  output of layer 1 (which is hidden_dim × num_heads wide because heads are
  concatenated). Both still satisfy the Wanda condition.

`collect_activations(model, x, edge_index)` returns a `dict[name → X]` keyed
by the same name path as `named_prunable_weights(model)` (without the
trailing `.weight`).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from gnn_pruning.models import named_prunable_linears


@contextmanager
def capture_inputs(model: nn.Module):
    """Context manager: yields a dict that fills with `name → input tensor`
    as `model` runs a forward pass. Each entry is the input to one prunable
    Linear (not its parent conv)."""
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for name, lin in named_prunable_linears(model):
        def make_hook(_name: str):
            def hook(_module, inputs):
                x = inputs[0]
                # Detach + clone so the captured tensor doesn't pin the graph.
                activations[_name] = x.detach().clone()
            return hook
        handles.append(lin.register_forward_pre_hook(make_hook(name)))
    try:
        yield activations
    finally:
        for h in handles:
            h.remove()


def collect_activations(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Run one forward pass under hooks and return per-Linear input tensors."""
    model.eval()
    with torch.no_grad(), capture_inputs(model) as bucket:
        if batch is not None:
            model(x, edge_index, batch=batch)
        else:
            model(x, edge_index)
    return dict(bucket)
