"""Forward-hook utility shared by every Wanda-family pipeline (#3, #4, #5).

Wanda-style scoring needs the *input* activation matrix `X^(ℓ)` at each
prunable layer ℓ. We attach a hook that captures `inputs[0]` (the tensor
passed into the conv) on a single forward pass.

Important architecture note from `PREFLIGHT_REPORT.md` §6 / issue #3:

- **GCN / SAGE**: `inputs[0]` at layer ℓ is the pre-conv feature matrix —
  exactly the matrix that the layer's weight `W^(ℓ)` is multiplied against.
  This is what Wanda needs.
- **GAT**: `GATConv` applies `W` *before* aggregation, so `inputs[0]` is the
  *raw, pre-aggregation* feature matrix at every layer. At layer 1 that's the
  original input X; at layer 2 it's the aggregated output of layer 1 (which is
  hidden_dim × num_heads wide because the heads are concatenated). Both cases
  still satisfy "the matrix W is multiplied against", which is the Wanda
  correctness condition.

`collect_activations(model, x, edge_index)` returns a `dict[layer_name → X]`
keyed by the full module path so callers can look up the captured tensor by
the same name they get from `named_prunable_weights(model)`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from gnn_pruning.models import prunable_layers


@contextmanager
def capture_inputs(model: nn.Module):
    """Context manager: yields a dict that fills with `name → input tensor`
    as `model` runs a forward pass."""
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for name, conv in prunable_layers(model):
        def make_hook(_name: str):
            def hook(_module, inputs, _output):
                x = inputs[0]
                # Detach + clone so the captured tensor doesn't pin the graph.
                activations[_name] = x.detach().clone()
            return hook
        handles.append(conv.register_forward_pre_hook(make_hook(name)))
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
    """Run one forward pass under hooks and return per-layer input tensors."""
    model.eval()
    with torch.no_grad(), capture_inputs(model) as bucket:
        if batch is not None:
            model(x, edge_index, batch=batch)
        else:
            model(x, edge_index)
    return dict(bucket)
