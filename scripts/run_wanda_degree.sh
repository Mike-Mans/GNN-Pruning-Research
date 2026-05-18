#!/usr/bin/env bash
# Issue #4: Wanda-Degree-Weighted sweep — 30 cells × 9 sparsities.
set -euo pipefail

CONFIG="src/gnn_pruning/configs/wanda_degree.yaml"

exec python -m gnn_pruning.cli sweep \
    --method wanda-degree \
    --config "${CONFIG}" \
    "$@"
