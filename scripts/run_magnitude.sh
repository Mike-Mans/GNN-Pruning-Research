#!/usr/bin/env bash
# Issue #2: magnitude pruning sweep — 30 cells × 9 sparsities.
#   ./scripts/run_magnitude.sh                              # full sweep
#   ./scripts/run_magnitude.sh --datasets cora,wisconsin    # subset
set -euo pipefail

CONFIG="src/gnn_pruning/configs/magnitude.yaml"

exec python -m gnn_pruning.cli sweep \
    --method magnitude \
    --config "${CONFIG}" \
    "$@"
