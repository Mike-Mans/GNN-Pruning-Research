#!/usr/bin/env bash
# Issue #1: dense-baseline sweep across all 30 cells.
#
#   ./scripts/run_no_pruning.sh                 # full sweep
#   ./scripts/run_no_pruning.sh --datasets cora,wisconsin   # subset
#
# All extra args are forwarded to `gnn_pruning.cli sweep` (e.g. --force).
set -euo pipefail

CONFIG="src/gnn_pruning/configs/no_pruning.yaml"

exec python -m gnn_pruning.cli sweep \
    --method no-pruning \
    --config "${CONFIG}" \
    "$@"
