#!/usr/bin/env bash
# Issue #5: Wanda-Per-Class sweep — 30 cells × 9 sparsities.
set -euo pipefail

CONFIG="src/gnn_pruning/configs/wanda_per_class.yaml"

exec python -m gnn_pruning.cli sweep \
    --method wanda-per-class \
    --config "${CONFIG}" \
    "$@"
