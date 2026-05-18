#!/usr/bin/env bash
# Issue #3: Wanda-Uniform sweep — 30 cells × 9 sparsities.
#   ./scripts/run_wanda_uniform.sh
#   ./scripts/run_wanda_uniform.sh --datasets cora,wisconsin
set -euo pipefail

CONFIG="src/gnn_pruning/configs/wanda_uniform.yaml"

exec python -m gnn_pruning.cli sweep \
    --method wanda-uniform \
    --config "${CONFIG}" \
    "$@"
