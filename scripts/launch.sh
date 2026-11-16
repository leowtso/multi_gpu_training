#!/usr/bin/env bash
# Launch train.py on all local GPUs with torchrun.
#   scripts/launch.sh <config.toml> [extra train.py flags...]
# Env: NGPU (default: all visible GPUs), MAX_RESTARTS (default 0).
# Multi-node: set NNODES, NODE_RANK and MASTER_ADDR on every node (same command on each).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?usage: scripts/launch.sh <config.toml> [train.py flags...]}; shift
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NNODES=${NNODES:-1}
if [[ "$NNODES" -gt 1 ]]; then
  RDZV=(--nnodes "$NNODES" --node-rank "${NODE_RANK:?}" --rdzv-backend c10d --rdzv-endpoint "${MASTER_ADDR:?}:${MASTER_PORT:-29500}")
else
  RDZV=(--standalone)
fi
exec uv run torchrun "${RDZV[@]}" --nproc-per-node "$NGPU" --max-restarts "${MAX_RESTARTS:-0}" \
  train.py --config "$CONFIG" "$@"
