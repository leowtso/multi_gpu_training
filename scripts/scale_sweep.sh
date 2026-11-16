#!/usr/bin/env bash
# Throughput / scaling-efficiency sweep: single-GPU baseline, then DDP and FSDP on 1..N GPUs.
#   scripts/scale_sweep.sh [max_steps] [extra train.py flags...]
#   e.g. scripts/scale_sweep.sh 30 --peak-tflops 243
# Writes results/sweep_<timestamp>.md (only this sweep's runs) and a full log next to it.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-30}
MAXGPU=$(nvidia-smi -L | wc -l)
COMMON=(--max-steps "$STEPS" --eval-every 0 "${@:2}")
START=$(date "+%Y-%m-%d %H:%M:%S")   # same clock/format as train.py's summary "time"
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results
{
  uv run python train.py --config configs/single_124m.toml "${COMMON[@]}"
  for n in 1 2 4 8; do
    [[ $n -le $MAXGPU ]] || break
    NGPU=$n scripts/launch.sh configs/ddp_124m.toml "${COMMON[@]}"
    NGPU=$n scripts/launch.sh configs/fsdp_124m.toml "${COMMON[@]}"
  done
} 2>&1 | tee "results/sweep_${STAMP}.log"
uv run python scripts/report.py --since "$START" --out "results/sweep_${STAMP}.md"
