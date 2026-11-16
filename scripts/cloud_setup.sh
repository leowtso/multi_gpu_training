#!/usr/bin/env bash
# Bootstrap a rented multi-GPU box. Run from the repo root after cloning:
#   bash scripts/cloud_setup.sh
# Captures the hardware facts the results need: GPU model/count, interconnect
# topology, achievable matmul TFLOPS (MFU denominator) and NCCL bus bandwidth.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
uv sync

{ nvidia-smi; echo; nvidia-smi topo -m; } | tee results/hardware.txt
uv run python scripts/gpucheck.py | tee -a results/hardware.txt   # use its TFLOPS as --peak-tflops

[[ -f data/train.bin ]] || uv run python data/prepare.py --train-tokens 100e6

# NCCL all-reduce bandwidth: the ceiling for DDP gradient sync on this box.
if [[ ! -x nccl-tests/build/all_reduce_perf ]]; then
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
  make -C nccl-tests -j CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
nccl-tests/build/all_reduce_perf -b 8M -e 1G -f 2 -g "$(nvidia-smi -L | wc -l)" | tee results/nccl_allreduce.txt
