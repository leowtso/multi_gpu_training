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
# Build against the NCCL that PyTorch ships in the venv (nvidia-nccl wheel), so we
# benchmark the exact library training uses and need no system libnccl-dev.
NCCL_HOME=$(uv run python -c "import nvidia.nccl as m; print(list(m.__path__)[0])")
ln -sf libnccl.so.2 "$NCCL_HOME/lib/libnccl.so"   # wheel ships only the soname; the linker wants libnccl.so
[[ -d nccl-tests ]] || git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
if [[ ! -x nccl-tests/build/all_reduce_perf ]]; then
  make -C nccl-tests -j CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" NCCL_HOME="$NCCL_HOME"
fi
LD_LIBRARY_PATH="$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}" \
  nccl-tests/build/all_reduce_perf -b 8M -e 1G -f 2 -g "$(nvidia-smi -L | wc -l)" | tee results/nccl_allreduce.txt
