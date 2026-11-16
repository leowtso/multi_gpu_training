# multi_gpu_training

A small, hand-written GPT training stack for learning multi-GPU training:
single GPU → DDP → FSDP2 → 2D (FSDP + tensor parallel), with profiling,
distributed checkpointing and fault-tolerant resume. Numbers: [RESULTS.md](RESULTS.md).

## Setup

```bash
uv sync                                              # Python 3.13, torch 2.15 (cu132)
uv run python data/prepare.py --train-tokens 100e6   # ~1 min, writes data/{train,val}.bin
uv run pytest tests/ -q                              # CPU/gloo: DDP & FSDP == single, kill+resume
```

On a rented multi-GPU box: `bash scripts/cloud_setup.sh` (installs, records
`nvidia-smi topo -m`, matmul TFLOPS and NCCL all-reduce bandwidth into `results/`).

## Run

Experiments are TOML configs in `configs/`; any CLI flag overrides them.

```bash
uv run python train.py --config configs/single_124m.toml        # single GPU
scripts/launch.sh configs/ddp_124m.toml                         # torchrun on all local GPUs
NGPU=4 scripts/launch.sh configs/fsdp_124m.toml --max-steps 50
scripts/launch.sh configs/fsdp_1.5b.toml                        # model too big for DDP
scripts/scale_sweep.sh 30                                       # single + DDP/FSDP at 1,2,4,8 GPUs
scripts/fault_test.sh                                           # kill a rank, auto-resume, compare loss
scripts/elastic_test.sh                                         # checkpoint on 4 GPUs, resume on 2
uv run python scripts/report.py                                 # markdown results table
```

Multi-node: run `scripts/launch.sh` on every node with `NNODES`, `NODE_RANK`, `MASTER_ADDR` set.

Profiling: add `--profile` (with a small `--batch-tokens`, e.g. 65536) and open
`traces/*.json` in <https://ui.perfetto.dev>.

Every run appends a JSON summary to `results/runs.jsonl`: tokens/s (global and
per GPU), MFU, peak memory per GPU (max over ranks), NVML GPU utilisation,
checkpoint blocking time, loss and val loss.

## Knobs

| flag | what it changes |
|---|---|
| `--strategy single\|ddp\|fsdp` | parallelism |
| `--precision bf16\|fp32` | autocast / FSDP `MixedPrecisionPolicy` |
| `--compile` | `torch.compile` the (wrapped) model |
| `--ac` | activation checkpointing per transformer block |
| `--bucket-cap-mb` | DDP gradient bucket size |
| `--[no-]reshard-after-forward` | FSDP ZeRO-3 (default) vs ZeRO-2 |
| `--sync-every-microstep` | ablation: all-reduce on every micro-step instead of `no_sync` |
| `--ckpt-every`, `--async-ckpt`, `--resume` | DCP checkpointing / resume |
| `--keep-last N`, `--[no-]keep-best` | retention: keep the newest N + the lowest-val-loss checkpoint, delete the rest |
| `--compile-scope whole\|per-block` | whole-model vs per-block `torch.compile` |
| `--stop-at-step` | exit cleanly mid-schedule (preemption; LR schedule unchanged) |
| `--fail-at-step` | fault injection (last rank exits on the first attempt) |
| `--peak-tflops` | MFU denominator: set to `scripts/gpucheck.py`'s number for the GPU |

## Files

| file | what |
|---|---|
| `data/prepare.py` | dataset download + tokenisation |
| `model.py` | GPT-2 style model, SDPA flash attention, FLOPs/token for MFU |
| `data.py` | deterministic rank-sharded loader; one-integer state for resume |
| `parallel.py` | process group setup, DDP / FSDP2 / activation-checkpoint wrapping |
| `checkpoint.py` | DCP save/load (sharded, async, atomic `latest` marker) |
| `metrics.py` | NVML GPU-utilisation sampler |
| `train.py` | training loop, eval, profiler, results logging |
| `scripts/` | launch, sweep, fault + elastic tests, cloud setup, report, gpucheck, trace_stats |
| `tests/` | CPU distributed correctness + resume tests |

## Roadmap

- [x] Single-GPU baseline, measured and profiled
- [x] DDP + FSDP2, verified equal to single-process on CPU
- [x] DCP checkpointing with retention, kill + auto-resume, elastic resume on fewer GPUs
- [x] 4× A100 NVLink: scaling sweep, NVLink-off comparison, FSDP compile fix, 6.7B DDP-vs-FSDP memory, AC trade-off
- [ ] 2D parallelism (FSDP + tensor parallel) on an 8-GPU NVLink node
- [ ] Meta-device init (build the model sharded, not full-size on every GPU) for >7B
- [ ] Batch-1 inference latency (eager vs compile + CUDA graphs vs TensorRT)
