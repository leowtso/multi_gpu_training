# Results log

Fill this in as you go. Raw numbers live in `results/runs.jsonl`;
`uv run python scripts/report.py` regenerates the tables.

## Setup

| | |
|---|---|
| Model | GPT-2 style decoder, 124M params (`124m`); larger presets up to 2.7B |
| Data | FineWeb-Edu `sample-10BT`, 100M train / 2M val tokens, GPT-2 BPE |
| Precision | bf16 autocast (single/DDP); FSDP2 `MixedPrecisionPolicy(param=bf16, reduce=fp32)` |
| Batch | 524,288 tokens/step (micro-batch 8 × 1024 tokens × grad-accum × GPUs) |
| Local box | 1× RTX 4070 Ti 12 GB, measured bf16 matmul ceiling 75 TFLOPS (MFU denominator) |
| Multi-GPU box | _TODO: GPU type × count, PCIe/NVLink (`results/hardware.txt`), NCCL all-reduce bus BW_ |

## 1. Single-GPU baseline (RTX 4070 Ti, 2026-11-16)

| config | tok/s | MFU | peak mem | notes |
|---|---|---|---|---|
| eager, micro-batch 16 | OOM | – | >12 GB | fp32 logits alone = 16×1024×50304×4 B = 3.07 GiB |
| eager, micro-batch 8 | 55.5k | 63% | 8.5 GiB | |
| `torch.compile`, micro-batch 8 | **68.9k** | **79%** | **5.5 GiB** | +24% throughput, −35% memory (fused loss/LayerNorm) |
| `torch.compile`, micro-batch 16 | 68.4k | 78% | 8.9 GiB | no gain: already compute-bound |
| 200-step run (105M tokens) | 66.8k | 76% | 5.5 GiB | sustained (thermals); train 10.96→6.05, **val loss 6.15** |

Profile (compiled, rank 0, `traces/124m_mbs8_compile_*.json`): GPU busy ≈99% of
wall time (944 ms kernel time per 951 ms step), ~75% of GPU time in GEMMs,
~7% flash-attention backward. No data-loader or CPU stall.

## 2. Wrapper overhead on 1 GPU (world size 1, no communication)

| strategy | tok/s | vs single | peak mem | notes |
|---|---|---|---|---|
| single | 68.9k | – | 5.5 GiB | |
| DDP | 62.7k | −9% | 5.9 GiB | |
| FSDP2 (ZeRO-3) | 56.1k | −19% | 8.1 GiB | OOM'd until `expandable_segments:True` (allocator fragmentation). Investigate: compile under FSDP? |

## 3. Scaling (TODO on the multi-GPU box)

`scripts/scale_sweep.sh` → `scripts/report.py`. Capture tok/s, scaling efficiency,
peak mem/GPU, MFU and GPU util for single / DDP / FSDP at 1, 2, 4, 8 GPUs.

## 4. Memory: DDP vs FSDP (TODO)

Largest model that trains with DDP vs FSDP on the rented GPUs
(`configs/fsdp_1.5b.toml`: ~23 GiB of fp32 weights+grads+Adam for 1.5B, more than a 24 GB card can hold for DDP).

## 5. Bottleneck story (TODO)

Before/after profiler screenshots go in `docs/`. Candidates the code has knobs for:
`--sync-every-microstep` (comm every micro-step vs `no_sync`), `--bucket-cap-mb`,
`--reshard-after-forward/--no-reshard-after-forward`, `--ac`, micro-batch size.

## 6. Fault tolerance

- CPU/gloo (2 ranks, FSDP): last rank hard-exits at step 5 → torchrun restarts →
  resumes from DCP checkpoint at step 3; post-resume losses **identical** to the
  uninterrupted run (`tests/test_distributed_cpu.py::test_kill_and_resume`).
- Found and fixed a torch 2.15 restart race: restarted workers reuse the agent's
  TCPStore and read the dead attempt's gloo peer addresses (~50% of restarts failed
  with `connectFullMesh ... Connection refused`). Fix: per-attempt `PrefixStore`
  (`parallel.py::_attempt_scoped_store`); 8/8 restarts succeed after.
- Async DCP save blocks the training loop 0.3–0.6 s per checkpoint (124M, 1 GPU).
- _TODO: same demo on GPUs (`scripts/fault_test.sh`)._
