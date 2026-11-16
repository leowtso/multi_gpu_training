# Results

Raw per-run JSON: `results/a100/runs.jsonl` (A100 session) and `results/runs.jsonl` (local);
regenerate tables with `uv run python scripts/report.py --file <runs.jsonl>`.
Profiler traces: `traces/a100/` (open in <https://ui.perfetto.dev>, summarise with `scripts/trace_stats.py`).

## Headline

| | |
|---|---|
| **DDP scaling (NVLink)** | 1 → 4× A100: 180.8k → 709.2k tok/s, **98% scaling efficiency** |
| **Same box, NVLink disabled** | DDP drops to **55%** efficiency; syncing every micro-step costs **−87%** (vs −10% with NVLink); FSDP ZeRO-3 **−84%** |
| **Profile-driven fix** | FSDP: Dynamo fell back to eager for the LM head + loss → refactor → **+12% throughput (611k → 685k tok/s), −43% memory/GPU (7.03 → 4.03 GiB)**; FSDP–DDP gap 14% → 3.5% |
| **Memory: 6.7B params** | DDP **OOM** on 80 GB (needs ~99 GiB of weights+grads+Adam per GPU); FSDP ZeRO-3 + activation checkpointing trains it at **47.2 GiB/GPU** |
| **Fault tolerance** | Rank killed at step 30 → torchrun restart → resume from DCP checkpoint; loss within 2e-4 of an uninterrupted run |
| **Elastic resume** | Checkpoint on 4 GPUs, resume on **2** (DCP reshards, loader remaps); loss within 3e-4 |

MFU below is relative to the **measured** bf16 matmul peak (A100: 243 TFLOPS, 4070 Ti: 75).
Against the A100 spec sheet (312 TFLOPS) multiply by 0.78: e.g. 4-GPU DDP 62.4% → **49%**, 6.7B FSDP 67.9% → **53%**.

## Setup

| | |
|---|---|
| Model | GPT-2 style decoder (pre-LN, SDPA flash attention, tied embeddings): 124M, 2.7B, 6.7B |
| Data | FineWeb-Edu `sample-10BT`, 100M train / 2M val tokens, GPT-2 BPE, uint16 memmap |
| Training | bf16 (autocast; FSDP2 `MixedPrecisionPolicy(param=bf16, reduce=fp32)`), fused AdamW, `torch.compile`, 524,288 tokens/step via grad accumulation |
| Cloud box | 4× A100-SXM4-80GB, NVLink NV12 between every pair, driver 595.91 / CUDA 13.2, 128 CPU, 2 TB RAM (RunPod) |
| Interconnect measured | `nccl-tests` all-reduce bus bandwidth: 222 GB/s at 1 GB, 155 GB/s at 32 MB, 84 GB/s at 8 MB |
| Local box | 1× RTX 4070 Ti 12 GB (development, single-GPU baseline, CPU/gloo tests) |
| Software | PyTorch 2.15 (cu132), NCCL 2.30.7, Python 3.13 |

## 1. Scaling: 124M, 4× A100 (NVLink)

Same global batch (0.5M tokens) at every GPU count; micro-batch 8, so grad accumulation shrinks with GPUs.

| strategy | 1 GPU | 2 GPUs | 4 GPUs | efficiency @4 | mem/GPU |
|---|---|---|---|---|---|
| single (baseline) | 180.8k | – | – | – | 5.51 GiB |
| DDP | 179.6k | 358.1k | **709.2k** | **98%** | 5.83 GiB |
| FSDP ZeRO-3 (before fix) | 160.6k | 311.7k | 610.8k | 84% | 7.03 GiB |
| FSDP ZeRO-3 (after fix, §3) | – | – | **684.6k** | **95%** | **4.03 GiB** |

On NVLink, the 0.5 GB fp32 gradient all-reduce takes a few ms against ~0.75 s of compute per step, so DDP
communication is effectively free. FSDP's original gap was already present on 1 GPU (no communication),
which pointed at per-step overhead rather than communication — see §3.

## 2. Interconnect: same box with NVLink disabled (`NCCL_P2P_DISABLE=1`)

Forces GPU↔GPU traffic through host memory, approximating a PCIe-only box (e.g. consumer GPUs without P2P).

| run (4 GPUs, 124M) | NVLink | no P2P | change |
|---|---|---|---|
| DDP (default: `no_sync` until last micro-step, 25 MB buckets) | 709.2k | 398.2k | −44% (98% → **55%** efficiency) |
| DDP, all-reduce every micro-step | 640.5k | **52.4k** | −92%; vs the no-P2P default: **−87%** |
| DDP, 1 MB buckets | – | 400.2k | ≈ default: bandwidth-bound, not latency-bound |
| FSDP ZeRO-3 (all-gathers params every micro-step) | 684.6k | **112.6k** | **−84%** |

Takeaways:
- With NVLink, communication strategy barely matters (syncing every micro-step costs 10%).
  Without it, **skipping gradient sync during accumulation is the difference between 398k and 52k tok/s**.
- Bucket size didn't matter here: once bandwidth is the limit, fewer/larger messages don't help.
- FSDP ZeRO-3 moves parameters on every micro-step (all-gather fwd + bwd, reduce-scatter grads),
  so on a slow interconnect it is ~3.5× slower than DDP for a model that fits anyway.
  Use FSDP for memory, not speed, unless the interconnect is fast.

## 3. Bottleneck story: FSDP under `torch.compile`

1. **Measure.** FSDP was 11% slower than single-GPU even on 1 GPU and used *more* memory than DDP (7.03 vs 5.83 GiB).
2. **Hypotheses tested** (4 GPUs):
   - Re-gathering params every micro-step → ZeRO-2 (`--no-reshard-after-forward`): **+3%** only.
   - Too many small micro-steps → micro-batch 32: **+17%** FSDP, **+12%** DDP (MFU 62 → 70%). Real, but affects both strategies.
3. **Profile** (`traces/a100/prof-fsdp4_*`, `scripts/trace_stats.py`): communication only **4.4%** exposed
   (rest overlapped with compute), but eager `cunn_SoftMaxForward/Backward` kernels (42 ms/3 steps) and eager
   `aten::mm` — the LM head and loss weren't compiled.
4. **Root cause.** FSDP2 hooks on each block cause a Dynamo graph break *inside* `for block in self.blocks`;
   a graph break inside a loop makes Dynamo run the rest of `forward()` eagerly, so final norm, LM head and
   cross-entropy (with fp32 B×T×vocab logits) ran uncompiled.
5. **Fix.** Move norm + LM head + loss into their own method (`GPT.head`), which Dynamo compiles as a separate frame.
6. **Result** (`prof-fsdp4-after_*`): 0 eager softmax kernels; **611k → 685k tok/s (+12%)**, **7.03 → 4.03 GiB/GPU (−43%)**;
   FSDP now within 3.5% of DDP. Per-block compilation (torchtitan style) was also tried: 664k, 3% slower than whole-model
   compile after the fix, so whole-model stays the default (`--compile-scope`).

Note: ~30–37% of profiled wall time shows idle GPU, but profiled steps take ~880 ms vs ~575 ms unprofiled,
so most of that gap is profiler overhead, not a training bottleneck.

## 4. Memory: DDP vs FSDP

| model | strategy | result | peak mem/GPU | tok/s |
|---|---|---|---|---|
| 6.7B (6.65B) | DDP, 4× A100 80GB | **OOM** — 78.07 GiB allocated (fp32 weights 24.8 + grads 24.8 GiB, Adam's 49.6 GiB not yet allocated) | – | – |
| 6.7B | FSDP ZeRO-3 + AC, 4× A100 | trains; val loss 10.97 @ step 10 | **47.16 GiB** | 15.9k (MFU 67.9% measured / 53% spec) |
| 2.7B | FSDP ZeRO-3, no AC | trains | 28.35 GiB | 42.6k |
| 2.7B | FSDP ZeRO-3 + activation checkpointing | trains | **19.12 GiB (−33%)** | 36.1k (**−15%**) |

Rule of thumb confirmed: fp32 master weights + grads + Adam = 16 B/param, so DDP on 80 GB caps out around
4–5B params before activations; FSDP divides that state by the number of GPUs.

## 5. Checkpointing, fault tolerance, elastic resume

| | result |
|---|---|
| Sharded DCP save, 6.7B (~80 GB incl. Adam), sync | blocked training **23.2 s** (~3.5 GB/s); retention kept latest + best-by-val-loss |
| DCP save, 124M | blocked ~1.1 s sync; async save 0.3–0.6 s (local 4070 Ti) |
| Kill + auto-restart (`scripts/fault_test.sh`) | last rank hard-exits at step 30 → torchrun `--max-restarts 1` restarts all ranks → resume from step 20; loss @50: 6.9460 vs 6.9458, @59: 6.9030 vs 6.9028 (reference) |
| Elastic 4 → 2 GPUs (`scripts/elastic_test.sh`) | resumed from step 20 on 2 GPUs (DCP reshards model + optimizer; loader maps global sample 10,240 to the new world size; grad accumulation doubles); loss @20/25: identical, @39: 7.1442 vs 7.1445 |
| CPU/gloo tests (`tests/`) | DDP and FSDP losses equal single-process to 4 decimals; kill+resume and 2→1 elastic resume reproduce the uninterrupted curve exactly |

Bugs found and fixed along the way:
- **torchrun restart race (torch 2.15):** restarted workers reuse the agent's TCPStore and could read the dead attempt's
  gloo peer addresses (~50% of restarts failed with `connectFullMesh ... Connection refused`). Fix: per-attempt
  `PrefixStore` (`parallel.py::_attempt_scoped_store`); 8/8 restarts succeed.
- **Allocator fragmentation:** FSDP OOM'd on a 12 GB card at ~8 GiB peak; `expandable_segments:True` fixed it.
- **Dataset streaming hang:** HF streaming left a prefetch thread that blocked interpreter exit on the 128-core pod.

## 6. Single-GPU baseline (RTX 4070 Ti, local)

| config | tok/s | MFU (of 75 TFLOPS) | peak mem |
|---|---|---|---|
| eager, micro-batch 16 | OOM | – | fp32 logits alone = 3.07 GiB |
| eager, micro-batch 8 | 55.5k | 63% | 8.5 GiB |
| `torch.compile`, micro-batch 8 | **68.9k (+24%)** | 79% | **5.5 GiB (−35%)** |
| 200-step run (105M tokens) | 66.8k sustained | 76% | 5.5 GiB; train loss 10.96 → 6.05, **val loss 6.15** |

Single-GPU profile: GPU busy ~99%, ~75% of GPU time in GEMMs, no data-loader stall.
