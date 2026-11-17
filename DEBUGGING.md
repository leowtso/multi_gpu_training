# Debugging log

The bugs hit while building and benchmarking this repo. Each entry gives the symptom, how it was
investigated, the root cause, the fix, and the evidence that it worked. The first four are the
ones worth talking about; the rest are smaller.

---

## 1. FSDP slower than DDP and using more memory: `torch.compile` graph break

**Symptom.** On 4× A100, FSDP reached 611k tok/s (84% scaling efficiency) against DDP's 709k (98%),
and used *more* memory per GPU than DDP (7.03 vs 5.83 GiB), even though FSDP shards the weights. The
gap already existed on **1 GPU**, where there is no communication at all (FSDP 160.6k vs single 180.8k).
The same gap had shown up on the local 4070 Ti.

**Investigation.**
1. *Wrong first guess:* I put the local gap down to eval and checkpoint overhead. That didn't explain
   the gap on the A100 runs, which used the same eval settings for both strategies, so I dropped it.
2. *The 1-GPU gap ruled out communication* as the main cause and pointed at per-step overhead.
3. *Tested the cheap hypotheses first* (4 GPUs):
   - Re-gathering parameters every micro-step → ZeRO-2 (`--no-reshard-after-forward`): **+3%**, so not it.
   - Too many small micro-steps → micro-batch 32: **+17%** FSDP, but also **+12%** DDP. Real, but not
     specific to FSDP.
4. *Profiled* (`--profile`, `traces/a100/prof-fsdp4_*`, summarised with `scripts/trace_stats.py`):
   - Only **4.4%** of communication was exposed (not overlapped with compute), so communication wasn't the problem.
   - There were **eager** `cunn_SoftMaxForward/Backward` kernels (42 ms over 3 steps) and eager
     `aten::mm`. Under `torch.compile`, cross-entropy should be a fused Triton kernel, so the LM head and
     loss were not being compiled.

**Root cause.** FSDP2 installs hooks on each sharded block, and Dynamo graph-breaks at those hooks
*inside* `for block in self.blocks`. When Dynamo graph-breaks inside a loop, it runs the **rest of
`forward()` eagerly**. So the final LayerNorm, the LM head and the cross-entropy ran uncompiled. That
also materialised the full fp32 `B×T×vocab` logits, which is where the extra memory came from.

**Fix.** Moved norm + LM head + loss into their own method, `GPT.head` ([model.py](model.py)).
Dynamo compiles a separate function as its own frame, so the head gets compiled even though the loop
before it graph-breaks.

**Evidence.**
- The after-fix trace (`prof-fsdp4-after_*`) has 0 eager softmax kernels.
- **611k → 685k tok/s (+12%)**, **7.03 → 4.03 GiB/GPU (−43%)**, and the FSDP–DDP gap went from 14% to 3.5%.
- I also tried per-block compile, as torchtitan does (`--compile-scope per-block`): 664k, 3% slower than
  whole-model compile after the fix, so whole-model stays the default.

**Lesson.** `torch.compile` fails silently: a graph break gives you slower code, not an error. Check
the profiler for eager kernels you expected to be fused. Next time, `TORCH_LOGS="graph_breaks"` would
list the breaks directly.

---

## 2. torchrun restarts failing about half the time: stale keys in the rendezvous store

**Symptom.** In the fault-tolerance test (one rank hard-exits, and `torchrun --max-restarts 1`
restarts every rank), about **50%** of restarts failed or hung with
`Gloo connectFullMesh ... Connection refused`.

**Investigation.**
- It failed intermittently, so it looked like a race rather than a logic error.
- I wanted to attach `py-spy` to the hung workers, but that wasn't permitted in the environment, so I
  wrote a **minimal reproduction** instead: a few lines that init a process group, kill one rank, and
  let torchrun restart them.
- PyTorch's comment says each restart attempt gets a fresh store. I checked whether the second
  attempt actually got a new `MASTER_PORT` or reused the first attempt's store. **It reused it:** in
  torch 2.15 the agent hosts the TCPStore, and restarted workers reconnect to the same one.

**Root cause.** The store still held the dead attempt's process-group keys, including the gloo peer
addresses. A restarted rank could read those **stale** addresses before the new ranks overwrote them,
then try to connect to a process that no longer existed.

**Fix.** Namespace every key by restart attempt: `PrefixStore(f"attempt_{TORCHELASTIC_RESTART_COUNT}", store)`
(`_attempt_scoped_store` in [parallel.py](parallel.py)), passed explicitly to `init_process_group`.

**Evidence.** **8/8** restarts succeeded afterwards. On the A100s, the fault test matched the
uninterrupted run to within 2e-4 (loss at step 50: 6.9460 vs 6.9458).

**Lesson.** For intermittent distributed failures, shrink the problem to a small reproduction and
check the documented assumptions ("fresh store per attempt") rather than trusting them.

---

## 3. Elastic resume didn't match the reference run: LR schedule changed

**Symptom.** In the 4 → 2 GPU elastic test, the resumed run's loss drifted from the uninterrupted
reference (10.3473 vs 10.3024), well beyond bf16 noise.

**Investigation.** The model, optimizer state and data position all restored correctly: the first
step after resuming was right, and the error grew afterwards. That pointed at something that changes
over the run instead of the restored state. The first half of the test had been launched with a smaller
`--max-steps`, so it would stop at the checkpoint.

**Root cause.** The cosine LR schedule is a function of `max_steps`. Stopping early by lowering
`max_steps` gave the first half a **different learning rate at every step** from the reference run.

**Fix.** Added `--stop-at-step`, which exits after a step without touching the schedule.
`scripts/elastic_test.sh` uses it.

**Evidence.** On 4× A100, 4 → 2 GPUs: identical loss at steps 20 and 25, and 7.1442 vs 7.1445 at step 39.
The CPU test (2 → 1 rank) reproduces the uninterrupted curve exactly.

**Lesson.** "Stop early" and "train for fewer steps" are different experiments when hyperparameters
depend on the run length.

---

## 4. FSDP out of memory on a 12 GB card at ~8 GiB peak: allocator fragmentation

**Symptom.** 1-GPU FSDP OOM'd on the local RTX 4070 Ti, while the logged peak of live tensors was only about 8 GiB.

**Investigation.** Allocated memory was well below capacity while *reserved* memory was near it, so the
memory was fragmented rather than full. FSDP repeatedly allocates and frees differently sized
all-gather buffers, which leaves gaps the caching allocator can't reuse.

**Fix.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set by default in `train.py`. Segments
can then grow in place instead of needing a new contiguous block.

**Evidence.** The same run then fits. The setting stayed on for every A100 run.

---

## Smaller bugs

| Bug | Cause | Fix |
|---|---|---|
| `view size is not compatible with input tensor's size and stride` in the loss | The targets are a slice (`buf[:, 1:]`), so they aren't contiguous and `.view()` can't be used | `targets.reshape(-1)` |
| Eager micro-batch 16 OOM on 12 GB | The fp32 logits alone are 3.07 GiB | Micro-batch 8, or `torch.compile` (fuses the loss). Kept as a data point in RESULTS.md. |
| `data/prepare.py` hung after writing its output on the 128-core pod | HF `datasets` streaming left a prefetch thread running, which blocked interpreter exit | Checked the output files were complete, then `os._exit(0)` after flushing |
| `nccl-tests` build failed: `nccl.h: No such file` | The pod image has no system NCCL; PyTorch's NCCL lives in the `nvidia-nccl` pip wheel | Build with `NCCL_HOME` pointing at the wheel, add a `libnccl.so → libnccl.so.2` symlink, and set `LD_LIBRARY_PATH` (`scripts/cloud_setup.sh`). Result: 222 GB/s bus bandwidth. |
| `fault_test.sh` compared the wrong lines | The grep for `step N` also matched checkpoint log lines | Match `^step N \| loss` |
| MFU reported as 174% | `--peak-tflops` wasn't passed, so it used the 4070 Ti peak | Pass the measured A100 peak (243 TFLOPS) → 53.5% |
| Wrong scaling efficiency in `report.py` | The baseline included world-size-1 DDP/FSDP runs and mixed eager with compiled runs | Baseline = `strategy == "single"` only, keyed on model, GPU, micro-batch, precision, sequence length and compile |
| `git` refused to run on the pod ("dubious ownership") | `rsync` preserved the local UID | `chown -R root:root` on the checkout |

## Notes
- **`nvidia-smi` shows 10.6 GB used while the log says 5.5 GiB:** `nvidia-smi` counts everything reserved (caching allocator, CUDA context, compile workspaces); the log counts live tensors.
- **The 6.7B DDP OOM:** expected. 16 B/param × 6.65B ≈ 99 GiB exceeds 80 GB. This is the point of the FSDP comparison.
- **30–37% idle GPU in the profiles:** profiled steps take ~880 ms vs ~575 ms unprofiled, so most of that idle time is profiler overhead, not a training bottleneck.
