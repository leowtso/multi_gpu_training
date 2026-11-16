"""Sanity-check a GPU box: versions, achievable bf16 matmul TFLOPS (use as --peak-tflops), NCCL init.

  uv run python scripts/gpucheck.py
"""
import sys
import time

import torch
import torch.distributed as dist

print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cudnn", torch.backends.cudnn.version(), "| nccl", torch.cuda.nccl.version())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu {i}: {p.name} | cc {p.major}.{p.minor} | {p.total_memory / 2**30:.1f} GiB | {p.multi_processor_count} SMs")

x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    x @ x
torch.cuda.synchronize()
n, t = 20, time.perf_counter()
for _ in range(n):
    x @ x
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / n
print(f"bf16 matmul 8192^3 on gpu 0: {2 * 8192**3 / dt / 1e12:.1f} TFLOPS")

dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1)
t = torch.ones(1, device="cuda")
dist.all_reduce(t)
print("nccl single-rank all_reduce ok:", t.item())
dist.destroy_process_group()
