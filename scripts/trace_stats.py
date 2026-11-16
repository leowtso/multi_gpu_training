"""Usage: uv run python scripts/trace_stats.py traces/<trace>.json

Summarise a torch.profiler chrome trace: GPU kernel time by category and how much
NCCL communication time overlaps with compute kernels (exposed vs hidden)."""
import json, sys
from collections import defaultdict

ev = json.load(open(sys.argv[1]))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
t0, t1 = min(e["ts"] for e in k), max(e["ts"] + e["dur"] for e in k)

def cat(n):
    n = n.lower()
    if "nccl" in n:
        return "nccl: " + ("allgather" if "allgather" in n else "reducescatter" if "reducescatter" in n else "allreduce" if "allreduce" in n else "other")
    if "gemm" in n or "sm80_xmma" in n or "cutlass" in n or "ampere" in n: return "gemm"
    if "flash" in n: return "flash-attn"
    if "triton" in n: return "triton (fused elementwise/norm/loss)"
    if "adam" in n or "multi_tensor" in n: return "optimizer"
    return "other"

tot = defaultdict(float)
for e in k: tot[cat(e["name"])] += e["dur"]

def union(iv):
    iv.sort(); out = []
    for s, e in iv:
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out

comp = union([[e["ts"], e["ts"] + e["dur"]] for e in k if "nccl" not in e["name"].lower()])
comm = union([[e["ts"], e["ts"] + e["dur"]] for e in k if "nccl" in e["name"].lower()])
def length(iv): return sum(e - s for s, e in iv)
# Exposed communication = NCCL time not covered by any compute kernel.
def subtract(a, b):
    """Total length of the union intervals `a` minus the union intervals `b` (both sorted)."""
    total, j = 0.0, 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while cur < e and k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                total += b[k][0] - cur
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            total += e - cur
    return total

exposed = subtract(comm, comp)
wall = t1 - t0
busy = length(union(comp + comm))
print(f"profiled window: {wall/1e3:.0f} ms (all ranks' kernels in this trace = rank 0 only)")
print(f"GPU busy (any kernel): {busy/wall:.1%} | idle gaps: {(wall-busy)/1e3:.0f} ms ({(wall-busy)/wall:.1%})")
print(f"compute kernels: {length(comp)/1e3:.0f} ms | NCCL kernels: {length(comm)/1e3:.0f} ms "
      f"| exposed NCCL (no compute running): {exposed/1e3:.1f} ms = {exposed/wall:.1%} of wall")
print("\nkernel time by category (sum of durations):")
for c, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"  {c:40s} {v/1e3:8.1f} ms  {v/sum(tot.values()):6.1%}")
