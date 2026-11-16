"""Turn results/runs.jsonl into a markdown table with scaling efficiency.

  uv run python scripts/report.py                  # all non-profiled runs
  uv run python scripts/report.py --since "2026-11-16 14:00"
  uv run python scripts/report.py --model 124m --last 10

Scaling efficiency = tokens/s / (world_size * single-GPU tokens/s), using the
fastest single-GPU run with the same model, GPU, micro-batch, precision and compile setting.
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(v, spec=""):
    return "–" if v is None else format(v, spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=os.path.join(ROOT, "results", "runs.jsonl"))
    p.add_argument("--model")
    p.add_argument("--since", help="only runs at or after this timestamp")
    p.add_argument("--last", type=int, help="only the N most recent runs")
    args = p.parse_args()

    with open(args.file) as f:
        runs = [json.loads(line) for line in f if line.strip()]
    runs = [r for r in runs if not r.get("profiled") and r["gpu"] != "cpu"]  # drop profiler + CPU test runs
    if args.model:
        runs = [r for r in runs if r["model"] == args.model]
    if args.since:
        runs = [r for r in runs if r["time"] >= args.since]
    if args.last:
        runs = runs[-args.last :]

    def key(r):
        return (r["model"], r["gpu"], r["micro_bs"], r.get("precision", "bf16"), r["seq_len"], r.get("compile", False))

    baseline = {}
    for r in runs:
        if r["strategy"] == "single":
            k = key(r)
            baseline[k] = max(baseline.get(k, 0), r["tokens_per_s_per_gpu"] if "tokens_per_s_per_gpu" in r else r["tokens_per_s"])

    cols = ["tag", "strategy", "GPUs", "model", "options", "tok/s", "tok/s/GPU", "scaling eff.",
            "MFU", "peak mem/GPU (GiB)", "GPU util %", "val loss"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in runs:
        world = r.get("world_size", 1)
        base = baseline.get(key(r))
        eff = r["tokens_per_s"] / (world * base) if base else None
        opts = [r.get("precision", "bf16")]
        if r.get("compile"):
            opts.append("compile")
        if r.get("ac"):
            opts.append("AC")
        if r.get("strategy") == "fsdp":
            opts.append("ZeRO-3" if r.get("reshard_after_forward", True) else "ZeRO-2")
        if r.get("strategy") == "ddp" and r.get("bucket_cap_mb") not in (None, 25.0):
            opts.append(f"bucket {r['bucket_cap_mb']:g}MB")
        if r.get("sync_every_microstep"):
            opts.append("sync every micro-step")
        opts.append(f"mbs {r['micro_bs']}")
        gpu = r["gpu"].replace("NVIDIA ", "").replace("GeForce ", "")
        row = [
            r.get("tag") or "",
            r["strategy"],
            f"{world}× {gpu}" + (f" ({r['nodes']} nodes)" if r.get("nodes", 1) > 1 else ""),
            f"{r['model']} ({r['params_m']}M)",
            ", ".join(opts),
            f"{r['tokens_per_s']:,}",
            f"{r.get('tokens_per_s_per_gpu', r['tokens_per_s']):,}",
            fmt(eff, ".0%"),
            fmt(r.get("mfu"), ".1%"),
            fmt(r.get("peak_mem_gib")),
            fmt(r.get("gpu_util")),
            fmt(r.get("val_loss")),
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
