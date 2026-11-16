"""GPT training: single GPU, DDP or FSDP2.

Examples:
  uv run python train.py --config configs/single_124m.toml
  scripts/launch.sh configs/ddp_124m.toml                 # all local GPUs via torchrun
  scripts/launch.sh configs/fsdp_124m.toml --max-steps 50 # CLI flags override the config
  NGPU=2 scripts/launch.sh configs/ddp_124m.toml --device cpu --model tiny   # CPU/gloo smoke test

Every run appends a JSON summary line to results/runs.jsonl
(scripts/report.py turns that into the README/RESULTS table).
"""

import argparse
import json
import math
import os
import time
import tomllib
from contextlib import nullcontext

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # silence Triton 3.9 default-change warning
# FSDP allocates/frees each block's gathered params every step; without this the
# caching allocator fragments and 124M FSDP OOMs on a 12 GB card despite ~8 GiB peak.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from checkpoint import Checkpointer
from data import TokenLoader
from metrics import GpuUtilSampler
from model import build_model
from parallel import all_reduce_scalar, apply_strategy, grad_sync_context, setup_distributed, teardown_distributed


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="TOML file of defaults; CLI flags override it")
    p.add_argument("--tag", default="", help="label stored in results and used for the checkpoint dir")
    # model / data
    p.add_argument("--model", default="124m")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--micro-bs", type=int, default=8, help="sequences per GPU per forward/backward")
    p.add_argument("--batch-tokens", type=int, default=2**19, help="global tokens per optimizer step")
    # optimisation
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # parallelism / performance knobs
    p.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compile-scope", choices=["auto", "whole", "per-block"], default="auto",
                   help="whole: torch.compile(model); per-block: each block + head/loss separately. "
                        "auto = per-block for fsdp (FSDP2 hooks break a whole-model graph), whole otherwise")
    p.add_argument("--ac", action=argparse.BooleanOptionalAction, default=False, help="activation checkpointing per block")
    p.add_argument("--bucket-cap-mb", type=float, default=25.0, help="DDP gradient bucket size")
    p.add_argument("--reshard-after-forward", action=argparse.BooleanOptionalAction, default=True,
                   help="FSDP: True = ZeRO-3/FULL_SHARD, False = ZeRO-2/SHARD_GRAD_OP")
    p.add_argument("--sync-every-microstep", action="store_true",
                   help="ablation: all-reduce grads on every micro-step instead of only the last")
    # eval / logging / profiling
    p.add_argument("--eval-every", type=int, default=50, help="0 disables")
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--skip-steps", type=int, default=5, help="steps excluded from throughput averages")
    p.add_argument("--peak-tflops", type=float, default=75.0, help="per-GPU MFU denominator (75 = measured 4070 Ti)")
    p.add_argument("--profile", action="store_true", help="torch.profiler trace of rank 0 into traces/")
    p.add_argument("--profile-start", type=int, default=10)
    # checkpointing / fault tolerance
    p.add_argument("--ckpt-dir", default="", help="default: checkpoints/<tag or model-strategy>")
    p.add_argument("--ckpt-every", type=int, default=0, help="0 disables")
    p.add_argument("--async-ckpt", action="store_true", help="DCP async_save (distributed only)")
    p.add_argument("--keep-last", type=int, default=1, help="number of most recent checkpoints to keep")
    p.add_argument("--keep-best", action=argparse.BooleanOptionalAction, default=True,
                   help="also keep the checkpoint with the lowest val loss (needs eval on checkpoint steps)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="resume from the latest checkpoint in ckpt-dir, if any")
    p.add_argument("--fail-at-step", type=int, default=-1,
                   help="fault injection: last rank hard-exits at this step on the first attempt")
    # misc
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    return p


def get_args():
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    if pre.config:
        with open(pre.config, "rb") as f:
            cfg = {k.replace("-", "_"): v for k, v in tomllib.load(f).items()}
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            parser.error(f"unknown keys in {pre.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args()


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model, args, fused):
    # No weight decay on biases and LayerNorm params (anything < 2D).
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


@torch.no_grad()
def evaluate(model, loader, iters, autocast, info):
    """Every rank evaluates its own shard of the same fixed val batches; result is the global mean."""
    model.eval()
    loader.load_state_dict({"cursor": 0})  # same batches every eval -> comparable curves
    total = 0.0
    for _ in range(iters):
        x, y = loader.next()
        with autocast:
            total += model(x, y).item()
    model.train()
    return all_reduce_scalar(total / iters, info)


def make_profiler(args, info):
    if not (args.profile and info.is_main):
        return nullcontext(None)
    os.makedirs("traces", exist_ok=True)

    def on_ready(prof):
        name = (f"traces/{args.tag or args.model}_{args.strategy}{info.world_size}"
                f"_mbs{args.micro_bs}{'_compile' if args.compile else ''}_{int(time.time())}.json")
        prof.export_chrome_trace(name)
        print(f"\nprofiler trace -> {name}  (open in https://ui.perfetto.dev)")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=15))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if info.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(skip_first=args.profile_start, wait=0, warmup=2, active=3, repeat=1),
        on_trace_ready=on_ready,
        record_shapes=True,
    )


def main():
    args = get_args()
    if args.compile_scope == "auto":
        args.compile_scope = "per-block" if args.strategy == "fsdp" else "whole"
    info = setup_distributed(args.device)
    log = print if info.is_main else (lambda *a, **k: None)
    cuda = info.device.type == "cuda"
    torch.set_float32_matmul_precision("high")

    tokens_per_micro = args.micro_bs * args.seq_len * info.world_size
    assert args.batch_tokens % tokens_per_micro == 0, \
        f"batch-tokens {args.batch_tokens} not divisible by micro_bs*seq_len*world = {tokens_per_micro}"
    grad_accum = args.batch_tokens // tokens_per_micro

    # Same seed on every rank -> identical initial weights. DDP would broadcast
    # rank 0's anyway, but FSDP2 shards each rank's local copy, so it matters.
    torch.manual_seed(args.seed)
    raw_model = build_model(args.model, block_size=args.seq_len).to(info.device)
    n_params, flops_per_token = raw_model.num_params(), raw_model.flops_per_token(args.seq_len)

    model = apply_strategy(raw_model, args, info)
    optimizer = make_optimizer(model, args, fused=cuda)  # after wrapping: FSDP params are DTensor shards
    # per-block compilation already happened inside apply_strategy
    train_model = torch.compile(model) if args.compile and args.compile_scope == "whole" else model
    autocast = (torch.autocast(device_type=info.device.type, dtype=torch.bfloat16)
                if args.precision == "bf16" else nullcontext())

    log(f"{args.strategy} x{info.world_size} ({info.num_nodes} node) | model {args.model} {n_params / 1e6:.1f}M "
        f"| micro_bs {args.micro_bs} x grad_accum {grad_accum} x world {info.world_size} "
        f"= {args.batch_tokens:,} tokens/step | {args.precision} | "
        f"compile={args.compile and args.compile_scope} ac={args.ac}")

    train_loader = TokenLoader("train", args.micro_bs, args.seq_len, info.device, info.rank, info.world_size, args.seed)
    val_loader = TokenLoader("val", args.micro_bs, args.seq_len, info.device, info.rank, info.world_size, args.seed)

    ckpt = None
    start_step = 0
    if args.ckpt_every or args.resume:
        ckpt_dir = args.ckpt_dir or os.path.join("checkpoints", args.tag or f"{args.model}-{args.strategy}")
        ckpt = Checkpointer(ckpt_dir, info, async_save=args.async_ckpt,
                            keep_last=args.keep_last, keep_best=args.keep_best)
        if args.keep_best and args.ckpt_every and (not args.eval_every or args.ckpt_every % args.eval_every):
            log(f"warning: keep_best needs ckpt_every ({args.ckpt_every}) to be a multiple of eval_every "
                f"({args.eval_every}); checkpoints without a val loss can't be ranked, only keep_last applies")
        if args.resume:
            t0 = time.perf_counter()
            state = ckpt.load(model, optimizer)
            if state is not None:
                start_step = state["step"]
                train_loader.load_state_dict(state["loader"])
                log(f"resumed from {ckpt.latest()} at step {start_step} ({time.perf_counter() - t0:.1f}s)")
            else:
                log(f"--resume: no checkpoint in {ckpt_dir}, starting fresh")

    restart_count = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    util = GpuUtilSampler(info.device)

    step_times, ckpt_block_s, val_loss, val_step, loss_val = [], [], None, -1, float("nan")
    x, y = train_loader.next()
    with make_profiler(args, info) as prof:
        for step in range(start_step, args.max_steps):
            if step == args.fail_at_step and restart_count == 0 and info.rank == info.world_size - 1:
                print(f"[rank {info.rank}] fault injection: hard exit at step {step}", flush=True)
                os._exit(1)
            if step == start_step + args.skip_steps:
                util.start()

            t0 = time.perf_counter()
            lr = lr_at(step, args)
            for g in optimizer.param_groups:
                g["lr"] = lr

            loss_accum = torch.zeros((), device=info.device)
            for micro in range(grad_accum):
                last = micro == grad_accum - 1
                with grad_sync_context(model, args.strategy, last or args.sync_every_microstep):
                    with autocast:
                        loss = train_model(x, y) / grad_accum
                    # Fetch the next batch while the GPU is still busy with this forward.
                    x, y = train_loader.next()
                    loss.backward()
                loss_accum += loss.detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if hasattr(grad_norm, "full_tensor"):  # FSDP2 returns a DTensor
                grad_norm = grad_norm.full_tensor()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if prof is not None:
                prof.step()

            sync()  # time GPU work, not kernel launches
            dt = time.perf_counter() - t0
            if step >= start_step + args.skip_steps:
                step_times.append(dt)

            if step % args.log_every == 0 or step == args.max_steps - 1:
                loss_val = all_reduce_scalar(loss_accum.item(), info)
                tps = args.batch_tokens / dt
                mfu = flops_per_token * tps / (info.world_size * args.peak_tflops * 1e12)
                mem = all_reduce_scalar(torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0, info, "max")
                log(f"step {step:5d} | loss {loss_val:.4f} | lr {lr:.2e} | gnorm {grad_norm.item():.2f} "
                    f"| {dt * 1000:7.0f} ms | {tps / 1e3:7.1f}k tok/s | MFU {mfu:5.1%} | mem/gpu {mem:.2f} GiB")

            done = step == args.max_steps - 1
            # Eval/checkpoint "after N steps" (step + 1), so with ckpt_every a multiple of
            # eval_every every checkpoint carries the val loss of exactly those weights.
            if args.eval_every and ((step + 1) % args.eval_every == 0 or done):
                util.pause()
                val_loss, val_step = evaluate(train_model, val_loader, args.eval_iters, autocast, info), step
                log(f"step {step:5d} | val loss {val_loss:.4f}")
                util.start()

            if ckpt and args.ckpt_every and ((step + 1) % args.ckpt_every == 0 or done):
                util.pause()
                # x, y for the next step were already fetched, so one micro-batch
                # is "in hand": save cursor - 1 so resume re-fetches exactly it.
                state = {"step": step + 1, "loader": {"cursor": train_loader.cursor - 1},
                         "val_loss": val_loss if val_step == step else None}
                blocked = ckpt.save(model, optimizer, state)
                ckpt_block_s.append(blocked)
                log(f"step {step:5d} | checkpoint step_{step + 1:06d} ({'async' if ckpt.async_save else 'sync'}, "
                    f"blocked {blocked:.2f}s)")
                util.start()

    if ckpt:
        ckpt.wait()
    util.close()

    if step_times:
        avg_dt = sum(step_times) / len(step_times)
        tps = args.batch_tokens / avg_dt
        gpu_util = util.mean()
        summary = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": args.tag,
            "config": args.config,
            "strategy": args.strategy,
            "world_size": info.world_size,
            "nodes": info.num_nodes,
            "gpu": torch.cuda.get_device_name() if cuda else "cpu",
            "model": args.model,
            "params_m": round(n_params / 1e6, 1),
            "precision": args.precision,
            "compile": args.compile,
            "compile_scope": args.compile_scope if args.compile else None,
            "ac": args.ac,
            "reshard_after_forward": args.reshard_after_forward if args.strategy == "fsdp" else None,
            "bucket_cap_mb": args.bucket_cap_mb if args.strategy == "ddp" else None,
            "sync_every_microstep": args.sync_every_microstep,
            "micro_bs": args.micro_bs,
            "seq_len": args.seq_len,
            "grad_accum": grad_accum,
            "batch_tokens": args.batch_tokens,
            "steps": args.max_steps,
            "resumed_from": start_step or None,
            "step_ms": round(avg_dt * 1000, 1),
            "tokens_per_s": round(tps),
            "tokens_per_s_per_gpu": round(tps / info.world_size),
            "mfu": round(flops_per_token * tps / (info.world_size * args.peak_tflops * 1e12), 4),
            "peak_mem_gib": round(all_reduce_scalar(torch.cuda.max_memory_allocated() / 2**30, info, "max"), 2) if cuda else None,
            "peak_reserved_gib": round(all_reduce_scalar(torch.cuda.max_memory_reserved() / 2**30, info, "max"), 2) if cuda else None,
            "gpu_util": round(all_reduce_scalar(gpu_util, info), 1) if gpu_util is not None else None,
            "ckpt_block_s": round(sum(ckpt_block_s) / len(ckpt_block_s), 2) if ckpt_block_s else None,
            "final_loss": round(loss_val, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "profiled": args.profile,
        }
        if info.is_main:
            os.makedirs("results", exist_ok=True)
            with open("results/runs.jsonl", "a") as f:
                f.write(json.dumps(summary) + "\n")
            print("summary:", json.dumps(summary))

    teardown_distributed(info)


if __name__ == "__main__":
    main()
