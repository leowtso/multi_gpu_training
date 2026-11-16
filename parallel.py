"""Process-group setup and parallelism strategies (single / DDP / FSDP2)."""

import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistInfo:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def num_nodes(self):
        return self.world_size // self.local_world_size


def setup_distributed(device: str) -> DistInfo:
    """Initialise from torchrun's env vars; plain `python train.py` gives a 1-process DistInfo."""
    if "RANK" not in os.environ:
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.cuda.set_device(dev.index or 0)
        return DistInfo(False, 0, 0, 1, 1, dev)

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world, local_world = int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_WORLD_SIZE"])
    store = _attempt_scoped_store(rank, world)
    if device.startswith("cuda"):
        dev = torch.device("cuda", local_rank)
        torch.cuda.set_device(dev)
        # NCCL for GPU tensors, gloo for CPU-side collectives (needed by DCP async save).
        dist.init_process_group("cuda:nccl,cpu:gloo", store=store, rank=rank, world_size=world, device_id=dev)
    else:
        dev = torch.device("cpu")
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
    return DistInfo(True, rank, local_rank, world, local_world, dev)


def _attempt_scoped_store(rank, world):
    """TCPStore with keys namespaced by torchrun restart attempt.

    After `torchrun --max-restarts` restarts the workers, they reconnect to the
    *same* agent-hosted store (same MASTER_PORT), which still holds the dead
    attempt's process-group keys (peer addresses). Restarted ranks can read
    those stale keys before the new ones are written and fail with
    "Gloo connectFullMesh ... Connection refused" (observed ~50% of restarts
    with torch 2.15). Prefixing keys per attempt removes the race.
    """
    attempt = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    agent_hosts_store = os.environ.get("TORCHELASTIC_USE_AGENT_STORE") == "True"
    base = dist.TCPStore(
        os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), world,
        is_master=(not agent_hosts_store and rank == 0),
    )
    return dist.PrefixStore(f"attempt_{attempt}", base)


def teardown_distributed(info: DistInfo):
    if info.distributed:
        dist.barrier()
        dist.destroy_process_group()


def apply_strategy(model, args, info: DistInfo):
    """Wrap `model` per args.strategy. Returns the wrapped module (not compiled)."""
    if args.ac:
        # Activation checkpointing: keep only each block's input, recompute the
        # block's internals during backward. Trades ~1/3 more compute for memory.
        for i, block in enumerate(model.blocks):
            model.blocks[i] = checkpoint_wrapper(block)

    if args.compile and args.compile_scope == "per-block":
        # Compile each block and the head+loss separately (torchtitan's approach),
        # *before* fully_shard. Compiling the whole FSDP2 model instead makes Dynamo
        # break the graph at FSDP's root hooks, so the LM head and loss ran eager:
        # unfused softmax kernels and fp32 logits materialised (seen in the A100 trace).
        # Module.compile() is in place, so state-dict keys don't change.
        for block in model.blocks:
            block.compile()
        model.head = torch.compile(model.head)

    if args.strategy == "single":
        assert info.world_size == 1, "strategy=single with >1 process; use ddp or fsdp"
        return model

    assert info.distributed, f"strategy={args.strategy} needs torchrun (see scripts/launch.sh)"

    if args.strategy == "ddp":
        return DDP(
            model,
            device_ids=[info.device.index] if info.device.type == "cuda" else None,
            bucket_cap_mb=args.bucket_cap_mb,  # gradient all-reduce bucket size
            gradient_as_bucket_view=True,  # grads alias the comm buckets: saves one copy of grads
        )

    if args.strategy == "fsdp":
        mesh = init_device_mesh(info.device.type, (info.world_size,))
        param_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
        # Params are all-gathered in bf16 (half the traffic); gradients are
        # reduce-scattered in fp32 for accuracy. Master weights/optimizer stay fp32.
        mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
        # Shard each transformer block as its own unit so only one block's full
        # params are materialised at a time (plus the prefetched next one).
        # reshard_after_forward=True  -> ZeRO-3 / FULL_SHARD (re-gather in backward)
        # reshard_after_forward=False -> ZeRO-2 / SHARD_GRAD_OP (keep params gathered)
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=args.reshard_after_forward)
        fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=args.reshard_after_forward)
        return model

    raise ValueError(f"unknown strategy {args.strategy}")


def grad_sync_context(model, strategy, sync: bool):
    """Skip gradient communication on all but the last micro-step of accumulation."""
    if strategy == "ddp":
        return nullcontext() if sync else model.no_sync()
    if strategy == "fsdp":
        model.set_requires_gradient_sync(sync)
    return nullcontext()


def all_reduce_scalar(value: float, info: DistInfo, op: str = "mean") -> float:
    if not info.distributed:
        return value
    t = torch.tensor(float(value), dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
    return t.item() / (info.world_size if op == "mean" else 1)
