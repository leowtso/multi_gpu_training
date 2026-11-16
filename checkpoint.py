"""Distributed checkpointing with torch.distributed.checkpoint (DCP).

The same code handles single-GPU, DDP and FSDP2: `get_state_dict` returns
sharded (DTensor) state for FSDP, so each rank writes only its own shard, and
DCP can reload into a *different* world size (resharding on load).

Layout:
  <ckpt_dir>/step_000050/      DCP shards + .metadata + train_state.json
  <ckpt_dir>/latest            name of the newest *complete* checkpoint
"""

import json
import os
import time

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from parallel import DistInfo


class Checkpointer:
    def __init__(self, ckpt_dir: str, info: DistInfo, async_save: bool = False):
        self.dir, self.info = ckpt_dir, info
        # async_save needs a CPU-capable process group (we create cuda:nccl,cpu:gloo).
        self.async_save = async_save and info.distributed
        self._pending = None  # (future, path, train_state) of an in-flight async save
        os.makedirs(ckpt_dir, exist_ok=True)

    def latest(self):
        marker = os.path.join(self.dir, "latest")
        if not os.path.exists(marker):
            return None
        with open(marker) as f:
            return os.path.join(self.dir, f.read().strip())

    def _commit(self, path, train_state):
        """Mark `path` complete. Only after this can a resume pick it up."""
        if self.info.is_main:
            with open(os.path.join(path, "train_state.json"), "w") as f:
                json.dump(train_state, f)
            tmp = os.path.join(self.dir, "latest.tmp")
            with open(tmp, "w") as f:
                f.write(os.path.basename(path))
            os.replace(tmp, os.path.join(self.dir, "latest"))  # atomic

    def wait(self):
        """Block until an in-flight async save is durable, then commit it."""
        if self._pending is not None:
            future, path, train_state = self._pending
            future.result()
            self._commit(path, train_state)
            self._pending = None

    def save(self, model, optimizer, train_state: dict) -> float:
        """Returns seconds the training loop was blocked."""
        t0 = time.perf_counter()
        self.wait()  # never have two async saves in flight
        model_sd, optim_sd = get_state_dict(model, optimizer)
        state = {"model": model_sd, "optim": optim_sd}
        path = os.path.join(self.dir, f"step_{train_state['step']:06d}")
        if self.async_save:
            # Copies state to pinned CPU memory, then writes in a background thread.
            self._pending = (dcp.async_save(state, checkpoint_id=path), path, train_state)
        else:
            dcp.save(state, checkpoint_id=path)
            if self.info.distributed:
                dist.barrier()
            self._commit(path, train_state)
        return time.perf_counter() - t0

    def load(self, model, optimizer):
        """Load the latest complete checkpoint in place; returns its train_state or None."""
        path = self.latest()
        if path is None:
            return None
        model_sd, optim_sd = get_state_dict(model, optimizer)
        state = {"model": model_sd, "optim": optim_sd}
        dcp.load(state, checkpoint_id=path)  # fills the (possibly sharded) tensors in place
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
        with open(os.path.join(path, "train_state.json")) as f:
            return json.load(f)
