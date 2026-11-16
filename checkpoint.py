"""Distributed checkpointing with torch.distributed.checkpoint (DCP).

The same code handles single-GPU, DDP and FSDP2: `get_state_dict` returns
sharded (DTensor) state for FSDP, so each rank writes only its own shard, and
DCP can reload into a *different* world size (resharding on load).

Layout:
  <ckpt_dir>/step_000050/      DCP shards + .metadata + train_state.json
  <ckpt_dir>/latest            name of the newest *complete* checkpoint
  <ckpt_dir>/best.json         {"name", "step", "val_loss"} of the lowest-val-loss checkpoint

Retention: after each checkpoint is committed, everything except the newest
`keep_last` checkpoints and the best one is deleted. Deletion happens only after
the new checkpoint is durable and `latest` points at it, so a crash at any
moment leaves at least one complete checkpoint. Peak disk use is therefore
(keep_last + 1 [being written] + 1 [best, if older]) checkpoints.
"""

import json
import os
import shutil
import time

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from parallel import DistInfo


class Checkpointer:
    def __init__(self, ckpt_dir: str, info: DistInfo, async_save: bool = False,
                 keep_last: int = 1, keep_best: bool = True):
        assert keep_last >= 1, "keep_last must be >= 1 (resume needs the latest checkpoint)"
        self.dir, self.info = ckpt_dir, info
        # async_save needs a CPU-capable process group (we create cuda:nccl,cpu:gloo).
        self.async_save = async_save and info.distributed
        self.keep_last, self.keep_best = keep_last, keep_best
        self._pending = None  # (future, path, train_state) of an in-flight async save
        os.makedirs(ckpt_dir, exist_ok=True)
        self.best = self._read_json("best.json")  # survives restarts

    def _read_json(self, name):
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _write_atomic(self, name, text):
        tmp = os.path.join(self.dir, name + ".tmp")
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, os.path.join(self.dir, name))

    def latest(self):
        marker = os.path.join(self.dir, "latest")
        if not os.path.exists(marker):
            return None
        with open(marker) as f:
            return os.path.join(self.dir, f.read().strip())

    def _commit(self, path, train_state):
        """Mark `path` complete (only then can a resume pick it up), update best, prune."""
        if not self.info.is_main:
            return
        name = os.path.basename(path)
        with open(os.path.join(path, "train_state.json"), "w") as f:
            json.dump(train_state, f)
        self._write_atomic("latest", name)
        val = train_state.get("val_loss")
        if self.keep_best and val is not None and (self.best is None or val < self.best["val_loss"]):
            self.best = {"name": name, "step": train_state["step"], "val_loss": val}
            self._write_atomic("best.json", json.dumps(self.best))
        removed = self._prune(name)
        if removed:
            print(f"checkpoint retention: removed {', '.join(removed)}"
                  f" (kept last {self.keep_last}" + (f", best {self.best['name']}" if self.best else "") + ")")

    def _prune(self, latest_name):
        # Only consider committed checkpoints (<= latest); never touch one being written.
        committed = sorted(d for d in os.listdir(self.dir)
                           if d.startswith("step_") and d <= latest_name and os.path.isdir(os.path.join(self.dir, d)))
        keep = set(committed[-self.keep_last:])
        if self.keep_best and self.best:
            keep.add(self.best["name"])
        removed = [d for d in committed if d not in keep]
        for d in removed:
            shutil.rmtree(os.path.join(self.dir, d))
        return removed

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
