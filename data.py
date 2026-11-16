"""Deterministic, rank-sharded, resumable loader over a flat uint16 token file."""

import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TokenLoader:
    """Serves (x, y) micro-batches of `micro_bs` windows of `seq_len` tokens.

    The token file is cut into non-overlapping windows, shuffled with a
    per-epoch permutation that every rank computes identically. Micro-batch
    `c` on rank `r` takes global sample indices

        (c * world_size + r) * micro_bs + [0, micro_bs)

    so ranks never see the same window, and the union over ranks at micro-step
    `c` is exactly what a single process with micro_bs * world_size would see.
    That makes DDP/FSDP losses directly comparable with a single-GPU run, and
    the whole loader state is one integer (`cursor`) for checkpoint/resume.
    """

    def __init__(self, split, micro_bs, seq_len, device, rank=0, world_size=1, seed=1337):
        self.data = np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
        self.B, self.T = micro_bs, seq_len
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.device = torch.device(device)
        self.num_windows = (len(self.data) - 1) // seq_len
        self.cursor = 0  # micro-batches consumed by this rank
        self._epoch, self._perm = -1, None

    def _window(self, g):
        epoch, i = divmod(g, self.num_windows)
        if epoch != self._epoch:
            self._epoch = epoch
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.num_windows)
        start = int(self._perm[i]) * self.T
        return self.data[start : start + self.T + 1]

    def next(self):
        base = (self.cursor * self.world_size + self.rank) * self.B
        self.cursor += 1
        buf = torch.from_numpy(np.stack([self._window(base + b) for b in range(self.B)]).astype(np.int64))
        if self.device.type == "cuda":
            buf = buf.pin_memory().to(self.device, non_blocking=True)
        return buf[:, :-1], buf[:, 1:]

    def state_dict(self, in_hand: int = 0):
        """Position as *global samples consumed*, so it survives a change of
        world size or micro-batch (elastic resume). `in_hand` = micro-batches
        already fetched but not yet trained on (the training loop prefetches one)."""
        return {"samples": (self.cursor - in_hand) * self.world_size * self.B}

    def load_state_dict(self, state):
        if "samples" not in state:  # plain per-rank cursor (also used to rewind the val loader)
            self.cursor = int(state["cursor"])
            return
        # Micro-step c covers global samples [c*W*B, (c+1)*W*B) across all ranks, so the
        # same global position maps to cursor = samples / (W*B) under any (W, B) that
        # divides it, and the resumed run sees exactly the samples a non-stop run would.
        per_step = self.world_size * self.B
        samples = int(state["samples"])
        if samples % per_step:
            raise ValueError(f"cannot resume at global sample {samples} with world_size*micro_bs={per_step}; "
                             "pick a micro-batch / GPU count that divides it")
        self.cursor = samples // per_step
