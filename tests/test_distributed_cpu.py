"""CPU/gloo correctness tests for the distributed paths (no GPU needed, ~1-2 min).

  uv run pytest tests/ -v

1. DDP and FSDP on 2 processes give the same losses as one process with the
   same global batch (the loader hands both setups identical samples).
2. A rank killed mid-run is restarted by torchrun, resumes from the latest DCP
   checkpoint, and reproduces the uninterrupted loss curve.
"""

import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = ["--device", "cpu", "--model", "tiny", "--seq-len", "256", "--precision", "fp32",
          "--batch-tokens", "8192", "--max-steps", "8", "--log-every", "1", "--eval-every", "0",
          "--warmup-steps", "2", "--lr", "1e-3"]
ENV = {**os.environ, "OMP_NUM_THREADS": "4"}

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "train.bin")),
                                reason="run data/prepare.py first")


def run(cmd):
    out = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout + out.stderr


def losses(log):
    """{step: loss} from the training log (last occurrence wins, i.e. post-resume)."""
    return {int(s): float(l) for s, l in re.findall(r"^step\s+(\d+) \| loss ([\d.]+)", log, re.M)}


def torchrun(*args, nproc=2, restarts=0):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}",
            f"--max-restarts={restarts}", "train.py", *COMMON, *args]


@pytest.fixture(scope="module")
def single_losses():
    return losses(run([sys.executable, "train.py", *COMMON, "--strategy", "single", "--micro-bs", "8",
                       "--tag", "test-single"]))


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_matches_single_process(strategy, single_losses):
    got = losses(run(torchrun("--strategy", strategy, "--micro-bs", "4", "--tag", f"test-{strategy}")))
    assert got.keys() == single_losses.keys()
    for step in got:
        assert got[step] == pytest.approx(single_losses[step], abs=2e-4), f"step {step}"


def test_kill_and_resume(single_losses):
    shutil.rmtree(os.path.join(ROOT, "checkpoints", "test-fault"), ignore_errors=True)
    log = run(torchrun("--strategy", "fsdp", "--micro-bs", "4", "--tag", "test-fault", "--ckpt-every", "3",
                       "--resume", "--fail-at-step", "5", restarts=1))
    assert "fault injection" in log
    assert "resumed from checkpoints/test-fault/step_000003 at step 3" in log
    got = losses(log)
    for step in range(3, 8):
        assert got[step] == pytest.approx(single_losses[step], abs=2e-4), f"step {step}"
