#!/usr/bin/env bash
# Fault-tolerance demo: the last rank hard-exits at step 30; torchrun tears the job
# down and restarts all ranks, which resume from the latest DCP checkpoint (step 20).
# Compare the loss after resume with an uninterrupted reference run.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf checkpoints/fault-tolerance checkpoints/fault-reference
MAX_RESTARTS=1 scripts/launch.sh configs/fault_tolerance.toml "$@" 2>&1 | tee results/fault_test.log
scripts/launch.sh configs/fault_tolerance.toml --fail-at-step -1 --tag fault-reference --ckpt-every 0 --no-resume "$@" \
  2>&1 | tee results/fault_reference.log
echo "--- loss after resume vs reference ---"
paste <(grep -E '^step +(40|50|59) \|' results/fault_test.log | tail -3 | cut -d'|' -f1-2) \
      <(grep -E '^step +(40|50|59) \|' results/fault_reference.log | cut -d'|' -f2)
