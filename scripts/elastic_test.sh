#!/usr/bin/env bash
# Elastic resume demo: train FSDP on 4 GPUs, checkpoint at step 20, "lose" half the GPUs,
# resume on 2 GPUs (DCP reshards model + optimizer state on load; the data loader maps its
# global sample position to the new world size; grad accumulation doubles to keep the
# global batch). Compare with an uninterrupted 4-GPU reference.
set -euo pipefail
cd "$(dirname "$0")/.."
C=(--max-steps 40 --eval-every 0 --log-every 5 "$@")
rm -rf checkpoints/elastic checkpoints/elastic-ref
NGPU=4 scripts/launch.sh configs/fsdp_124m.toml "${C[@]}" --tag elastic --ckpt-every 20 --stop-at-step 20 \
  2>&1 | tee results/elastic_4gpu.log | grep -E "^step|checkpoint"
NGPU=2 scripts/launch.sh configs/fsdp_124m.toml "${C[@]}" --tag elastic --resume \
  2>&1 | tee results/elastic_2gpu.log | grep -E "resumed|elastic|^step"
NGPU=4 scripts/launch.sh configs/fsdp_124m.toml "${C[@]}" --tag elastic-ref \
  2>&1 | tee results/elastic_ref.log | grep -E "^step" >/dev/null
echo "--- loss: 4->2 GPU elastic resume vs uninterrupted 4 GPU ---"
paste <(grep -E '^step +(20|25|30|35|39) \| loss' results/elastic_2gpu.log | cut -d'|' -f1-2) \
      <(grep -E '^step +(20|25|30|35|39) \| loss' results/elastic_ref.log | cut -d'|' -f2)
rm -rf checkpoints/elastic
