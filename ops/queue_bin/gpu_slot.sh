#!/usr/bin/env bash
# Block until at most ($1, default 1) OTHER GPU processes are running, so the box never has
# more than 2 GPU processes at once. Measured 9 Oct: 3 concurrent GPU processes (2 trainings +
# 1 eval, or 2 evals + training, or 4-8 eval replicas) livelock (0 CU occupancy, ~160 % CPU spin)
# or crawl (ACT 12 -> 1.7 step/s, SmolVLA 2 -> 0.3 step/s); 2 processes are fine.
# Counts: lerobot-train python processes + sim_eval.py --mode policy python processes.
# Usage (first line of every queue job / extra training launcher): bash /root/work/queue/bin/gpu_slot.sh [max_others]
MAX="${1:-1}"
count() {
  ps -eo args | grep -E '^(/opt/venv/bin/)?python[0-9.]* .*(lerobot-train|eval/sim_eval.py --mode policy)' | grep -v -c grep
}
while [ "$(count)" -gt "$MAX" ]; do
  echo "$(date -u +%T) gpu_slot: $(count) GPU processes running (max others $MAX); waiting" >&2
  sleep 60
done
