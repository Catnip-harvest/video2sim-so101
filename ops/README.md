# ops/: how the overnight runs were actually launched (9-10 Oct 2026, MI300X droplet)

Copies of host-side scripts (they ran on the HOST, outside git, and call `docker exec sprint ...`).
- `queue_bin/gpu_slot.sh`: blocks until <= N other GPU processes run. Measured: 3 concurrent GPU processes
  livelock/crawl on this box, 2 are fine, so every policy-eval job and extra training waits on it.
- `queue_bin/make_jobs.sh`: writes one model's P2.4 eval jobs (seed 3000 + seed 4000 none + 9 ladder rungs).
- `queue_bin/train_chain.sh`: ACT-wide (P2.5) then the P2.3 capped/reference runs, one training at a time.
- `queue_bin/p23_maxbatch.sh`: largest batch that fits each emulated memory cap.
- `queue_bin/p3_train.sh`: P3-A trainings, started only when the eval queue is empty (07:00 last start).
- `queue_jobs/*.sh`: every policy-inference job the serial queue ran, in name order (exact commands + flags).
- `p3a/`: the P3-A agent's CPU helpers (datagen runner, post-processing/tar, gate statistics).
The queue loop itself was `while true; do j=$(ls /root/work/queue/*.sh | sort | head -1); [ -n "$j" ] && { bash "$j"; mv "$j" /root/work/queue/done/; } || sleep 30; done`.
