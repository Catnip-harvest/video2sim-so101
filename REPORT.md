Money: alive 12.5 h = $24.79 now (21:24 UTC). If destroyed at 14:00 UTC Sat $57.81 | 16:00 $61.79 | 20:00 $69.75 (cap $60)
# SO-101 overnight report - updated 21:25 UTC (Sat 04:25 Hanoi)
Status: RUNNING
Now running (work-22): policy queue job 01 (MolmoAct2 seed 4000 baseline, ~12 min), then ACT and SmolVLA 3-episode smoke evals; training ACT (30k steps) and SmolVLA (20k steps, ~3.3 h) outside the queue. OOD ladder code in progress.
Next: P1.2 OOD ladder rungs as they pass their tests; P2.1/P2.2 trainings; P2.4 evals. ETA for everything: not estimated yet.

## Headline numbers (sim only; k/n, 95% Wilson)
- MolmoAct2 sim-only, final checkpoint 023400, 20-trial smoke, seed 1000 (videos 4):
  - L_full 15/20 = 75% (53-89%)
  - L_task 17/20 = 85% (64-95%)
  - Failures: 3 never grasped, 1 arm-table collision, 1 dropped or slipped, 1 self-collision.
- Same 20 seeds, checkpoint 001400: L_full 10/20. Final vs 001400: 6 episodes improved, 1 worsened, exact McNemar p = 0.125. Not significant at n=20.
- HEADLINE (in-range, seed 3000, 200 held-out scenes, never used in training):
  - L_full 137/200 = 68.5% (Wilson 95%: 61.8-74.5%)
  - L_task 158/200 = 79.0% (72.8-84.1%)
  - Gate gates/gate_p1_eval200.md, PASS (a result, no pass mark). Finished 21:29 UTC, EXIT=0.
- Speed: about 210 episodes/h with one replica on a free GPU.
- Not yet measured: the OOD ladder, SmolVLA, ACT, the laptop caps, any P3 task.

## What finished / failed / skipped (and why)
- Finished:
  - MolmoAct2 sim-only training: 23,400 steps, EXIT=0, 8 h 52 min.
  - Eval harness gate PASS: expert through the harness 58/60; dataset replay 5/5.
  - Final checkpoint smoke (20 trials).
  - Final model on Drive: 7 files, `rclone check --size-only`: 0 differences.
  - P2.5 wide dataset (camera and exposure ranges widened): 1000 accepted of 1055 attempts (94.8%). Gate PASS. Both dataset tars on Drive (wide and narrow).
  - P3-A-lite expert gate PASS: 169/200 accepted (84.5%, 78.8-88.9%); distractors moved 0.000 mm. Dataset: 600 episodes (200 per red, green, blue), 145,629 frames, 625 of 726 accepted, validated. Demos come from the SCRIPTED EXPERT, not a human video. Dataset tar on Drive (copy in progress at 23:10 UTC); sheets to eval/p3a_sheets.
- Failed or inconclusive:
  - JEPA judge gate FAIL: probe AUC 0.784 vs 0.802 last-frame DINOv2 baseline. Outputs backed up to HF dataset VietHwang/so101-jepa-judge on 11:59 UTC, before the HF stop (per work-22's ledger).
  - G0 overlay gate INCONCLUSIVE: held-out IoU 0.49 vs 0.8 pass mark.
  - PaliGemma tokenizer gated (403): the pi0.5 smoke is BLOCKED until the owner's access request is approved.
  - Multi-replica policy inference hangs on this box: K=8 and K=4 pools, and 4 independent processes, all hung at 0-7 episodes. Single process reliable.
- Skipped:
  - B+ filtered-BC round: dropped by the owner at 20:45 UTC. Code and logs kept.
  - Drawer (stage B), C, D, Tower of Hanoi, dexterous tasks: next steps only (brief section 6).
  - P2.1-P2.2 small-model training: SmolVLA (20k steps) and ACT (30k steps) trained and gated (gate_p2_train). ACT-wide (P2.5) trained too.
  - P2.3 memory check (gate_p2_memcap, PASS). Wording: "fits under an emulated 5.0 GiB PyTorch allocator cap (proxy for a 6 GB laptop card; context and driver overhead on CUDA not measured)", and the same for 7.0 GiB (proxy for 8 GB). The cap changes memory only; capped and uncapped loss curves match.
    - Training, 1000 steps, none OOM, PyTorch reserved GB: SmolVLA bs8 @7.0 3.80; SmolVLA bs4 @5.0 2.65; ACT bs8 @7.0 1.71; ACT bs4 @5.0 1.21.
    - Largest batch that fits: SmolVLA bs16 @7.0 (6.11 GB) and bs8 @5.0 (3.80 GB); ACT bs32 @5.0 (4.13 GB). The next size up OOMs in each case.
    - Inference, 10 episodes, dev seed 9000 (success not reported): ACT 0.32 GB reserved, about 11 ms per chunk; SmolVLA 0.99 GB, about 170-200 ms per chunk. Timings are MI300X, not laptop.
  - P3-A-lite training (SmolVLA-multi, ACT-multi-no-language): armed, not yet started. It starts only when the queue is empty and the GPU is free, and is dropped if not started by 07:00 UTC Sat.

## Deviations from the plan
- HF: no uploads since 12:16 UTC. HF holds only checkpoint 002800 of this run. The HF storage limit blocked the rest. The checkpoint pusher was stopped at 19:45 UTC on the owner's "do not touch HF". The CLAUDE.md 30-minute push rule is therefore not being followed. The brief's stricter rule (no HF at all) applies.
- Checkpoint 023400 is kept only on local disk (outputs_large/final_models/...) and on Drive.
- The 20:51-20:54 Drive copy of the final model: nobody on this box started it. Its source and destination are not in its log. Probably the laptop session.
- 11:37-11:44 GPU stall (about 9 s/step for 7 min): two policy replicas plus training plus V-JEPA. Rule since then: one replica, under 170 GB.
- 21:35-21:40 UTC, box finding: with two trainings (ACT and SmolVLA) on the GPU, the queued eval (01_molmo_s4000_none) hung at 0 episodes (0% GPU occupancy, about 160% CPU spin). Both trainings slowed sharply (ACT 12 to 1.7 step/s, SmolVLA 2 to 0.3 step/s). Work-22 killed the eval at 21:39 and both trainings recovered at once. Rule from now on: at most 2 GPU processes at any time. The brief's "2 trainings plus the queue" plan cannot run on this box. Every queue job now starts with `gpu_slot.sh 1`, which waits until at most one other GPU process is running. Job 01 is re-queued and waits for ACT training to end (about 22:14 UTC). The killed attempt is logged as EXIT=143 in logs/eval_molmo_s4000_none.log; the rerun appends to the same log.

## Where things are (Drive paths, or 'local only: <path>')
- Final model, Drive: gdrive:so101-sprint/models/molmoact2_sim_v1_023400/pretrained_model (verified).
- Final model, local: outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model
- 20-trial smoke, local only: outputs_large/eval/policy_final_n20 (4 videos in videos/, all passes).
- 200-trial eval, local only: outputs_large/eval/policy_final_n200_seed3000
- Gates, local only: /root/work/gates/
- Ledger, local only: /root/work/LEDGER.md
- REPORT itself: gdrive:so101-sprint/REPORT.md (uploaded 21:20, verified by size).
- Text copy: branch report/overnight, commit d41080a, pushed to origin (REPORT, gates, ledger, eval summaries; no binaries).
- Infrastructure now running: uploader loop (tmux "uploader", copy-only, every 30 min, log logs/uploader.log); policy queue loop (tmux "policyq", empty queue); status.sh extended at 21:23 (uploader stale, silent train or eval, GPU idle before 12:30, queue dead, droplet over 29 h).

- Brief-allowed cut ("if short of time"): the small-model seed-3000 evals (ACT, SmolVLA, ACT-wide) are cut from ids 0-199 to ids 0-99. Purpose: P2.4 and P2.5 evals finish about 06:10 UTC instead of 07:20, so P3-A training can still start before the 07:00 limit. MolmoAct2 keeps n=200 on seed 3000. The scoreboard joins on common ids.
- P3-A launch rule: 2 GPU processes at most. The launcher starts SmolVLA-multi (15k steps, bs64) and ACT-multi-no-language (30k steps, bs32) together once the queue is empty and the GPU is free. If not started by 07:00 UTC, it logs "P3-A training DROPPED" and exits.
- Honesty line for P3-A: its demos come from the SCRIPTED EXPERT, not a human video.

## Left for the owner
- Hugging Face: storage decision (paid plan; default no) and whether to resume pushes.
- Confirm who started the 20:51 Drive copy and that its destination is intended.
- PaliGemma access request (blocks the pi0.5 smoke).
- Rotate the rocm container's Jupyter token (visible in the process list).
- Laptop replication: not started.
