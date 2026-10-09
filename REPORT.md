Money: alive 12.4 h = $24.58 now. If destroyed at 14:00 UTC Sat $57.81 | 16:00 $61.79 | 20:00 $69.75 (cap $60)
# SO-101 overnight report - updated 21:20 UTC (Sat 04:20 Hanoi)
Status: RUNNING
Now running: P1.1 200-trial eval of the final checkpoint, seed 3000 (work-22 runs it; 165/200 done at 21:20, no final numbers yet).
Next: per the brief, P1.2 OOD ladder code, then P2 setup. Not yet verified as started. ETA for everything: not estimated yet.

## Headline numbers (sim only; k/n, 95% Wilson)
- MolmoAct2 sim-only, final checkpoint 023400, 20-trial smoke, seed 1000 (videos 4):
  - L_full 15/20 = 75% (53-89%)
  - L_task 17/20 = 85% (64-95%)
  - Failures: 3 never grasped, 1 arm-table collision, 1 dropped or slipped, 1 self-collision.
- Same 20 seeds, checkpoint 001400: L_full 10/20. Final vs 001400: 6 episodes improved, 1 worsened, exact McNemar p = 0.125. Not significant at n=20.
- Seed 3000, 200 trials: in progress, 165/200 at 21:20. Numbers come when it finishes.
- Speed: about 210 episodes/h with one replica on a free GPU.
- Not yet measured: the OOD ladder, SmolVLA, ACT, the laptop caps, any P3 task.

## What finished / failed / skipped (and why)
- Finished:
  - MolmoAct2 sim-only training: 23,400 steps, EXIT=0, 8 h 52 min.
  - Eval harness gate PASS: expert through the harness 58/60; dataset replay 5/5.
  - Final checkpoint smoke (20 trials).
  - Final model on Drive: 7 files, `rclone check --size-only`: 0 differences.
- Failed or inconclusive:
  - JEPA judge gate FAIL: probe AUC 0.784 vs 0.802 last-frame DINOv2 baseline. Outputs backed up to HF dataset VietHwang/so101-jepa-judge on 11:59 UTC, before the HF stop (per work-22's ledger).
  - G0 overlay gate INCONCLUSIVE: held-out IoU 0.49 vs 0.8 pass mark.
  - PaliGemma tokenizer gated (403): the pi0.5 smoke is BLOCKED until the owner's access request is approved.
  - Multi-replica policy inference hangs on this box: K=8 and K=4 pools, and 4 independent processes, all hung at 0-7 episodes. Single process reliable.
- Skipped:
  - B+ filtered-BC round: dropped by the owner at 20:45 UTC. Code and logs kept.
  - Drawer (stage B), C, D, Tower of Hanoi, dexterous tasks: next steps only (brief section 6).
  - P2 (SmolVLA, ACT) and P3-A: not started as of this report.

## Deviations from the plan
- HF: no uploads since 12:16 UTC. HF holds only checkpoint 002800 of this run. The HF storage limit blocked the rest. The checkpoint pusher was stopped at 19:45 UTC on the owner's "do not touch HF". The CLAUDE.md 30-minute push rule is therefore not being followed. The brief's stricter rule (no HF at all) applies.
- Checkpoint 023400 is kept only on local disk (outputs_large/final_models/...) and on Drive.
- The 20:51-20:54 Drive copy of the final model: nobody on this box started it. Its source and destination are not in its log. Probably the laptop session.
- 11:37-11:44 GPU stall (about 9 s/step for 7 min): two policy replicas plus training plus V-JEPA. Rule since then: one replica, under 170 GB.

## Where things are (Drive paths, or 'local only: <path>')
- Final model, Drive: gdrive:so101-sprint/models/molmoact2_sim_v1_023400/pretrained_model (verified).
- Final model, local: outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model
- 20-trial smoke, local only: outputs_large/eval/policy_final_n20 (4 videos in videos/, all passes).
- 200-trial eval, local only: outputs_large/eval/policy_final_n200_seed3000
- Gates, local only: /root/work/gates/
- Ledger, local only: /root/work/LEDGER.md
- REPORT itself: gdrive:so101-sprint/REPORT.md (uploaded with this version).

## Left for the owner
- Hugging Face: storage decision (paid plan; default no) and whether to resume pushes.
- Confirm who started the 20:51 Drive copy and that its destination is intended.
- PaliGemma access request (blocks the pi0.5 smoke).
- Rotate the rocm container's Jupyter token (visible in the process list).
- Laptop replication: not started.
