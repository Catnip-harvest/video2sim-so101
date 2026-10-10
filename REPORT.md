Money: alive 17.6 h = $34.97 now (02:31 UTC). If destroyed at 14:00 UTC Sat $57.81 | 16:00 $61.79 | 20:00 $69.75 (cap $60)
# SO-101 overnight report - updated 02:31 UTC (Sat 09:31 Hanoi)
Status: RUNNING
Now running (work-22): queue jobs 400-410 (ACT-wide evals, P2.5). Then p3_train.sh starts the P3-A trainings (SmolVLA-multi, ACT-multi-no-language), about 03:20 UTC, if the queue and GPU are free.
Next: P3-A evals; P4 packaging (scoreboard, videos, LAPTOP.md, README). Hard deadlines: 12:30 UTC no new GPU job; 13:30 DONE target.

## Headline numbers (sim only; k/n, 95% Wilson)
- In-range, seed 3000, 200 held-out scenes, MolmoAct2 (final checkpoint 023400): L_full 137/200 = 68.5% (61.8-74.5%); L_task 158/200 = 79.0% (72.8-84.1%). Gate gate_p1_eval200 PASS (a result, no pass mark).
- Small models, seed 3000, ids 0-99 only (see Deviations): SmolVLA L_full 38/100 (29.1-47.8%), L_task 75/100 (65.7-82.5%). ACT L_full 17/100 (10.9-25.5%), L_task 69/100 (59.4-77.2%).
- 20-trial smoke, seed 1000: L_full 15/20 = 75% (53-89%), L_task 17/20 = 85%. Checkpoint 001400 on the same seeds: 10/20 L_full; the final-vs-001400 exact McNemar p = 0.125, not significant at n=20.

## Scoreboard (SCOREBOARD.md, regenerated 02:30 UTC by eval/scoreboard.py, manifest a014c8d; Holm across 100 tests)
- MolmoAct2 > SmolVLA on in-range L_full: 66 vs 38, Holm p 0.018. L_task not distinguishable.
- MolmoAct2 > ACT on in-range L_full: 66 vs 17, Holm p 1.6e-8. L_task not distinguishable (p 1.0).
- Seed 4000: MolmoAct2 > ACT on L_full on 8 of 9 rungs (camera, colour, combo, distractor, camera_height, camera_pitch, camera_yaw, camera_distance). Exposure is not significant (Holm p 0.093). The seed-4000 baseline (none) is also significant.
- Seed 4000 L_task: MolmoAct2 > ACT significantly only on colour (Holm p 0.009), combo (0.001) and distractor (5e-5).
- SmolVLA > ACT on L_full on colour and camera_distance (Holm p 0.001 each).
- ACT's L_task drop on the distractor rung against its own none run is significant (Holm p 0.032). No MolmoAct2 drop against its own none run is significant.
- All other rows: not distinguishable at n=50 or n=100.
- Expert ceiling on each seed-4000 rung: 17/20 L_full, 19/20 L_task.

## What finished / failed / skipped (and why)
- Finished:
  - MolmoAct2 training: 23,400 steps, EXIT 0, 8 h 52 min. Eval harness gate PASS (expert 58/60, replay 5/5).
  - P1.1 200-trial eval PASS. P1.2 OOD ladder code PASS (gate_p1_ood_code), 9 rungs.
  - P2.1-P2.2: SmolVLA (20k steps) and ACT (30k steps) trained, gate_p2_train PASS. ACT-wide (P2.5) trained, loss 0.083.
  - P2.3 memory caps, gate_p2_memcap PASS. Wording: "fits under an emulated 5.0 GiB PyTorch allocator cap (proxy for a 6 GB laptop card; context and driver overhead on CUDA not measured)"; same for 7.0 GiB (proxy for 8 GB). Largest batch that fits: SmolVLA bs16 @7.0 (6.11 GB reserved), bs8 @5.0 (3.80 GB); ACT bs32 @5.0 (4.13 GB). Next size up OOMs. Inference ACT 0.32 GB, SmolVLA 0.99 GB. Timings are MI300X, not laptop.
  - P2.4 small-model evals complete, gate_p2_eval PASS (seed 3000 cut to ids 0-99; see Deviations).
  - P2.5 wide dataset: 1000 of 1055 accepted (94.8%), gate_p25_data PASS.
  - P3-A-lite expert gate PASS: 169/200 accepted (84.5%, 78.8-88.9%); distractors moved 0.000 mm. Dataset 600 episodes (200 per red, green, blue), 145,629 frames, 625 of 726 accepted. Demos come from the SCRIPTED EXPERT, not a human video.
- Failed or inconclusive:
  - JEPA judge FAIL: probe AUC 0.784 vs 0.802 DINOv2 last-frame baseline. Outputs in HF dataset VietHwang/so101-jepa-judge, pushed 11:59 UTC before the HF stop.
  - G0 overlay INCONCLUSIVE: held-out IoU 0.49 vs 0.8 pass mark.
  - PaliGemma tokenizer gated (403): the pi0.5 smoke is BLOCKED until the owner's access request is approved.
  - Multi-replica policy inference hung (K=8, K=4 pools, and 4 independent processes). Single process is reliable.
- Skipped:
  - B+ filtered-BC round: dropped by the owner at 20:45 UTC; code and logs kept.
  - Drawer (stage B), C, D, Tower of Hanoi, dexterous tasks: next steps only (brief section 6).
  - P3-A training (SmolVLA-multi, ACT-multi-no-language): not started yet. Dropped if not started by 07:00 UTC Sat.

## Deviations from the plan
- Brief-allowed cut ("if short of time"): the small-model seed-3000 evals (ACT, SmolVLA, ACT-wide) cut from ids 0-199 to 0-99, so P2.4/P2.5 finish about 06:10 UTC instead of 07:20 and P3-A can start before 07:00. MolmoAct2 keeps n=200. The scoreboard joins on common ids.
- P3-A launch rule: at most 2 GPU processes. The launcher starts SmolVLA-multi (15k steps, bs64) and ACT-multi-no-language (30k, bs32) together once the queue is empty and the GPU is free.
- HF: no uploads since 12:16 UTC; HF holds only checkpoint 002800 of this run. The pusher was stopped at 19:45 UTC on the owner's "do not touch HF". The CLAUDE.md 30-minute push rule is therefore not followed; the brief's stricter rule applies.
- The 20:51-20:54 Drive copy of the final model: not started by any agent on this box; its source and destination are not in its log.
- GPU stall 11:37-11:44 (about 9 s/step for 7 min): two policy replicas, training and V-JEPA. Rule since: one replica, under 170 GB.
- 21:35-21:40 box finding: two trainings plus the queued eval hung at 0 episodes. The eval was killed at 21:39. Rule now: at most 2 GPU processes, and every queue job starts with gpu_slot.sh 1.
- Uploader eval copy timed out (exit 124) at 22:49 and 23:54, because it copied the dropped B+ tree (test_bplus, 6,458 files, 1.2 GB). Fixed 00:00 UTC: test_bplus and rollouts excluded, 15-minute cap per upload. Exit 1 at 01:21 and 02:02 was a file still being written by a running eval, which is benign.

## Where things are (Drive paths, or 'local only: <path>')
- Final MolmoAct2: gdrive:so101-sprint/models/molmoact2_sim_v1_023400/pretrained_model (7 files, size-verified). Local: outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model.
- Small-model finals: models/smolvla_single_bs64_020000, models/act_single_bs32_030000, models/act_wide_bs32_030000 (all size-verified).
- Datasets: datasets/sugar_v10_n1000_wide_small.tar.zst, datasets/sugar_v10_n1000_small.tar.zst, datasets/multitask_A_v1_small.tar.zst (size-verified).
- Eval results: eval/ on Drive (uploader; rollouts and test_bplus excluded). Local: outputs_large/eval/. Sheets: eval/ood_sheets, eval/p3a_sheets. P2.3 memory JSON: p23/ (14 files, verified).
- Gates, ledger and SCOREBOARD.md: copied to Drive by the uploader. Local: /root/work/.
- Text copy of this report: branch report/overnight, commit d1efe86, pushed to origin.
- Not yet written: replication/LAPTOP.md (packaging step).

## Left for the owner
- Hugging Face: storage decision (paid plan; default no) and whether to resume pushes. Without a decision, the final checkpoint is on Drive and local disk only.
- Confirm who started the 20:51 Drive copy and that its destination is intended.
- PaliGemma access request (blocks the pi0.5 smoke).
- Rotate the rocm container's Jupyter token (visible in the process list).
