Money: alive 21.5 h = $42.72 at 06:24 UTC. If destroyed at 14:00 UTC Sat $57.81 | 16:00 $61.79 | 20:00 $69.75 (cap $60)
# SO-101 overnight report - final, updated 06:24 UTC (Sat 13:24 Hanoi)
Status: DONE - DESTROY THE DROPLET NOW (owner: DigitalOcean > Droplets > Destroy)
Now running: nothing. GPU idle since 06:12, no GPU process. Uploader loop stopped 06:09; queue loop stopped at the final step.
Basis: gates/gate_final.md. Every item in outputs_large/FINAL_ARTEFACTS.txt is on Drive. Size-only checks passed on all items. An MD5 spot-check by the main session (06:24 UTC) matched on the 6 models, 3 dataset tars, 10 bundles, gates, SCOREBOARD.md and FINAL_ARTEFACTS.txt.
Demos come from the SCRIPTED EXPERT, not a human video. All results are in simulation.

## Headline numbers (sim only; k/n, 95% Wilson)
- In-range, seed 3000, 200 held-out scenes, MolmoAct2 (final checkpoint 023400): L_full 137/200 = 68.5% (61.8-74.5%); L_task 158/200 = 79.0% (72.8-84.1%). Gate gate_p1_eval200 PASS (a result, no pass mark).
- Small models, seed 3000, ids 0-99 only (see Deviations): SmolVLA L_full 38/100 (29.1-47.8%), L_task 75/100 (65.7-82.5%). ACT L_full 17/100 (10.9-25.5%), L_task 69/100 (59.4-77.2%).
- 20-trial smoke, seed 1000: L_full 15/20 = 75% (53-89%), L_task 17/20 = 85%. Checkpoint 001400 on the same seeds: 10/20 L_full; the final-vs-001400 exact McNemar p = 0.125, not significant at n=20.

## Scoreboard (SCOREBOARD.md, regenerated 03:02 UTC by eval/scoreboard.py; Holm across 204 tests)
Verdict wording is copied word for word from the SCOREBOARD verdict column.
- Significant, MolmoAct2 vs others (in-range seed 3000; ACT and ACT-wide on ids 0-99):
  - vs SmolVLA, L_full: "A better (significant)", Holm 0.032.
  - vs ACT, L_full: "A better (significant)", Holm 2.8e-08.
  - vs ACT-wide, L_full: "A better (significant)", Holm 0.00015.
- Significant, seed 4000:
  - vs ACT, L_full: "A better (significant)" on the none baseline and on camera, colour, combo, distractor, camera_height, camera_pitch, camera_yaw and camera_distance.
  - vs ACT, L_task: "A better (significant)" on colour (Holm 0.016), combo (0.0021) and distractor (9.1e-05).
  - vs ACT-wide, "A better (significant)": distractor L_full (8.3e-05) and L_task (0.0020), camera_height L_full (0.040).
- Significant, SmolVLA vs ACT: "A better (significant)", L_full, on colour and camera_distance (Holm 0.0021 each).
- Significant, ACT-wide vs ACT (P2.5 data engine, same scenes): "B better (significant)", L_full, on colour (2 to 18 of 50, Holm 0.027) and camera_distance (4 to 20 of 50, Holm 0.027).
- Significant drop: ACT-wide, distractor, L_task, 39 to 17 of 50: "significant drop" (Holm 9.3e-05). It is the only row with that verdict.
- Not significant, quoted from the verdict column:
  - MolmoAct2 vs ACT, seed 4000 exposure, L_full: "not distinguishable at n=50" (Holm 0.16).
  - ACT, distractor, L_task, 31 to 12 of 50, vs its own none: "not distinguishable at n=50" (Holm 0.056).
  - MolmoAct2 and SmolVLA rung drops vs their own none runs: "not distinguishable at n=50".
- Expert ceiling on each seed-4000 rung: 17/20 L_full, 19/20 L_task.
- P2.5 data engine (ACT trained on wide camera/exposure data vs the same ACT on narrow data), same scenes. Wide >= narrow on all 22 paired comparisons (raw). Verdicts in the whole-scoreboard table: "B better (significant)" on colour L_full (2 to 18 of 50, Holm p 0.027) and camera_distance L_full (4 to 20 of 50, Holm p 0.027). Other rows: "not distinguishable at n=50".
- P2.5 pre-specified family (owner's 22 tests, Holm within family, from gate_p25_eval): camera L_task 30 to 44, colour L_task 18 to 34, camera_pitch L_full 5 to 19, and also colour and camera_distance L_full. Labelled as the family result, not the whole-scoreboard verdict.
- Caveat: most camera rungs lie inside the wide model's training ranges. Colours were not widened, so the colour gain is the cleaner evidence. Wide data did not fix distractors: ACT-wide still drops 39 to 17 of 50 on L_task, "significant drop" (Holm p 9e-5).

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
  - P3-A ACT-multi-no-language (no language input): trained 30k steps, EXIT 0, final loss 0.084, final model on Drive (7 files, size-verified).
  - P3-A-lite SmolVLA-multi trained 15k steps (loss 0.029); ACT-multi-no-language trained 30k steps (loss 0.084). Both started together at 03:03 UTC. Final models on Drive.
  - P3-A eval, FAIL (negative result, gates/gate_p3A_eval.md). Neither model learned the three-cube task at 600 demos. Named cube moved in 13-15% of held-out scenes, and no cube moved in about 75% of runs. The named-cube rates are not distinguishable between models at n=60 ("not distinguishable at n=60"). "Language beats no-language" is NOT shown.
    - Held-out ids 0-59 (training templates): SmolVLA-multi named 9/60 (15%, 8.1-26.1%); ACT-multi 8/60 (13%, 6.9-24.2%). Named in tray: 5/60 vs 5/60.
    - Held-out paraphrases ids 0-29: SmolVLA-multi named 2/30 (7%, 1.8-21.3%). ACT-multi has no paraphrase run.
    - Swap ids 0-19 x 3 colours: SmolVLA-multi named 5/60; ACT-multi named 4/60.
    - Yellow target ids 100-119 (never a training target): SmolVLA-multi 1/20; ACT-multi 3/20.
    - Honesty: demos come from the SCRIPTED EXPERT, not a human video. These are simulation results.

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
- replication/LAPTOP.md: written (draft, all figures from the memcap gate). Drive: replication/LAPTOP.md. Nothing in it has been run on an NVIDIA laptop card.
- Final artefacts: gates/gate_final.md (size-only checks on every item in outputs_large/FINAL_ARTEFACTS.txt, all PASS). Size-only, not content, so a content spot-check is recommended before relying on any single file.

## Left for the owner
- Hugging Face: storage decision (paid plan; default no) and whether to resume pushes. Without a decision, the final checkpoint is on Drive and local disk only.
- Confirm who started the 20:51 Drive copy and that its destination is intended.
- PaliGemma access request (blocks the pi0.5 smoke).
- Rotate the rocm container's Jupyter token (visible in the process list).
