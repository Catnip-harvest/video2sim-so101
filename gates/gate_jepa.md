FAIL: JEPA lost. The pre-registered probe (mean-pooled V-JEPA 2, front camera, 16 frames) got test AUC 0.784 [0.720, 0.849]; the best trained last-frame baseline (DINOv2 last frame) got 0.802 [0.737, 0.868]. The difference is -0.018 (paired DeLong p = 0.59), and the gate needed at least +0.02. Two variants that were not gated did beat their last-frame baselines; see section 3.

# Gate: JEPA judge (FINAL_DESIGN §4), 9 Oct 2026

Branch `jepa/judge` (worktree /root/work/wt-jepa). Code is in `jepa/`. Data and features are in `/work/outputs_large/jepa/`. Logs are `/root/work/logs/jepa_{gen,extract,realism,probe}.log`.

## 1. Judge set (sim episodes, labels from the physics verifier only)

All runs use base seed 1, so attempt *i* is the **same scene** (layout, colours, lights, camera) in every condition. Only the physics or the expert's commands differ.

| source | how | positives (verifier passed) | negatives |
|---|---|---:|---:|
| base `sugar_v10_n1000` | normal run (foh servos) | 193 (attempts < 200) | 58 natural rejects (55 arm-table collision, 3 unstable) |
| zoh | `--servo-hold zoh` | 113 | 58 (51 dropped/slipped, 7 arm-table) |
| widen | `--widen`: mass ×0.61–4, friction 0.15–1.0, finger friction 0.15–1.4, jaw kp 150–2000, arm torque ×0.6–1 | 203 | 8 (5 arm-table, 2 grasp failed, 1 dropped) |
| no_close | `--inject no_close`: the expert never closes the jaw | 0 | 71 (grasp failed) |
| early_release | `--inject early_release`: the expert opens the jaw at 20–80 % of the carry | 0 | 71 (dropped; 42 of 71 still landed in the bin) |
| **total** | 775 episodes, 260 scenes | **509** | **266** |

- **Injections change only the expert's jaw commands.** There are no welds and no teleports. Every label is `accepted` from the verifier in `rollout.py`. The injected runs simply had no episodes that passed.
- **Positives from the zoh and widen runs are included.** So "which run did it come from" is not a shortcut. Checked on the test splits: mean probe score for base, zoh and widen positives is 0.60, 0.63 and 0.61.

**Splits are by scene, so all episodes of one scene stay together.**

| split | rule | episodes | scenes | y_full pos / neg |
|---|---|---:|---:|---:|
| `cam_test` | camera raised by cam_dz ≥ 4.8 cm (top ~20 % of the ±8 cm band) | 146 | 51 | 96 / 50 |
| `test` | 40 % of the other scenes (random, seed 0) | 235 | 84 | 164 / 71 |
| `train` | the rest | 394 | 125 | 249 / 145 |

**Two labels:**
- `y_full` is the verifier's verdict. This is the gate label.
- `y_task` asks whether the object ended released and resting in the bin. A cube that was dropped into the bin, or a transient arm-table touch, still counts as y_task = 1.

## 2. Method

**Encoder:** frozen `facebook/vjepa2-vitl-fpc64-256`, run in bf16 through transformers `get_vision_features`. Preprocessing is the model's own: short edge resized to 292, centre crop 256, ImageNet normalisation. My version matches `AutoVideoProcessor` to within 0.018.

**Clip:** 16 frames spread uniformly from the first to the last frame of the episode. Sim videos are 10 Hz, 14–26 s, so the frames are about 1–1.7 s apart. The encoder outputs 8 time slots × 256 patches × 1024 numbers. We keep the spatial mean per time slot, (8, 1024), and the probe uses the mean over time slots (1024 numbers).

**Contenders:** all use the same recipe: standardise → L2 logistic regression with `class_weight=balanced`. C is picked by 5-fold cross-validation on train, grouped by scene; the model is then refit on train and scored once on each test split.
- (a) **probe:** the V-JEPA video features above. This is the pre-registered contender.
- (b) **last-frame baselines,** trained the same way:
  - V-JEPA on the last frame repeated 16 times.
  - DINOv2-small on the last frame (CLS + mean patch token, 768 numbers).
  - The gate compares against the **better** of the two.
- (c) **shuffled-label control:** the probe recipe trained on permuted train labels, repeated 20 times.

**Gate (fixed beforehand):** on `test`, y_full, AUC(probe) − AUC(best last frame) ≥ 0.02 with paired DeLong p < 0.05. As a robustness check I also run a scene-cluster bootstrap with 2000 resamples, because episodes that share a scene are not independent.

## 3. Results: success probe (AUC with 95 % DeLong CI)

### y_full (the gate label)

| contender | test (164 / 71) | cam_test (96 / 50) |
|---|---|---|
| **probe (gated)** | **0.784 [0.720, 0.849]** | 0.829 [0.756, 0.901] |
| last frame, V-JEPA | 0.799 [0.736, 0.861] | 0.752 [0.666, 0.838] |
| last frame, DINOv2 | 0.802 [0.737, 0.868] | 0.822 [0.752, 0.893] |
| shuffled labels (20×) | mean 0.512 (0.391–0.616) | mean 0.496 (0.350–0.620) |
| *secondary:* probe as an MLP (256 hidden units) | 0.777 [0.712, 0.842] | 0.781 [0.696, 0.866] |
| *secondary:* 8 time slots concatenated, 8192 numbers | 0.866 [0.820, 0.913] | 0.938 [0.896, 0.980] |
| *secondary:* wrist-camera video (V-JEPA) | 0.912 [0.866, 0.958] | 0.968 [0.927, 1.000] |
| *secondary:* wrist last frame (DINOv2) | 0.798 [0.732, 0.865] | 0.855 [0.790, 0.920] |
| *secondary:* front + wrist video | 0.900 [0.855, 0.945] | 0.974 [0.934, 1.000] |

**Paired comparisons:**

| comparison | test | cam_test |
|---|---|---|
| **GATE: probe − last DINOv2** | **−0.018, CI [−0.084, +0.048], p = 0.59** | +0.006, CI [−0.082, +0.094], p = 0.89 |
| scene bootstrap, probe − last DINOv2 | CI [−0.085, +0.054]; P(diff < 0.02) = 0.85 | CI [−0.088, +0.094] |
| probe − last V-JEPA | −0.014, p = 0.65 | +0.077, p = 0.057 |
| *secondary:* time-concat − last DINOv2 | **+0.064, CI [+0.002, +0.125], p = 0.042** | **+0.116, CI [+0.035, +0.196], p = 0.005** |
| *secondary:* wrist video − wrist last frame | **+0.113, CI [+0.033, +0.193], p = 0.005** | **+0.113, CI [+0.034, +0.192], p = 0.005** |

**Per failure type, y_full, `test` split.** Each cell is the AUC of that type's negatives against all test positives, then the share of that type caught at p < 0.5.

| type (n) | probe | last V-JEPA | last DINOv2 | time-concat | wrist video | wrist last |
|---|---|---|---|---|---|---|
| natural, mostly arm-table touch (17) | 0.69 / 0.65 | 0.71 / 0.59 | 0.69 / 0.65 | 0.85 / 0.88 | 0.90 / 0.82 | 0.83 / 0.65 |
| zoh, mostly drops (18) | 0.75 / 0.67 | 0.72 / 0.56 | 0.74 / 0.56 | 0.85 / 0.89 | 0.87 / 0.89 | 0.71 / 0.61 |
| early_release (16) | 0.72 / 0.62 | 0.80 / 0.75 | 0.79 / 0.69 | 0.79 / 0.56 | 0.95 / 1.00 | 0.63 / 0.38 |
| no_close (16) | 0.97 / 1.00 | 0.98 / 1.00 | 0.99 / 1.00 | 0.98 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 |
| widen (4) | 0.86 / 0.75 | 0.77 / 0.50 | 0.88 / 0.75 | 0.88 / 1.00 | 0.68 / 0.50 | 0.95 / 1.00 |

The `cam_test` per-type table is in `probe_results.json`. It shows the same pattern.

### y_task (secondary: does the object end in the bin?)

There are only 25 test negatives, so the CIs are wide.

| contender | test | cam_test |
|---|---|---|
| probe | 0.929 [0.857, 1.000] | 0.951 |
| last frame, V-JEPA | 0.876 | 0.916 |
| last frame, DINOv2 | 0.882 | 0.954 |
| wrist last frame | 0.951 | 0.932 |

The probe minus the best last-frame baseline is +0.046 (p = 0.19) on test and −0.002 (p = 0.96) on cam_test. That is not significant.

### What this means in plain words

- **The gated probe loses.** Averaging V-JEPA's features over the clip and the image does no better than one picture of the last frame. So "JEPA lost" as specified.
- **The useful signal is in *when* things happen and in the wrist view.**
  - Keeping the 8 time slots in order (time-concat) beats the last frame by +0.06 to +0.12 AUC.
  - The wrist video beats the wrist last frame by +0.11. It catches 100 % of early releases and 89 % of zoh drops, against 38 % and 61 % from the wrist last frame.
  - These failures are visible during the episode but often not at its end: 42 of 71 early-release cubes still landed in the bin.
- **Treat those two as hypotheses, not results.** I picked them after seeing several variants, so they need a fresh scene split, or a re-run with them pre-registered, before anyone relies on them.
- **No contender cracks the "natural" arm-table touches.** These are the hardest negatives: the best is time-concat at 0.85.
- **The shuffled-label control sits at about 0.5.** So the pipeline is not leaking labels.

## 4. Results: realism score (Fréchet distance in V-JEPA space)

**Setup:**
- Sim side: 400 accepted base episodes on the front camera.
- Real side: 400 random episodes from MinhDuk/final_dataset_annotated `cam_head`, decoded with PyAV over each episode's own time span, with 16 uniform frames.
- Every FD compares 200 clips against 200, over 20 random half-splits. The table gives the mean and the min–max.

| | full 1024-d | PCA-64 |
|---|---:|---:|
| real vs real, floor | 15.0 (11.4–18.6) | 9.6 (5.1–18.7) |
| sim vs sim, floor | 13.2 (11.9–14.7) | 7.5 (6.1–11.1) |
| **sim vs real** | **1328.8 (1328.6–1329.3)** | **1317.3** |
| sim−real ÷ real floor | 89× | 137× |
| sim−real by camera-height tercile (low / mid / high, n ≈ 133) | 1332 / 1335 / 1341 (real floor at n = 133: 22) | 1317 / 1321 / 1326 |

A logistic sim-vs-real classifier separates the two perfectly: 5-fold AUC 1.000.

**In plain words:**
- To V-JEPA, our renders and the friend's clips are completely different worlds. The gap is about 90 times larger than the gap between two random halves of the real data.
- Moving our camera up or down within ±8 cm changes nothing (1332 vs 1341). The mismatch is not camera height. It is:
  - Viewpoint: the real cam_head is a close top-down view; the sim front camera is a far three-quarter view.
  - Scene: the real scene has a big green cube, a white bowl and a wood table; the sim has a ~2 cm cube in a red tray on random colours.
  - The flat "fast" renders.
- So this score cannot yet "pick camera pose and randomisation ranges". First render sim from a cam_head-like pose (the calib G0 per-episode camera fit), then use the score to compare variants. The floors are the yardstick for that.

## 5. Compute

- **GPU:**
  - V-JEPA extraction took one continuous run: about 13 min for the sim (775 clips × 3 variants, plus 400 realism clips) and about 1 min to encode the real clips.
  - Peak memory was about 1.2 GB (bf16, batch 8).
  - DINOv2 on the front last frame ran on the GPU in the same batch. DINOv2 on the wrist last frame ran on the CPU afterwards.
  - The probes and FD ran on the CPU only.
  - Training slowed while the GPU was shared (the coordinator reported 1.20 → 1.79 s/step). I then stopped using the GPU.
- **CPU:**
  - Negative generation took about 38 min with 6 workers at nice 19, front + wrist cameras, fast render. Training data_s stayed between 0.11 and 0.29 s.
  - Real decoding took 139 s with 4 workers.

## 6. Limitations

1. The cube is about 6 px in a 256 px front render, smaller than one 16 px V-JEPA patch. The front camera barely sees the outcome, which handicaps every front-camera contender equally.
2. Widened randomisation produced only 8 failures in 211 attempts. That type is underpowered (4 test negatives).
3. The test split has 71 negatives, so the CI half-width is about ±0.065, not the ±0.022 the design expected at AUC 0.95 with 200/200.
4. DeLong assumes independent episodes. The scene bootstrap gives similar, slightly wider CIs.
5. Only a logistic probe and a small MLP were tried on pooled features. The attentive probe from 2510.16240 was not tried, and neither was V-JEPA 2.1 at 384 px.
6. Real clips cover 156 task phrasings of green-cube-to-bowl tasks, not our exact task.
7. Nothing was pushed to Hugging Face. Features can be regenerated from the staged videos with `jepa/extract.py`.
8. Robometer-4B was not attempted.

## 7. Files and how to reproduce

- `jepa/gen_negatives.sh` runs the 4 generation runs.
- `datagen/generate.py` gained `--widen` and `--inject {no_close,early_release}`. The supporting changes are in `datagen/r2s/{variations,planner,rollout}.py`.
- `jepa/build_judge_set.py` writes `judge_set.json`.
- `jepa/vjepa.py` has the encoder and preprocessing.
- `jepa/extract.py sim|real|wrist_last` writes `feats_*.npz`.
- `jepa/stats.py` has DeLong and the scene bootstrap.
- `jepa/probe.py` writes `probe_results.json` and `probe_scores_*.npz`.
- `jepa/realism.py` writes `realism_results.json`.
- Generated episodes are in `/work/outputs_large/jepa/gen/{no_close,early_release,zoh,widen}/`.
- `datagen/tests/test_datagen.py` passes (7 tests) after the changes.
