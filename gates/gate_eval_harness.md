PASS: expert through harness 58/60 L_full (96.7 %, 60/60 L_task); dataset replay 5/5; phase-free vs expert verifier agree 59/60 expert + 20/20 zoh + 10/10 no-close; policy smoke ckpt 001400 = 10/20 L_full (50 %, CI 30-70 %), 15/20 L_task (75 %, CI 53-89 %); eval ~116 episodes/h per policy replica (expert mode ~2,200/h on 3 CPU workers).

# Gate: closed-loop sim eval harness + phase-free verifier (9 Oct, branch eval/harness)

Code (worktree /root/work/wt-eval, branch `eval/harness`):
- `datagen/r2s/verify_policy.py`: the phase-free verifier. `PhaseFreeMonitor` plus `run_expert_judged`, which attaches the monitor to the unchanged `rollout.run_episode`.
- `eval/sim_eval.py`: the harness CLI, with modes `policy`, `expert` and `dataset-replay`.
- `eval/policy_io.py`: loads the lerobot checkpoint and runs `select_action` through the pre/post processors.
- Tests: `datagen/tests/test_verify_policy.py` (6 tests) and `eval/tests/test_sim_eval.py` (5 tests). All 11 pass in 61 s.
- `datagen/tests/verify_agreement.py`: the 100-episode agreement study.

Outputs are in `/root/work/outputs_large/eval/`. Logs are `/root/work/logs/eval_*.log`.
Eval seeds are `--seed 1000` with variety on. Training used `--seed 1`, so no episode seed overlaps.

## 1. Phase-free verifier (FINAL_DESIGN §3)

**How it decides, with no phase labels:**
- **Carry:** both jaws touch the cube, the cube is more than 5 mm above its settled height, and it touches neither the table nor the bin floor. The carry lasts while both jaws touch it.
- **Release:** the carry ends while the jaw servo goal is at least 0.1 rad above its lowest value during that carry (the gripper is opening). Any other end of a carry counts as a drop or slip.
- **Torque check:** only the 0.25 s saturation-and-sag window. There is no "target still" test.

**Clauses:**
- **L_task** (task done): the cube was carried with both jaws, a carry ended over the bin, the cube ends in the bin region resting, and no jaw touches it in the last 0.5 s.
- **L_full** adds:
  - Physical-honesty clauses: drop/slip; release height of the cube bottom at most 10 mm above the surface below; cube speed at most 0.5 m/s; no foreign contact while carried; no arm-body contact with the cube; no arm-table, arm-bin or self collision; joint limits; torque saturation; target speed cap; robot contact penetration deeper than 2 mm for 3 or more steps; lift of at least 2 cm.
  - Style clauses, which are also reported separately: cube disturbed more than 10 mm before the grasp; tracking error.

**Agreement study.** Each row judges the same simulation with both verifiers. Wall time was 95 s on 4 workers.

| group (seed 1000, variety) | n | expert verifier pass | phase-free pass | agree |
|---|---:|---:|---:|---:|
| scripted expert, FOH servos | 60 | 58 | 57 | **59/60** |
| zero-order-hold servos (drops) | 20 | 16 | 16 | **20/20** |
| jaw never closes (`close_jaw=False`) | 10 | 0 | 0 | **10/10** |
| release 6 cm above the floor (`release_gap=0.06`) | 10 | 9 | 0 | 1/10 (intended) |

**Disagreements, all explained:**
- **Expert attempt 43:** the expert verifier passes it and the phase-free one fails it on `penetration`. The moving-jaw box sank 2.055 mm into the cube for 6 steps while squeezing. The expert verifier has no penetration clause; the -2 mm threshold comes from the design. It is 1 of 60 episodes.
- **Both verifiers fail attempts 32 and 44** on `arm_table_collision`. These two are agreements, not disagreements.
- **ZOH group:** both verifiers fail the same 4 episodes (10, 11, 14, 19) with `dropped_or_slipped` and `joint_limit`. The phase-free one also flags `release_too_high` and `object_fling` for the drop: the cube fell 18-35 mm at 0.57-0.81 m/s.
- **High-release group:** the expert verifier passes 9 episodes that the phase-free one fails, on `release_too_high` (59 mm) and `object_fling` (about 1.05 m/s). This is the intended new clause. L_task still passes 9/10 because the cube did end in the bin. The 10th episode is an expert planning failure (`ik_unreachable`) that both verifiers fail.

**Measured on the 60 expert episodes, where the thresholds have margin:**

| quantity | measured | threshold |
|---|---|---|
| release height | 4.1-5.0 mm | 10 mm |
| max cube speed | 0.28-0.38 m/s | 0.5 m/s |
| disturbance at first both-jaw contact (the expert's own close pushes the cube about 4 mm) | 2.4-4.6 mm | 10 mm |
| saturated-sag window | 0 s | 0.25 s |
| tracking error | 0.004 rad max | 0.1 rad |

**Change from the design:** penetration counts only contacts involving a robot geom. Passive landings are only reported. The expert's own 4-5 mm release drop sinks the cube 1.5-4.5 mm into the bin floor for up to 10 steps, because the scene uses default soft contacts. Counting that made 36 of 60 expert episodes fail.

**Negative controls (pytest):**
- **(b) Jaw never closes:** fails on `never_carried`.
- **(c) Cube starts resting in the bin and the arm stays still:** fails ONLY on `never_carried` and `not_carried_into_bin`. The in-region, resting and released checks all pass, so the both-jaw clause alone stops this kind of push-in. A literal push into the tray is impossible in this scene: the walls are 25 mm high and the cube is 20-26 mm.
- **(d) ZOH drops:** attempts 10 and 14 fail with `dropped_or_slipped` in both verifiers.
- **6 cm release:** fails with `release_too_high`.

## 2. Harness

**Each 10 Hz step:**
1. **State:** the measured joints are converted sim rad → `sim_to_lerobot` (default ArmCalib) → `lerobot_to_molmoact2`, as float32.
2. **Images:** cam_head (sim `front`) and cam_right (sim `wrist_cam`) are rendered with the datagen path: its own render model, a separate MjData set to the live qpos, quality `fast`, 256×256.
3. **Codec:** each image gets one h264 encode/decode with the dataset settings (crf 23, veryfast).
4. **Tensors:** lerobot `prepare_observation_for_inference` turns the images into float32 [0,1] CHW. This is the same as what LeRobotDataset gives: float32 [3,256,256] in [0,1], measured.
5. **Task string:** "pick up the cube and place it in the bin".
6. **Policy call:** `select_action` through the checkpoint's pre/post processors. Images are only rendered when a new chunk is generated.

**Applying the action:**
1. Convert the action molmoact2 → lerobot → sim rad.
2. Clip it to the joint range.
3. Expand it to 3 frame goals: by default (`--interp lag`) a linear ramp from the previous action.
4. Rate-limit to 98 % of the expert caps (0.5 rad/s arm, 2 rad/s jaw).
5. Ramp each frame's goal over its 7 physics steps (first-order hold, the expert's servo model).

**Episode end:** the episode stops at 40 s sim, or early once the task has been done for 1 s.

**Outputs:**
- `episodes.jsonl` with seed, pass, L_task, reasons, steps, inference ms, timing and clipping.
- `actions/*.npz`, the 10 Hz actions, so any policy episode can be replayed and re-judged on CPU. Replay reproduces the verdict: tested.
- Optional mp4s.
- `summary.json` with Wilson 95 % CIs and throughput.

### Smoke results

| run | n | L_full | L_task | notes |
|---|---:|---:|---:|---|
| expert mode, seeds 1000/0-19 | 20 | 20 (CI 84-100 %) | 20 | 0 rate-limit clips |
| expert mode, seeds 1000/0-59 | 60 | 58 (96.7 %, CI 89-99 %) | 60 | agrees with the native expert verifier on 58/60. Native fails 32 (harness passes). Harness fails 51 on arm-table contact (native passes): the 10 Hz subsampling and 67 ms ramp lag shift the jaw tips near the table at the grasp (planned tip clearance 2 mm) |
| expert mode, `--interp hold` | 20 | 19 | 20 | 40.5 % of frames rate-clipped. This is why `lag` is the default |
| dataset replay, train episodes 0-4 | 5 | 5 | 5 | 0 clips. State at step 0 identical (0.000°). Median 0.1°, p95 1.7°, max 6.6° (the ramp lag) |
| policy, dryrun ckpt 000030 (HF) | 3 | 0 | 0 | plumbing only. Raw images (before the h264 default). One episode grasped and lifted 68 mm, then released 57 mm high outside the bin |
| **policy, live ckpt 001400** | 20 | **10 (50 %, CI 30-70 %)** | **15 (75 %, CI 53-89 %)** | see below |

**Image check (dataset replay):**
- Raw harness frame vs the decoded training frame at step 0: PSNR 31-34 dB.
- Same frames after the dataset's own h264 sequence encode: 45-99 dB, where 99 means bit-identical.
- So the renders match training exactly, up to codec noise.
- The per-frame h264 pass used in policy mode brings step 0 to 37-39 dB (mean abs diff 1.4/255 instead of 2.5/255).

**Checkpoint 001400 failures:**
- 5 never grasped the cube (time limit).
- 5 `arm_table_collision`; 4 of these had the task done (L_task pass).
- 1 `arm_bin_collision`.

Other measurements:
- 4.0 % of frames were rate-limited (max excess 0.12 rad), and 50 steps were joint-range-clipped.
- Chunk inference: 833 ms median while sharing the GPU with training.
- Mean episode: 31 s wall for 21.8 s sim. Policy 22.4 s, render 6.1 s, physics 1.9 s.

## 3. Throughput and resources

- **Policy:** 1 replica, bf16, 12-14 GB VRAM, load 144 s. About **116 episodes/h** (3600 / 31 s) with training live.
- **Two replicas:** used 28 GB and fit in memory, but they raised training `updt_s` from 1.20 to 1.79 s/step. One replica raised it only to about 1.25.
- **Stall caused by this harness:** when I stopped the 2-replica run with `pkill`, its spawned pool workers kept their replicas loaded until about 11:44. During that window all GPU work stalled for about 6.5 min (11:37-11:44). Pool workers now exit when their parent dies.
- **While training is live: ONE policy process only.** K>1 should be measured after training ends.
- **Expert/replay modes:** CPU only, about 2,200 episodes/h on 3 nice-19 workers. A rendered replay takes about 13 s per episode.
- **CPU:** at most 4 nice-19 workers. Training `data_s` stayed at 0.13-0.31 s.

## Known limitations

- Policy-mode images go through a single-frame h264 encode, not a sequence encode as in training. The residual gap is 37-39 dB, against 45-99 dB for a sequence encode.
- The 10 Hz action is ramped from the previous action, which runs about 67 ms behind the expert's timing. Replay still succeeds (5/5 and 58/60).
- `arm_table_collision` is the main L_task/L_full gap for the policy. The 001400 run predates contact-geom logging (`arm_table_first_contact`), and its actions were not saved, so the touching geom is unknown for those 5 episodes. Later runs record both.
- Early stop ends the episode 1 s after the task is done, so later collisions while the arm retreats are not judged.
- 20 seeds give a wide CI (±20 points). The full SFT eval is scheduled after training ends (~20:00 UTC).
