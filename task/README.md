# SO101-PickPlace-State

An mjlab reinforcement-learning task. A LeRobot SO-101 arm (5 joints plus a one-joint jaw) picks a 30 mm cube off a table and drops it into a shallow bin.

The reward is **object-centric**: its main dense term makes the *object* follow the object trajectory from a human demo video. The video's object path, grasp and release times, and approach side arrive as a JSON **task spec** (see "Task spec format" below).

Training happens later on an AMD MI300X. Everything here was built and smoke-tested on the Windows laptop.

## 1. Status

| Item | State |
|---|---|
| SO-101 MJCF (Menagerie `robotstudio_so101`, Apache-2.0) | cloned, unmodified, at `manip/third_party/menagerie_so101` |
| mjlab robot config: STS3215 servo model, collision masks, grasp site | done (`so101_pickplace/robot_model.py`, `so101_constants.py`) |
| Grasp-physics test (CPU MuJoCo, IK pick, no welds) | **passes**: 12 pick cases plus a negative control (section 4) |
| Gripper and workspace measurements | done (`measurements/so101_measurements.json`) |
| `SO101-PickPlace-State` registered | done (entry point, plus a no-install wrapper) |
| Smoke test on Warp CPU, 2 envs | env builds and steps; rewards finite; 2 PPO iterations; ONNX export matches torch to 1e-7 (section 6) |
| Scripted pick-and-place through the env (MuJoCo Warp, full DR) | both envs grasp, lift ~82 mm, place and release in the bin within 1–6 mm of centre (section 6) |
| mjlab CLI train path, `eval_policy.py` | CLI: 1 iteration through `mjlab_cli.py train` (gate → train → ONNX). Eval: run on the 2-iteration smoke checkpoint; it works, and the untrained policy scores 0 % as expected (`logs/eval_smoke.log`). |
| Training | **not run**, by instruction |
| MI300X / ROCm run | **not run** (no access from here) |

## 2. Layout

```
manip/task/
  so101_pickplace/
    robot_model.py      pure MuJoCo: loads the MJCF, servo numbers, sites, collision masks
    mj_scene.py         pure MuJoCo: object / bin / riser specs, CPU test scene, solver options
    ik.py               pure MuJoCo: damped-least-squares IK (tests and tools only)
    task_spec.py        pure Python: task-spec JSON parser/validator (format v2)
    so101_constants.py  mjlab EntityCfg for the arm
    env_cfg.py          the task (obs / actions / rewards / DR / terminations)
    mdp/                command (object + bin + reference path), action, obs, rewards, events
    rl_cfg.py           PPO settings
    runner.py           mjlab manipulation runner + deploy_contract.json sidecar
    register.py         registers SO101-PickPlace-State (mjlab.tasks entry point)
    platform_patches.py opt-in ROCm switches (env vars)
  specs/default_task_spec.json            hand-written placeholder spec
  specs/example_from_video_task_spec.json the shape the demo-video worker should emit
  tests/test_grasp_physics.py             physics-honesty hard gate
  tests/test_task_spec.py                 spec parser checks (also validates one file: `python -I tests/test_task_spec.py spec.json`)
  scripts/measure_so101.py                gripper + workspace measurements
  scripts/smoke_env.py                    laptop smoke test (env, scripted pick, PPO, ONNX)
  scripts/eval_policy.py                  headless checkpoint evaluation
  scripts/mjlab_cli.py                    mjlab train/play with the task registered, no install
  measurements/                           JSON outputs of the above
  pyproject.toml                          `pip install -e manip/task` -> entry point
```

Nothing under `sprint/third_party/mjlab` is edited.

## 3. Commands

### Laptop (Windows, `sprint/.venv-train`; mjlab 1.5.0, torch 2.7.1+cu126, warp 1.14.0, mujoco 3.11.0)

The laptop hit "paging file is too small" when several torch processes loaded at once. Check `tasklist` before starting any torch or warp process, and run one at a time.

```bash
cd "C:/Users/vieth/Documents/lablab hackathon/mi300x-rss/sprint"
PY313=/c/Users/vieth/AppData/Local/Programs/Python/Python313/python.exe

# physics gate (mujoco + numpy only, about 15 s)
$PY313 -I -m pytest -q -p no:cacheprovider manip/task/tests/test_grasp_physics.py
$PY313 -I manip/task/tests/test_grasp_physics.py          # same, and writes measurements/grasp_physics_result.json
$PY313 -I -m pytest -q -p no:cacheprovider manip/task/tests/test_task_spec.py
# gripper / workspace numbers (about 50 s)
$PY313 -I manip/task/scripts/measure_so101.py --samples 400000
# env smoke test on the Warp CPU device: ~70 s with cached kernels (first compile adds 1-2 min); ~2.5 GB commit
MJLAB_WARP_QUIET=1 ./.venv-train/Scripts/python.exe -I -u manip/task/scripts/smoke_env.py --device cpu --num-envs 2
# tiny training run through mjlab's own CLI (CPU; no install needed)
./.venv-train/Scripts/python.exe -I manip/task/scripts/mjlab_cli.py train SO101-PickPlace-State \
  --gpu-ids None --env.scene.num-envs 4 --agent.max-iterations 2 --agent.num-steps-per-env 8
```

### MI300X (assumes a working mjlab + ROCm install, per AMD-Ecosystem/warp#20)

```bash
cd sprint
pip install -e manip/task                 # editable: asset paths resolve into this checkout
export MJLAB_DISABLE_CUDA_GRAPH=1         # ROCm Warp has no graph capture (platform_patches.py)
export MJLAB_GRAPH_CONDITIONAL=0          # needed per AMD-Ecosystem/warp#20
export SO101_TASK_SPEC=$PWD/manip/task/specs/default_task_spec.json   # or the demo-video spec
python -m mjlab.scripts.list_envs | grep SO101                         # sanity: task is registered
# train through the wrapper: it runs the grasp-physics gate first and refuses to train on failure
python manip/task/scripts/mjlab_cli.py train SO101-PickPlace-State \
  --env.scene.num-envs 4096 --agent.max-iterations 4000 --agent.logger tensorboard
# (python -m mjlab.scripts.train SO101-PickPlace-State ... also works, but skips the gate)
# checkpoints, <run>.onnx and deploy_contract.json land every 100 iterations in
#   logs/rsl_rl/so101_pickplace_state/<date_time>/
python manip/task/scripts/eval_policy.py --device cuda:0 --num-envs 256 \
  --checkpoint logs/rsl_rl/so101_pickplace_state/<run>/model_4000.pt               # noise off
python manip/task/scripts/eval_policy.py --device cuda:0 --num-envs 256 --train-cfg-dr \
  --checkpoint logs/rsl_rl/so101_pickplace_state/<run>/model_4000.pt               # full DR
python -m mjlab.scripts.play SO101-PickPlace-State --checkpoint-file <run>/model_4000.pt  # needs a display
```

The task spec is read when the task is registered, so set `SO101_TASK_SPEC` before the process starts. One spec applies per training run.

## 4. Robot model, measurements, and the physics gate

### Source and licence

- **MJCF:** MuJoCo Menagerie `robotstudio_so101`, commit `0059d43` (2026-10-07), sparse-cloned into `manip/third_party/menagerie_so101`.
- **Licence:** Apache-2.0 (`robotstudio_so101/LICENSE`).
- **Upstream:** Menagerie derived the model from TheRobotStudio's `so101_new_calib.xml`.
- **Collision:** this Menagerie revision adds primitive gripper collision boxes and decomposed jaw meshes. An earlier revision used a convex hull that filled the jaw gap; see `lablab-dinner-table/FINDINGS-SIM.md` section 5.

### Edits made at load time

These are applied in `robot_model.so101_spec()`; the file on disk is never touched.

- The XML actuators are removed, and mjlab adds its own.
- New sites: `grasp_site`, `fixed_tip_site` and `moving_tip_site`.
- Every collision geom gets a name.
- Collision masks: robot geoms are `contype 2 / conaffinity 3`; table, object and bin are `1 / 1`. So:
  - Robot-table, robot-object and robot-robot contacts are all on. Robot-robot means non-adjacent links; 12.6 % of uniformly random joint poses self-penetrate, so self-collision is real.
  - The model contains **no** equality constraints, welds or mocap attachments. The test asserts this.

### Servo model (Feetech STS3215) and sources

| Quantity | Value | Source |
|---|---|---|
| kp / kv | 998.22 N·m/rad / 2.731 N·m·s/rad | Menagerie defaults, derived from LeRobot servo P=16 (Gregory119/RBE501-RL-arm-project). LeRobot writes P=16, I=0, D=32 (`config_so_follower.py`, checked 2026-10-09). |
| Arm torque limit | 2.94 N·m nominal; DR ×0.55–1.0 | 30 kg·cm is the 12 V STS3215. The BOM ships the **7.4 V C001** version: 16.5 kg·cm @ 6 V = 1.62 N·m (TheRobotStudio README). |
| Jaw torque limit | 50 % of arm (1.47 N·m) | LeRobot `configure()` writes Max_Torque_Limit=500, Protection_Current=250 for the jaw "to avoid burnout" (checked 2026-10-09). |
| Armature / viscous damping / friction loss | 0.028 kg·m² / 0.60 N·m·s/rad / 0.052 N·m | Menagerie `sts3215` class |
| No-load speed | ~4.7 rad/s (informational) | Saturated-torque terminal speed in sim is about 4.8 rad/s with the damping above. |
| Command latency | 0 or 1 control step (100 ms), drawn per episode with p = 0.5 | design update (Squint) |

### Gripper measurements

From `scripts/measure_so101.py`, in the gripper-body frame. The fingers point along −z and the moving jaw closes along −x.

| | |
|---|---|
| Jaw joint range | −0.1745 … 1.7453 rad; the pads meet at the lower limit |
| Mouth (closing-axis distance from the fixed-jaw face to the moving tip) | 48 mm @ 0.42 rad, 63 mm @ 0.61, 70 mm @ 0.71, 87 mm @ 1.0, **max 107 mm @ 1.745**. The moving tip also swings up: it is 32 mm above the fixed tip at 1.0 rad and 86 mm above at full open. |
| Pad-to-pad 3-D gap, max | 111.5 mm (mid pads), 123.8 mm (tip pads) |
| Fixed-jaw inner face / lowest point | x = −7.5 mm / z = −107.2 mm. Found by probing with `mj_geomDistance`; the tip spheres protrude 2.3 mm past the pad boxes. |
| `grasp_site` | (+7.5, 0, −92) mm: the centre of a 30 mm object squeezed against the fixed jaw, 15.2 mm above the lowest jaw point |
| Squeeze force on a 30 mm cube | ~20 N per side at 1.47 N·m; ~11 N at 0.81 N·m (weak 7.4 V jaw) |

### Workspace

Base on the table (mount height 0). Sampled 400 k joint poses, keeping only poses with:
- no self-contact,
- static gravity torque under 1.62 N·m on every joint,
- every robot collision geom (AABB corners) at least 2 mm above the table.

Radii are measured from the pan axis at (0.0388, 0) in the base frame. "Band" means the grasp site 15–25 mm above the table, i.e. at the centre of a 30 mm cube.

| Finger axis within … of straight down | Radius at which the grasp site can sit in the band | Table area (pan ±110°) |
|---|---|---|
| 15° | 6 – 32 cm | 1897 cm² |
| 30° | 6 – 35 cm | 2283 cm² |
| 45° | 6 – 38 cm | 2703 cm² |
| any | 6 – 43 cm | 3481 cm² |

- **Lowest reachable grasp-site height with a near-vertical (≤ 15°) approach:** 14.5–19.5 mm above the table across r = 5–32 cm. The limit is jaw-tip clearance, not kinematics: with no table, the same approach reaches 218 mm below the base.
- **Highest grasp-site height with ≤ 15° approach** (wrist_flex runs out): 51 mm at r = 5 cm, 102 at 10, 126 at 15, 133 at 17.5, 114 at 25, 72 at 30. So a vertical-fingered carry above ~13 cm is impossible, and the fingers must tilt.
- **Mount height:** H = 0, 2, 4, 6, 8 and 10 cm give near-identical areas (1.8–2.1 k cm² at ≤ 15°). **Chosen: H = 0**, the base clamped on the table like the real arm. A riser is supported (`mount_height` in the spec, plus a collision block).
- **Object:** a **30 mm cube, 25 g nominal**. Its centre (15 mm) is reachable top-down. With the jaw at 0.75 rad (~72 mm mouth at the moving tip) it has ~20 mm of clearance per side. Size DR is ±3 mm.
- **Note:** this contradicts `lablab-dinner-table/FINDINGS-SIM.md` section 3 ("lowest near-vertical grasp centre z = 0.085 m"). That number was measured on the older Menagerie revision whose jaw hull filled the mouth, with a different grasp-centre definition. It does not hold for this model.

### Physics gate: `tests/test_grasp_physics.py`

Plain CPU MuJoCo with the task's exact robot, servo model and solver options (5 ms, implicitfast, elliptic, impratio 10). The pick is scripted:
1. IK to a pre-grasp above the object, fingers vertical, fixed jaw 4 mm clear of the object face.
2. Descend, close, lift 8 cm, hold 1 s, open.

Joint *targets* are interpolated and the servos must track them through the dynamics. Nothing writes the object's state after placement.

Each case asserts:
- the object lifts at least 5 cm;
- on **every** airborne step, both jaws touch it and nothing else does;
- the vertical contact force on the object equals its weight within 10 %;
- slip during the hold is under 3 mm;
- after opening, the object is back on the table within 1 s with zero jaw contacts.

| Case | Lift | Airborne steps (both jaws / foreign contact) | Contact Fz vs weight | Slip | Fall after open |
|---|---|---|---|---|---|
| cube 30 mm, r = 20 cm | 79.7 mm | 499 / 0 | 0.293 vs 0.294 N | 0.13 mm | 0.21 s |
| cube 30, r = 12 cm / r = 28 cm | 79.7 / 82.0 mm | 499 / 502, 0 foreign | ✓ | ≤ 0.15 mm | 0.21 s |
| cube 30, 7.4 V servos (arm 1.62, jaw 0.81 N·m), finger μ 0.5, 50 g | 80.0 mm | 501 / 0 | 0.488 vs 0.491 N | 0.15 mm | 0.21 s |
| cube 27 mm r = 27 / 33 mm r = 13 (±3 mm DR corners) | 83.3 / 79.2 mm | all both-jaw, 0 foreign | ✓ | ≤ 0.14 mm | ≤ 0.22 s |
| cube 25.5 / 34.5 mm (±15 %) | 80.3 / 78.9 mm | all both-jaw, 0 foreign | ✓ | ≤ 0.15 mm | ≤ 0.22 s |
| cylinder 30×40 mm, nominal / ±3 mm / weak servos + μ 0.4 + 50 g | 78.5–83 mm | all both-jaw, 0 foreign | ✓ | ≤ 0.16 mm | ≤ 0.23 s |
| **negative control**: same script, jaw never closes | **0.0 mm** (max z +0.1 mm) | 0 | — | — | — |

Full numbers are in `measurements/grasp_physics_result.json`.

The first version of this test **failed** for the weak-servo and +15 % cases. The fixed-jaw tip protrudes 2.3 mm past the pad box, so with zero lateral clearance it landed on the cube's top edge and tipped it 20°. The fix was geometric: the measured inner face and 4 mm clearance. No success threshold was loosened.

## 5. Task design

- **Control:**
  - 10 Hz policy (decimation 20 × 5 ms).
  - The action is a **delta on the previous joint target**: `target += clip(a, −1, 1) · v_max · 0.1 s`, clamped to 95 % soft joint limits.
  - `v_max` is 0.5 rad/s for the arm and 2.0 rad/s for the jaw, both config values in `env_cfg.py`. The arm figure matches the real-servo command speed used by livekit-examples/so-frame; the jaw is quicker so it can open and close within one reach.
  - The episode starts with target = measured position. 0–1 step latency is drawn per episode.
- **Episode:** 15 s (150 steps), or the reference length + 3 s if longer.
- **Actor observations (25-D for the cube):** only what the real loop has.

  | Term | Dim | Notes |
  |---|---|---|
  | joint position − home | 6 | encoder frame; Gaussian noise **σ ramps 2° → 5°** over 60 k env steps |
  | own joint target − home | 6 | exact |
  | object xy | 2 | table tracker: σ = 1 cm, 5 % per-step dropouts hold the last value |
  | object yaw as (cos 4ψ, sin 4ψ) | 2 | box only; an AprilTag on the cube; held on the same dropouts |
  | bin xy | 2 | ±5 mm noise |
  | episode clock / reference length | 1 | |
  | last action | 6 | |

  The critic additionally sees true joint position and velocity, the object's 3-D pose and velocity relative to the gripper, the finger axis, jaw contacts, object size, and the reference point.
- **Reward** (weights × dt):

  | Term | Weight | Notes |
  |---|---|---|
  | object follows the demo trajectory | 4 | the main dense term; see the reference construction below |
  | reach | 1 | |
  | grasp (both jaws on the object) | 1 | |
  | lift | 2 | saturates at 5 cm |
  | carry towards the bin | 2 | |
  | placed (inside the walls, resting on the floor) | 4 | |
  | released (placed, no jaw contact) | 6 | |
  | approach side | 0.25 | finger-axis alignment with the spec's direction near the object |
  | dropped | −2 | |
  | action rate | −0.02 | |
  | joint limits | −10 | |

  Reference for the main term: the demo's object path is smoothed, resampled at 0.1 s and moved onto this episode's sampled start and bin. The start offset fades into the end offset between grasp and release, which keeps the demo's lift height and arc. The term is `exp(−d²/σ²)`, where d is the distance to the nearest reference sample within ±1 s of now (tolerates being early or late).

  **No term makes the gripper follow the human hand.**
- **Terminations:** time-out; gripper presses the table with more than 15 N; object knocked beyond 0.42 m, thrown above 0.35 m, or non-finite.
- **Domain randomisation:**

  | What | Range | Mode |
  |---|---|---|
  | object size (edge) | ±3 mm | reset |
  | object mass | ×0.61–1.65 (pseudo-inertia) | startup |
  | object friction | 0.4–1.0 | startup |
  | finger friction | slide 0.4–1.4, plus YAM's spin/roll | startup |
  | arm kp, kd | ×0.5–1.5 each, per joint | startup |
  | jaw kp | 500–2000 N·m/rad absolute | startup |
  | jaw kd | ×0.5–2 | startup |
  | torque limits | ×0.55–1.0 | startup |
  | encoder bias | ±0.02 rad | startup |
  | start pose | ±0.08 rad around home | reset |
  | action latency | 0–1 step | per episode |
  | observation noise / dropouts | as listed above | per step |

  The jaw kp range is Squint's 500–2000, taken in the same N·m/rad units. Squint's damping of 50–200 is a PhysX value about 36× our calibrated kv, so the same ×0.5–2 spread is applied around the MuJoCo nominal instead. This unit mapping is a judgement call; check it against a real jaw step response.
- **PPO** (`rl_cfg.py`, untuned):
  - networks: actor 256-256-128, critic 512-256-128, ELU, observation normalisation, initial std 0.6;
  - γ 0.98, 32 steps/env, 4000 iterations, TensorBoard.
  - An ONNX file and `deploy_contract.json` (joint order, action rule, max deltas, observation layout, frames, home pose) are written at every save.

## 6. Smoke test result (laptop, Warp CPU device, 2 envs)

See `measurements/smoke_cpu.json` and `logs/smoke_cpu.log`.

The run below is `scripts/smoke_env.py --device cpu --num-envs 2`, about 70 s once Warp's CPU kernels are cached. The first compile adds ~1–2 min.

| Check | Result |
|---|---|
| Env builds | yes, 0.3 s (cached kernels) |
| Layout | 10 Hz, decimation 20, **150-step (15 s) episodes**, action dim 6 |
| Actor observation | 25-D: joint_pos 6, joint_target 6, object_xy 2, target_xy 2, clock 1, actions 6, object_yaw 2 |
| Critic observation | 50-D |
| 20 random-action steps | observations and rewards finite; ~130–200 physics steps/s on the CPU device (2 envs) |
| **Scripted pick-and-place through the env** | see below |
| PPO | 2 iterations × 8 steps; `model_0.pt`, `model_1.pt` saved; `smoke_cpu.onnx` exported |
| ONNX vs torch policy (onnxruntime) | max abs diff 7.8e-8 |
| mjlab CLI path | `mjlab_cli.py train … --gpu-ids None --env.scene.num-envs 2 --agent.max-iterations 1` ran the grasp gate (PASS), trained 1 iteration, and wrote `model_0.pt`, `<run>.onnx`, `deploy_contract.json` and `params/*.yaml`. The mjlab tree's git status stayed clean (`logs/cli_train_smoke.log`). |

How the scripted pick-and-place works:
- It runs on MuJoCo Warp physics under the full startup DR, through the speed-capped target-delta action term and the contact sensors.
- IK goals are solved on the CPU copy of the model. Jaws close across the cube faces.
- The object is never written.

| | env 0 | env 1 |
|---|---|---|
| Completed | all 9 stages (pre-grasp → retreat) by step 116 | by step 111 |
| Jaw closed | step 41 | step 41 |
| Peak object height | 81.7 mm | 81.5 mm |
| Steps with both-jaw contact / lifted | 57 / 39 | 54 / 37 |
| Placed, released at end | yes, yes | yes, yes |
| Final cube offset from bin centre | 0.9, 1.5 mm | 5.5, 2.7 mm |
| DR draws | 37 g, jaw kp 1906, jaw torque 0.96 N·m, finger μ 0.93 | 33 g, kp 1618, 1.02 N·m, μ 0.79 |

Earlier runs of the same script failed on the controller side, and each failure was fixed there:
1. Jaw goals beyond the 95 % soft limit were never "reached".
2. Jaws closing across a cube corner lost the cube.
3. Episodes time out at the episode length mid-script.

The time-out is lifted for this section only, and the step it would have fired at is reported. Even a sequential scripted controller at the 0.5 rad/s cap needs 11–20 s, which is why episodes are 15 s and the default reference is 11 s.

## 7. Task spec format (`so101_pickplace_task_spec/v2`)

All positions are in the **robot base frame**: origin at the base's mounting point on the table, x forward (the arm's direction at shoulder_pan = 0), y left, z up, table at z = 0. Units are metres, seconds and radians. `specs/example_from_video_task_spec.json` shows the full shape.

| Field | Meaning |
|---|---|
| `format` | `"so101_pickplace_task_spec/v2"` (v1 is accepted; its `hand_path` is ignored with a warning) |
| `frame` | must be `"robot_base"` |
| `mount_height` | base height above the table (m). Default 0; >0 adds a collision riser. |
| `object.kind` | `"box"` (default) or `"cylinder"`; see risks below |
| `object.size`, `object.height`, `object.mass` | edge or diameter, 20–50 mm; cylinder height; kg |
| `object.start_pose_range.{x,y,yaw}` | uniform ranges for the object start (centre on the table) |
| `place_target.pos_range.{x,y}` | uniform ranges for the bin centre. The bin is 11 × 11 cm inside, with 25 mm walls. |
| `place_target.min_separation` | minimum xy distance between object start and bin centre (rejection sampled) |
| `place_target.success_xy_margin` | how far inside the walls the object centre must be |
| `approach.direction` | finger-axis direction near the grasp, **from the hand** (approach side), e.g. `[0,0,-1]` = top-down |
| `approach.weight`, `approach.gate_std` | small shaping weight; distance gate (m) |
| `object_trajectory.waypoints` | `[[x, y, z, t], …]`: the **object centre** over time on the demo clock (seconds), ≥ 2 points, z ≥ 0 |
| `object_trajectory.grasp_time`, `release_time` | demo-clock times the **hand** closed on and let go of the object |
| `object_trajectory.robot_time_scale` | robot time = demo time × scale. Use ~2 for human speed; the SO-101 is capped at 0.5 rad/s. |
| `object_trajectory.smoothing_window_s`, `resample_dt` | centred moving average (s); resample period (s) |
| `object_trajectory.weight`, `std`, `time_slack_s` | tracking reward weight; σ (m); ± time window (s) |

How the spec is checked and used:
- Validation rejects any start range, bin range or trajectory point outside the measured reach (6–32 cm from the pan axis, +3 cm slack for the trajectory).
- The trajectory is warped per episode, so it only needs to be *shaped* right. Absolute xy may differ from the sampled start and bin.
- With no `object_trajectory`, a synthetic rest–lift–carry–lower path between the range centres is used.
- Hand-path waypoints are deliberately not used.

## 8. Known risks

1. **Nothing is trained.** The reward weights, speeds and PPO settings are a starting point. The scripted pick through the env (section 6) is the only evidence that the env pipeline can reach every reward stage.
2. **MuJoCo Warp contact differs from CPU MuJoCo.**
   - mujoco_warp 3.11 reports that CYLINDER-BOX, CYLINDER-MESH and CAPSULE-CYLINDER pairs get at most one contact. A cylinder standing on the box-shaped bin floor would rock, and its grasp has single-point pad contacts. That is why the default object is a cube; use cylinders only after checking them in the env.
   - The CPU grasp gate does not cover mjwarp; the smoke test's scripted pick does.
3. **Warp CPU quirk in mjlab.** Un-randomised model fields are not broadcast to `nworld` rows on the CPU device, so `soft_joint_pos_limits[env_ids]` fails in mjlab's `reset_joints_by_offset`. This task uses its own `reset_joints_around_home`. It is not an issue on CUDA or ROCm, where stride-0 fields broadcast.
4. **wrist_roll zero.** livekit-examples/so-frame (unlicensed; read for conventions only, no code copied) measured a **+90° wrist_roll offset** between the `so101_new_calib` model zero and a LeRobot-calibrated real arm, and found the other joints at 0. Menagerie derives from that same model. Verify every joint's zero and sign on the real arm before the first rollout.
5. **Servo torque.** The nominal 2.94 N·m is the 12 V servo; the BOM arm is 7.4 V (~1.6–1.9 N·m). DR covers both, but a policy evaluated at nominal will look stronger than the real arm.
6. **Jaw damping DR unit mapping** (section 5) is a judgement call; check it against a real jaw step response.
7. **Time-indexed reference.** If the robot is much slower than `robot_time_scale` predicts, the tracking term pulls in the wrong place. The ±1 s slack and the shaping terms limit the damage; set the scale from a scripted rollout.
8. **ROCm.** AMD's mujoco_warp fork has an open contact-dropping bug (PR #11 in the sprint PLAN). Manipulation is contact-heavy, so run `eval_policy.py` early and watch the `grasped` metric.
9. **Cost of state recomputation.** Reward and observation terms share a per-step cache (`mdp/state.py`), but without CUDA graphs on ROCm, kernel-launch overhead per term is real. Profile one iteration before a long run.
