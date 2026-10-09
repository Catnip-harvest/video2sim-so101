# Real-to-sim data engine: one human demo in, many verified SO-101 demos out

**What it does.** You film yourself picking up a cube and dropping it in a bin, once.
`manip/demo/` turns that clip into a small JSON *task spec*: where the cube started,
the path the **cube** took through the air, when the hand closed and opened, and which
side the hand came from. This folder turns that one spec into hundreds of
robot demonstrations of the same task on a simulated LeRobot SO-101 arm, each one:

- **different**: the cube position, yaw, size, mass and friction change, as do the bin
  position, the servo strength (including the 7.4 V servo the kit ships with), the
  lighting, the table and the camera height and pitch;
- **physically real**: full MuJoCo contact physics with position servos only. The cube
  is never glued, latched or teleported to the gripper, and it moves only because the
  two jaws squeeze it;
- **checked**: an episode is kept only if a verifier (below) confirms that the jaws
  held the cube, the arm hit nothing, and the cube ended at rest inside the bin with
  the gripper open. Every rejected episode is logged with the reason;
- **saved in LeRobot format** (`LeRobotDataset` v3.0), ready for `lerobot` imitation
  learning (ACT, Diffusion, SmolVLA, pi0).

There is no reinforcement learning. A scripted expert (inverse kinematics plus joint
trajectories capped at the real servo speed) does the motion. Physics plus the verifier
decide which runs count.

```
human clip --(manip/demo)--> so101_task_spec.json --(this folder)--> N x [sample variation
  -> plan (scripted expert, re-plans from the simulated state at 3 points)
  -> simulate (CPU MuJoCo, 210 Hz physics, 30 Hz recording) -> verify -> keep / reject]
  -> LeRobot v3.0 dataset (+ optional rendered front and wrist cameras)
```

## Where each part comes from

| Part | Taken from | What we do differently |
|---|---|---|
| One video -> many demos by re-posing the object and replaying its tracked path, smoothed **and** densely interpolated | **R2R2R**, Real2Render2Real (Yu et al. 2025, `uynitsuj/real2render2real`, MIT). Their mug task fell from 80 % to 0 % without interpolation. | R2R2R replays kinematically and only renders. We **simulate** every episode with contact physics and throw away the ones that fail. |
| The **object** follows the demo, not the hand | **X-Sim** (portal-cornell/X-Sim). We take the idea only; there is no licence file, so no code. | A human arm hangs from a shoulder above the table and the SO-101 stands on it, so hand paths do not transfer. The cube's path is mapped onto each episode's pick and place exactly as the mjlab task's tracking reward does. |
| Randomisation ranges: servo gains, jaw stiffness 500–2000, colour and light jitter, tight camera randomisation | **Squint** (arXiv 2602.21203, `aalmuzairee/squint`, MIT): SO-101 sim-to-real, 73/80 real | Same jaw-stiffness range. The camera band (height ±8 cm, pitch ±8°, xy ±3 cm, FOV ±3°) is wider than Squint's ±2–2.5 cm, as `manip/DESIGN.md` asked. |
| SO-101 model | **MuJoCo Menagerie** `robotstudio_so101` (Apache-2.0) via `manip/task/so101_pickplace` | Reused unmodified: load-time edits, STS3215 servo numbers, measured jaw geometry, grasp site, and the IK used by the 12-case grasp-physics gate. |
| Physics-honesty rules (no welds or latches; both jaws on an airborne object; the object falls when released) | `manip/task/tests/test_grasp_physics.py` | The same rules are checked on **every physics step of every episode**, not just in 12 test cases. |

## How an episode is made

1. **Sample a variation** (`r2s/variations.py`). Attempt `i` with `--seed s` always gets
   the same draw. The cube's start is the spec's pick ±3 cm with the spec's yaw ±0.3 rad;
   its size is the spec's ±3 mm, clamped to the 20–50 mm band the grasp gate validated;
   its mass is ×0.61–1.65 and its friction 0.4–1.0. Finger friction is 0.4–1.4. The bin
   centre is the spec's ±2 cm, at least 11 cm from the cube. Servo kp and kv are ×0.5–1.5
   per joint and jaw kp is 500–2000 N·m/rad. Torque is the 7.4 V STS3215 (1.62–1.91 N·m)
   in half the episodes and the 12 V one (2.94 N·m ×0.85–1.0) in the other half; the jaw
   gets 50 % of the arm's torque, as LeRobot configures it. The start pose is home
   ±0.05 rad. Render-only draws: table colour and checker, cube and bin colour, key-light
   direction and strength, ambient, headlight, and the front camera (height ±8 cm,
   pitch ±8°, xy ±3 cm, FOV ±3° around a 3/4 view).
2. **Plan and simulate in three stages** (`r2s/planner.py`, `r2s/rollout.py`). Each
   stage is planned from the *simulated* state at that moment, the way any scripted
   collector with state feedback works:
   - *Pick*: transit to a pre-grasp point offset towards the human's approach side,
     move in a straight line to a hover point, descend vertically with the fixed jaw
     4 mm clear of the cube face, close the jaw, squeeze 0.35 s. The open width comes from
     the demo's pinch width plus 4 cm.
   - *Lift and carry*: lift vertically 2 cm, then follow the human's cube path. The
     path is split into its chord and the arc around the chord. The chord is replaced by
     this episode's pick-to-place line, the xy arc is rotated and scaled with it, and the
     height arc is kept in metres. Three corrections apply only where needed and are
     logged per episode: stay above the lift height, ramp up to clear the bin walls near
     the bin, and stay under the measured reach ceiling (`manip/task/README.md` §4). Path
     points closer than 15 cm to the base are pushed out radially. Timing is
     max(demo time × `robot_time_scale`, servo speed cap).
   - *Place*: measure where the cube actually sits in the jaws and how much it has
     turned, re-centre over the bin, lower until the cube's lowest corner is 5 mm above
     the floor, open just enough to let go (cube + 2 cm, so the swinging jaw misses the
     wall), lift 6 cm, return home, rest 0.5 s.

   The inverse kinematics is the task's damped-least-squares solver, run on a copy of the
   model whose joint ranges are cut to 95 %. Among the solutions within tolerance it
   takes the one nearest the current pose, rejecting any that would put the arm into
   itself or the table, or the fingers into the cube on the way in. If a waypoint can't
   be reached, it tries other finger tilts (0, ±10°, ±20°, ±30°, radial only, since a
   5-joint arm can only pitch its fingers in its own vertical plane); if that fails, it
   shrinks the human arc's height (×0.7, ×0.4, flat).
3. **Servo speed caps.** Targets move at most 0.5 rad/s on the arm (the real STS3215
   command speed the mjlab task uses) and 2 rad/s on the jaw. Plans run at 90 % of the
   cap, then a moving average and a final rate limiter guarantee the cap. The validator
   re-checks it on the saved actions.
4. **Physics.** CPU MuJoCo 3.11 with the grasp gate's solver (implicitfast, elliptic
   cones, impratio 10). The timestep is 1/210 s, so 7 physics steps make one 30 Hz
   frame. Within a frame the servo goal ramps linearly from the previous target to the new
   one (*first-order hold*), standing in for the STS3215's onboard motion profile. See
   "Findings" for why this matters. After the initial placement nothing writes positions
   or velocities. The model is asserted to have no equality constraints and no mocap
   bodies, and the cube is a free body with no actuator.

### The verifier: an episode is kept only if all of these hold

Every check runs on every physics step (210 Hz), not just at recorded frames.

| Reason logged on failure | Rule |
|---|---|
| `ik_unreachable`, `joint_limit_plan` | the expert found a plan inside 95 % of the joint ranges |
| `object_disturbed_before_grasp` | the cube moved < 5 mm before the jaw started closing |
| `grasp_failed` | after the lift, the cube is ≥ 1 cm up and both jaws touch it |
| `dropped_or_slipped` | from lift-off (cube 5 mm up) until the jaw opens, **every** step has contact with **both** jaws |
| `foreign_contact_while_carried` | while carried, the cube touches nothing but the jaws: no table, bin wall or arm link (the bin floor is allowed only in the final lowering) |
| `arm_body_touched_object` | no non-jaw robot geom ever touches the cube |
| `arm_table_collision`, `arm_bin_collision`, `self_collision` | never, at any step |
| `joint_limit` | no joint more than 0.01 rad past its range |
| `torque_saturation` | no arm servo "fails to hold": at ≥ 98 % torque **and** sagging > 0.02 rad for > 0.25 s while its target is still |
| `tracking_error` | no arm joint lags its target by > 0.1 rad |
| `speed_cap` | the targets respect 0.5 rad/s (arm) and 2 rad/s (jaw) |
| `not_lifted` | the cube rose ≥ 2 cm |
| `not_in_place_region` | at the end, the cube's centre is inside the bin walls by the spec's margin (5 mm) and it rests on the bin floor (±4 mm) |
| `not_resting` | the cube moves < 2 cm/s at the end |
| `not_released` | no jaw touches the cube during the final 0.5 s |

The *primary* reason is whichever happened first in sim time. If the cube drops and
the re-planner then can't reach it, the log blames the drop.

Two tests show the verifier rejects things:
- **Negative control** (`tests/test_datagen.py`): the same episode with the jaw never
  closing is rejected as `grasp_failed`, and the cube never rises 5 mm.
- **Ablation**: switching the servo to step commands (`--servo-hold zoh`) rejects
  27.5 % of sugar-cube attempts as drops (Findings).

### What a frame contains

| Key | Shape | Meaning |
|---|---|---|
| `observation.state` | 6 | measured joint positions at the start of the frame: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (`.pos`), **radians** |
| `action` | 6 | absolute joint position targets held for the next 1/30 s, radians |
| `observation.environment_state` | 7 | cube position (m, robot base frame) and quaternion (w, x, y, z) |
| `observation.images.front` | 256×256×3 video | front 3/4 camera, randomised pose (only with `--render`) |
| `observation.images.wrist` | 256×256×3 video | the Menagerie model's own `wrist_cam` on the gripper's camera mount (only with `--render`) |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | 1 | LeRobot bookkeeping; task = "pick up the cube and place it in the bin" |

Beyond LeRobot's own files, `meta/datagen_episodes.jsonl` holds each episode's full
variation (every sampled number, including the camera pose), the verifier statistics,
the plan details (finger tilt, arc scale, how many path points were corrected and why)
and the phase start frames. `meta/datagen.json` holds the generator config and the
acceptance table.

**Units.** Joints are in radians in the MuJoCo convention. The real LeRobot SO-101
reports degrees for arm joints and 0–100 for the gripper, after its own calibration, and
`manip/task/README.md` §8 notes a +90° wrist_roll offset between this model and a
LeRobot-calibrated arm. Convert, and check every joint's zero and sign on the real arm,
before mixing this data with real data.

## Commands

### Laptop (Windows, no OpenGL: state-only datasets)

```bash
cd "C:/Users/vieth/Documents/lablab hackathon/mi300x-rss/sprint/manip/datagen"
PY=/c/Users/vieth/AppData/Local/Programs/Python/Python313/python.exe   # mujoco 3.11, numpy, av, duckdb
# tests (~5 s): determinism, no-weld model, one accepted episode, negative control, writer + video path
$PY -I -m pytest -q -p no:cacheprovider tests/test_datagen.py
# 50 verified episodes, 8 worker processes, state only
$PY -I generate.py --spec ../demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json \
    --n 50 --workers 8 --no-render --out runs/sugar_cube_n50
$PY -I validate_dataset.py runs/sugar_cube_n50/lerobot
$PY -I plot_trajectories.py --dataset runs/sugar_cube_n50/lerobot \
    --spec ../demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json --out runs/sugar_cube_n50/trajectories.png
# ablation: servo goal jumps once per frame instead of ramping
$PY -I generate.py --spec ... --n 50 --workers 8 --no-render --servo-hold zoh --no-dataset --out runs/ablation_zoh
```

`generate.py` options:
- `--n` accepted episodes wanted;
- `--workers` processes;
- `--seed`;
- `--render` / `--no-render`;
- `--size 256x256`;
- `--no-wrist`;
- `--codec h264|av1`;
- `--servo-hold foh|zoh`;
- `--max-attempts` (default 4n+50);
- `--finalize-only` (rewrite the dataset from staged episodes);
- `--no-dataset` (log and acceptance table only).

**Resumable**: re-running the same command skips every attempt already in
`attempts.jsonl` and redoes any accepted attempt whose staged files are missing.
**Deterministic**: the dataset is always the first N accepted attempt ids, so it doesn't
depend on `--workers` or scheduling. Two runs with 8 and 2–3 workers gave bit-identical
episodes. One heavy process at a time on the laptop: two parallel batches hit "Could not
allocate memory" during model compile.

### Droplet (Linux, 20 vCPU): the full rendered run

```bash
bash manip/datagen/run_droplet.sh                 # N=1000 per spec, both specs, 19 workers
N=200 SPECS=sugar bash manip/datagen/run_droplet.sh
```

The script:
1. creates a venv (mujoco 3.11, av, pyarrow, pandas, duckdb, matplotlib, lerobot);
2. picks `MUJOCO_GL=egl` and falls back to `osmesa`;
3. runs the grasp gate and these tests;
4. generates with `--render`;
5. validates, including loading with `lerobot.LeRobotDataset` (`--lerobot`);
6. writes `trajectories.png`, `contact_sheet.png` (front and wrist), `montage.mp4` and
   `side_by_side_ep{0,1,2}.mp4` (human clip left, robot episode right, aligned on grasp
   and release).

GL libraries need root: `apt-get install -y libegl1 libgl1 libosmesa6`.

## Measured numbers (laptop, 9 Oct 2026, no render, seed 0, 8 workers on 16 logical cores)

Acceptance (the deterministic prefix of attempts up to the 50th accepted one; full logs
in `runs/*/attempts.jsonl`, tables in `runs/*/acceptance.md`):

| | sugar cube (stock clip) | synthetic clip |
|---|---:|---:|
| attempts / accepted | 50 / 50 | 51 / 50 |
| **acceptance rate** | **100 %** | **98.0 %** |
| rejections (primary reason) | none | `arm_bin_collision` 1 |
| acceptance, 7.4 V / 12 V servos | 25/25, 25/25 | 24/24, 26/27 |
| cube size range | 20–23 mm | 36.5–42.5 mm |
| mean episode length | 523 frames (17.4 s) | 915 frames (30.5 s) |
| peak cube lift (median) | 60 mm | 95 mm |
| in-hand slip, lift-off to lowering (median / p90 / max) | 0.15 / 0.28 / 0.42 mm | 1.6 / 3.4 / 5.8 mm |
| cube tilt in the hand before lowering (median / max) | 0.4° / 10.5° | 24.6° / 37.2° |
| final cube offset from bin centre (median / p90) | 4.0 / 5.9 mm | 7.9 / 11.8 mm |
| max arm tracking lag | < 0.003 rad | < 0.014 rad |
| path corrections | raised for clearance in 50/50; none lowered or pushed | raised 50/50, reach ceiling 47/50, base keep-out 50/50, arc ×0.7 or ×0.4 in 3/50 |
| finger tilt used | vertical 50/50 (spec wants −10° inward: infeasible) | vertical 10, 15° outward 40 |
| pre-grasp from the human's side | 21/50 | 47/50 |

Ablation, same seeds, `--servo-hold zoh` (servo goal jumps once per frame):

| | sugar cube | synthetic |
|---|---:|---:|
| acceptance | **72.5 %** (50/69) | 89.3 % (50/56) |
| primary rejections | `dropped_or_slipped` 19 | dropped 4, foreign contact 1, self collision 1 |
| acceptance, 7.4 V / 12 V | 94.6 % / 46.9 % | 96.2 % / 83.3 % |
| in-hand slip of the *accepted* ones (median) | 9.8 mm | 8.8 mm |

Speed (CPU time = planning + physics + verification, per worker process; laptop
hyper-threads, so per-core numbers on the droplet should be similar or better):

| | sugar cube | synthetic |
|---|---:|---:|
| CPU seconds per accepted episode | 2.65 | 8.5 |
| accepted episodes / s / core | **0.38** | **0.12** |
| simulated seconds per CPU second | 6.6× real time | 3.7× real time |
| 50 episodes, 8 workers, wall | 23 s | 71 s |

Projection for the droplet (19 workers, no render): about 7 sugar episodes/s, so 1000 in
about 2.5 min, and about 2.2 synthetic episodes/s, so 1000 in about 8 min. Rendering two
256×256 cameras adds an unmeasured cost per accepted episode (~1050 renders per sugar
episode and ~1830 per synthetic one). Measure it on the first droplet batch before
committing to N.

Figures: `runs/sugar_cube_n50/trajectories.png`, `runs/synthetic_n50/trajectories.png`.
Each has a top view of the 50 generated cube paths and bins with the human's cube path in
red, and the cube's height while carried. The two progress axes differ slightly: the
human's runs grasp to release and includes a dwell; the robot's runs lift-off to release
and includes the vertical lift and lowering.

## Findings worth knowing

1. **30 Hz step commands shake small cubes out of the jaws.** With the servo goal
   jumping once per 33 ms frame (zero-order hold), a kp ≈ 1000 N·m/rad servo snaps to
   every 15 mrad step. The jolts made 20 mm cubes creep 10–60 mm in the jaws and drop.
   The rate depended on arm torque: the same episode was accepted at 1.7 N·m and dropped
   at 2.2 and 2.9 N·m, while jaw torque and jaw stiffness made no difference. Ramping the
   goal across the frame cut median slip to 0.15 mm, the same order as the grasp
   gate's 0.15 mm static hold. The real STS3215 smooths its goal with an onboard
   acceleration and speed profile, so `foh` is the default; `zoh` stays as the ablation.
   I first blamed the hinged jaw's V-shaped mouth and was wrong. The A/B test above
   settled it.
2. **Large cubes rotate in the hand.** A two-finger pinch lets the cube pivot about the
   jaw-to-jaw axis. Synthetic-spec cubes (≈ 40 mm) turned a median 25° before
   lowering. The place stage now lowers by the measured lowest corner; before that fix,
   2 of 53 synthetic attempts drove a corner through the bin floor and were (correctly)
   rejected.
3. **The human's approach side is mostly honoured as a direction of travel, not a finger
   tilt.** The sugar-cube hand came in 67° from vertical, pointing towards the robot. The
   arm can't tilt its fingers inward at that radius without hitting itself or the cube,
   so every accepted sugar episode used vertical fingers. A share of episodes (21/50 sugar, 47/50 synthetic)
   approach from the human's side via the offset pre-grasp point; the rest fall back to
   straight down.
4. **The synthetic demo's 15 cm lift arc and near-base path are outside what an SO-101
   can do with a held cube.** The arc is lowered to the measured reach ceiling, and points
   are pushed out to 15 cm from the base. The trajectory figure shows both.

## Rendering status

- **Laptop:** no OpenGL backend works. `mujoco.Renderer` raises `gladLoadGL error` under
  the default, `glfw` and `wgl` backends, and `egl`/`osmesa` don't exist on Windows. So
  nothing here has been rendered from MuJoCo, and `generate.py --render` exits with that
  message.
- **What is tested without GL:** the PyAV encoder, the LeRobot writer's video path (one
  mp4 per episode, `from_timestamp` 0), image-statistics fallback, the validator's
  decode-every-frame check, `contact_sheet.py` and `side_by_side.py`. They ran on staged
  episodes with placeholder drawings substituted for renders (scratch only, not shipped).
- **Untested until the droplet run:** `mujoco.Renderer` itself, camera framing, and
  render speed.

Rendering is playback. Accepted episodes are re-rendered from their recorded full state
(`mj_forward` on a separate `MjData`), so it never touches the physics and rejected
episodes cost no render time.

## Dataset format validation

- **Laptop, DuckDB reader** (pyarrow's DLL is blocked by Windows Application Control
  here): `validate_dataset.py` passes all 56 checks on both datasets. It checks
  `info.json` fields (codebase v3.0), the episode, task and data tables, index
  continuity, per-episode frame indices and timestamps, task pointers, stats keys,
  speed caps on the saved actions, and that every episode's recorded cube pose shows
  table → ≥ 2 cm lift → at rest inside *its* bin. That last check is recomputed from the
  dataset alone, independent of the generator's verifier.
- **Not yet done:** loading with `lerobot.LeRobotDataset`. lerobot needs pyarrow, which
  can't load on this laptop. `run_droplet.sh` runs `validate_dataset.py --lerobot`, which
  loads the folder with LeRobot itself and compares items against the Parquet rows,
  including video frame shapes. Until that has run, the format is "written to the v3.0
  layout and field names, structurally validated", not "proven to load". On machines
  with pyarrow the writer uses pyarrow and pandas directly; the DuckDB path is the
  fallback.

## Known limits

- **One task family**: a cube into an open tray, top-down pinch. Other shapes need their
  own grasp rules. The task module's cylinder works in CPU MuJoCo, but is not wired here.
- **Scripted expert, not a human-like policy.** Every episode has the same phase
  structure (approach, descend, squeeze, lift, carry, lower, release, retreat, home),
  with only the variations above. There are no recoveries and no failed-grasp retries in
  the data.
- **Episodes are long**: sugar about 17 s, synthetic about 30 s.
  That comes from the 0.5 rad/s cap and the spec's `robot_time_scale = 2`.
- **Spec quality limits the "human" part.** The sugar-cube spec comes from a 3 s stock
  clip with the no-A4 fallback calibration ("expect several-cm errors"), a 16 mm cube
  clamped to 20 mm, and a cube that leaves view before set-down, so the human path ends
  in mid-air at 5 cm. The synthetic spec is a rendered test clip.
- **Sim-to-real is untested.** The servo model is Menagerie's (kp derived from LeRobot's
  P=16). FOH is an assumption about the servo's onboard motion profile. Soft-contact
  creep, the jaw's damping randomisation units and the wrist_roll zero (§8 of the task
  README) carry over.
- **Speed**: 57 % of CPU time is the Python IK and 30 % is per-step contact bookkeeping;
  physics is the minority. A vectorised IK or a C contact filter would roughly double
  throughput if needed.
- **Disk on the laptop**: the 50-episode runs take 21 MB (staged arrays plus the
  dataset). The full 1000-episode rendered runs belong on the droplet.

## Files

```
generate.py           CLI: sample -> plan -> simulate -> verify -> stage -> LeRobot v3.0 dataset
validate_dataset.py   structural checks (+ --lerobot: load with LeRobotDataset)
plot_trajectories.py  human cube path vs generated cube paths (works without rendering)
contact_sheet.py      phase-by-episode PNG grid and a tiled montage mp4 (needs --render)
side_by_side.py       human clip | robot episode, aligned on grasp and release (needs --render)
run_droplet.sh        full rendered run on Linux (EGL, OSMesa fallback)
r2s/                  paths, variations, scene, planner, rollout (verifier), render, lerobot_io
tests/test_datagen.py
runs/                 local 50-episode results (attempt logs, acceptance tables, datasets, figures)
```

Nothing under `manip/task`, `manip/demo` or `third_party` is modified. Those modules
are imported, and the grasp gate still passes (4 tests).
