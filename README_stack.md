# Scattered-cube stacking (`stack/v1`)

Three cubes lie anywhere in front of the arm. The policy must build one tower: large at the
bottom, medium on it, small on top. Each episode is ONE correct next move, and the
instruction is always the same:

> stack the cubes into a tower with the biggest at the bottom

The demonstrations come from a **scripted expert in simulation**, not from a human video.
Every kept episode passed a physics-only verifier (rejection sampling).

| Piece | File |
|---|---|
| Rules, next-move rule, BFS check, spawn zone, start buckets, per-episode draw | `datagen/r2s/stack.py` |
| Scripted expert for one move; episode runner; whole-tower runner | `datagen/r2s/stack_expert.py` |
| Verifier (L_task / L_full / L_strict) | `datagen/r2s/stack_verify.py` |
| Reachability sweep behind the zone | `datagen/r2s/stack_reach.py` |
| Datagen flag | `datagen/generate.py --stack` |
| Dataset check | `datagen/validate_dataset.py` (stacking branch) |
| Closed-loop eval | `eval/stack_eval.py`, run as `eval/sim_eval.py --stack move\|tower` |
| Tests | `datagen/tests/test_stack.py`, `eval/tests/test_stack_eval.py` |

It reuses the Tower-of-Hanoi code from `hanoi/v1` (`hanoi_expert.py`, `hanoi_verify.py`,
`hanoi_eval.py`); Hanoi behaviour and tests are unchanged.

## Scene

- Cubes: large 45 mm blue, medium 35 mm green, small 25 mm red, hue jitter +-0.02, masses
  40-60 / 25-40 / 15-25 g, friction 0.4-1.0. No spot marks, no bin.
- Domain randomisation: the pick-and-place one (`--variety`: table colour, lights, camera,
  servo model).
- Cube yaw: uniform over the full circle, every cube.
- Physics honesty: no welds, attaches, mocap or teleports. Cube poses are set only at the
  episode start, then the scene settles before recording. A grasp counts only if both jaws
  carry the cube.

## Spawn zone and the reachability sweep

**Zone: a polar sector around the shoulder-pan axis (x = 0.0388 m), r = 0.15-0.24 m,
azimuth +-60 deg.** The brief asked for r = 0.10-0.20 m; the arm cannot do the task there.

Why: the hardest point of the task is the last move, small cube onto medium-on-large. The
grasp site must reach ~113 mm to carry the small cube over the two-cube stack, and ~124 mm
to lift the open jaws 20 mm clear of the finished tower. Highest grasp-site height (mm) the
IK reaches, worst case over closing directions 0 / 30 / 45 deg (position <= 3 mm, finger
axis <= 5 deg, joints inside 95 % of range, no arm-arm or arm-table contact):

| finger tilt \ r (m) | 0.11 | 0.12 | 0.13 | 0.14 | 0.15 | 0.16 | 0.17 | 0.18 | 0.19 | 0.20 | 0.21 | 0.22 | 0.23 | 0.24 | 0.25 | 0.26 | 0.27 | 0.28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 deg (vertical) | 72 | 78 | 82 | 85 | 88 | 88 | 88 | 88 | 85 | 82 | 78 | 72 | 68 | 60 | 50 | 40 | 25 | 15 |
| 10 deg | 82 | 90 | 98 | 102 | 108 | 110 | 112 | 112 | 112 | 112 | 110 | 108 | 105 | 100 | 95 | 88 | 80 | 70 |
| 15 deg | 88 | 98 | 105 | 112 | 118 | 120 | 125 | 125 | 128 | 128 | 125 | 125 | 122 | 118 | 115 | 110 | 102 | 95 |
| 20 deg | 90 | 102 | 112 | 120 | 128 | 132 | 138 | 140 | 140 | 142 | 142 | 140 | 140 | 138 | 132 | 130 | 122 | 118 |

- Lowest reachable grasp site: 15 mm (vertical) to 20 mm (tilted) at every r, so table-height
  grasps work everywhere.
- Vertical fingers top out at 88 mm, below the final move. Tilting the fingertips outward
  raises the reach; 20 deg reaches 124 mm only from r = 0.145 m (seen: at r = 0.142 m the
  retreat over a finished tower failed IK by 3.6 mm), so the zone starts at 0.15 m. At
  0.24 m the 20 deg tilt still reaches 138 mm.
- Azimuth: no effect inside +-60 deg (first sweep); the test checks the zone corners.
- Reproduce: `cd datagen && python -m r2s.stack_reach --out sweep.json` (~2 min, 6 workers).
- Camera: in 300 drawn scenes, 0 of 900 cube centres fell outside the cam_head image. The far
  corner (azimuth -60 deg, r >= 0.21 m) is outside the image for ~20 % of the randomised
  camera poses, mostly at tower-top height.

## Start buckets (per episode)

| Bucket | Share | Start |
|---|---|---|
| scatter | 50 % | all three on the table. Strata: default 70 % (>= 20 mm between footprint circles), near 15 % (a tight cluster, 3-10 mm gaps), far_near 15 % (one cube at r >= 0.85 r_max, the other two 3-10 mm apart) |
| partial | 25 % | medium on large, small elsewhere (`TLT`) |
| wrong | 17 % | a stable wrong stack: small on large / on medium, medium on small (weight 1 each); large on medium / on small, medium on small on large (weight 0.5 each). Settled before recording; rejected (`start_unstable`) if it does not survive |
| noise | 8 % | a start from the buckets above, executed by the expert with joint-target noise (Ornstein-Uhlenbeck, sigma 0.012 rad, 0.25 s correlation, MimicGen-style); the noisy targets are the recorded actions; kept only if the verifier passes |

- Footprint circle = edge x 0.707 (half the diagonal). Separations are centre distance >=
  r_i + r_j + gap. Draws that break a rule are resampled; after 5000 tries the sampler
  raises, never accepts an invalid layout.
- A cube taken off a wrong stack goes to a free table spot: inside the zone (1 cm margin),
  >= 20 mm from every other footprint. The verifier accepts ANY free spot in the zone.

## The next-move rule (symbolic solver)

State = where each cube sits, `(large, medium, small)`, each `T` (table) or the cube under
it; 13 physical states, goal `TLM`. Rule, in order:

1. If a cube sits wrongly (on the wrong cube, or on a cube that is itself wrong) and nothing
   is on it, put it on the table.
2. Else, if the medium cube is not on the large one, put it there.
3. Else, put the small cube on the medium one.

Tests check the rule reaches the goal from all 12 non-goal states (<= 4 moves). A
breadth-first search gives the true minimum; the rule matches it except in `TST` and `MST`
(medium on small, large free), where it spends one extra move (medium to the table first).

## Scripted expert

The Hanoi expert (straight-down approach, per-cube grasp height, vertical lift, arc carry,
slow final lowering, retreat) with four changes:

- **Finger tilt per move.** The arm has 5 joints, so the fingers tilt only in the arm's own
  vertical plane. The expert takes the smallest tilt (0-20 deg) whose reach (table above)
  covers the move, and grasps the face pair closest to perpendicular to the arm. With
  random yaw the leftover angle tilts the cube in the jaws by asin(sin(face angle) x
  sin(tilt)); the place stage lowers by the measured lowest corner, so it lands flat.
  Tilted, off-perpendicular jaws also dip one tip, so the grasp is raised by that dip.
- **Carry clearance 12 mm** (Hanoi: 20 mm) above the tallest other cube, to stay in reach.
- **Smaller jaw opening:** cube + 25 mm (Hanoi: + 40 mm), to fit between cubes 3-10 mm apart.
- **Gentle release:** lowered to 1.5 mm above the landing surface (Hanoi: 3 mm), jaws opened
  at 0.3 rad/s until the pads clear the cube, then at full speed. A full-speed release let
  the squeezed jaw spring the cube up to 13 mm sideways.

## Verifier (`stack_verify.py`)

Physics only, every physics step, no planner phase labels (it judges the expert, a 10 Hz
replay and a policy alike).

- **L_task**: the arrangement read from the simulation equals the rule's next state; every
  stacked pair: centres <= 10 mm apart (xy), upper cube within 4 mm of its nominal height;
  a cube put on the table: inside the zone, touching no cube; all cubes still for the last
  0.5 s; gripper open and no jaw touching a cube at the end.
- **L_full**: L_task plus the moved cube carried by both jaws, lifted >= 2 cm, not dropped,
  released <= 10 mm above its landing surface; the other cubes moved < 1 cm and tipped
  < 15 deg; no arm-link contact with a cube, fling, arm-table / self collision, joint limit,
  torque saturation, speed-cap breach, penetration.
- **L_strict**: L_full plus every stacked pair within 5 mm (xy).
- Reading the arrangement: a cube must lie flat (<= 10 deg); it is ON cube X if its lowest
  point is within 6 mm of X's top and its centre is over X's top face; on the TABLE if its
  lowest point is within 6 mm of the table.
- Two stacking-only allowances (Hanoi unchanged): a "carry" shorter than 0.1 s does not
  count (a pad brushing the falling cube during the slow release); and the table may touch
  the carried cube once the jaw goal has opened >= 0.05 rad, if the cube is not sliding
  (the 1.5 mm touchdown of a table placement). Any other table contact while carried fails.

Each episode records: bucket, stratum, symbolic start state, move, expected end state,
observed end state, every pair's xy / height error, the table placement (r, azimuth, in
zone, touching), the finger tilt, the grasp raise, the reasons and their first times.

## Measured results (laptop, physics only, no rendering)

**Single moves, the datagen mix** (seed 1, attempts 0-119, `STACK_LONG=1 ... -k long`):

| Bucket / stratum | L_full | L_task | L_strict |
|---|---|---|---|
| scatter / default | 41/41 | 41/41 | 41/41 |
| scatter / near (3-10 mm gaps) | 14/14 | 14/14 | 14/14 |
| scatter / far_near | 10/10 | 10/10 | 10/10 |
| partial (`TLT`) | 36/36 | 36/36 | 36/36 |
| wrong (`TTL` 4, `TSL` 2, `TTM`, `TST`, `MTT`, `STT` 1 each) | 10/10 | 10/10 | 10/10 |
| noise | 5/9 | 9/9 | 3/9 |
| **all** | **116/120 (96.7 %)** | **120/120** | **114/120** |

- Every rejection is in the noise bucket and is `foreign_contact_while_carried` (the
  perturbed arm lets the carried cube brush the table). Those episodes are not kept.
- Finger tilt used: 0 deg 10, 5 deg 44, 7.5 deg 22, 10 deg 4, 12.5 deg 17, 15 deg 17,
  17.5 deg 2, 20 deg 4. Mean episode 25 s of simulated time (~250 frames at 10 Hz).
- No wrong-stack start collapsed in the settle (`start_unstable`: 0).

**Single moves, every bucket forced** (test `test_expert_moves_from_every_bucket`, 14 moves:
two per scatter stratum, two partial, all six wrong starts): 14/14 L_full and L_strict.

**Whole towers by the expert** (`run_stack_tower_expert`, one continuous scene, rule's next
move each time):

| Start | Solved | In the rule's move count | Every move L_full |
|---|---|---|---|
| 10 full scatters (`TTT`, 2 moves) | 10/10 | 10/10 | 10/10 |
| 10 wrong stacks (`TTL` 2, `STT` 2, `TST` 4, `TSL`, `TTM`; 3-4 moves) | 10/10 | 10/10 | 10/10 |

**Eval harness, expert mode** (seed 1000, actions replayed at 10 Hz through the policy
stepping path):

- `--stack move`, 20 scenes: L_full 20/20, L_strict 19/20; harness and native expert agree
  20/20.
- `--stack tower`, 10 scenes: solved 10/10 (5 scatters in 2 moves, 5 wrong stacks in 3-4),
  advancement 1.0; harness and native agree 10/10.

**Datagen smoke** (`generate.py --stack --variety --record-hz 10`, 8 episodes, state only):
10 attempts, 10 accepted; `validate_dataset.py` 58/58 checks.

Tests: `datagen/tests` + `eval/tests` on the laptop: 104 passed, 5 skipped (no OpenGL, and
the two long tests), 0 failed, including the Hanoi tests.

## Commands

Tests (from the repo root; laptop, CPU):

```
python -m pytest -q -s -p no:cacheprovider datagen/tests/test_stack.py eval/tests/test_stack_eval.py
STACK_LONG=1 python -m pytest -q -s -p no:cacheprovider datagen/tests/test_stack.py -k long
```

Generate on the server (`export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`), from the repo root:

```
python datagen/generate.py --spec demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json \
  --stack --variety --record-hz 10 --seed 1 --n 1000 --workers 16 \
  --render --render-quality fast --export-frame molmoact2 \
  --camera-names front=cam_head,wrist=cam_right --out <out_dir> [--dataset-dir <dataset_dir>]
python datagen/validate_dataset.py <dataset_dir or out_dir/lerobot> --lerobot
```

- Drop `--render ...` for a state-only run (works on the laptop; keep `--workers` <= 6 there).
- `meta/datagen_episodes.jsonl` has, per episode, `"stack"`: bucket, base_bucket, stratum,
  start_state, start_desc, move, moved_cube, end_state, free_spot, gaps_mm, action_noise.
  `acceptance.json` has acceptance by bucket and by start state.

Evaluate (eval seed 1000 does not overlap training seed 1):

```
# (a) one move per episode, datagen start buckets (noise-bucket scenes run noise-free)
python eval/sim_eval.py --stack move --mode expert --n 20 --workers 6 --out <dir>
python eval/sim_eval.py --stack move --mode policy --checkpoint <ckpt>/pretrained_model \
  --n 60 --videos 4 --device cuda --out <dir>
# (b) whole tower: even ids from a full scatter, odd ids from a wrong stack; <= 8 moves
python eval/sim_eval.py --stack tower --mode expert --n 10 --workers 5 --out <dir>
python eval/sim_eval.py --stack tower --mode policy --checkpoint <ckpt>/pretrained_model \
  --n 20 --videos 2 --device cuda --out <dir>
```

`summary.json` gains a `"stack"` block: for `move`, L_task / L_full / L_strict by bucket, by
bucket + stratum and by start state, end-state match, reason counts; for `tower`, solved
(with a Wilson interval), solved in the rule's move count, moves used, mean advancement
(fraction of the rule's moves done at the end), stop reasons and failure reasons, overall
and per start kind. Scenes whose start stack collapses are reported as `start_unstable` and
left out of the rates. Policy move-episodes stop 1 s after a move is done, when a cube falls,
or at `--max-sim-s` / `--move-max-s` (40 s).

## Deviations from the brief

1. Zone r = 0.15-0.24 m instead of 0.10-0.20 m (reach; see the sweep).
2. Random yaw with a 5-joint arm: when the fingers must tilt, the cube sits up to
   asin(sin 45 deg x sin tilt) tilted in the jaws (<= 14 deg at 20 deg). It lands corner-first
   and settles flat; placement accuracy is measured, not assumed.
3. The rule spends one extra move in `TST` and `MST` versus the shortest solution.
4. Expert changes listed above (carry clearance 12 mm, jaw opening cube + 25 mm, gentle
   release 1.5 mm / 0.3 rad/s, grasp raise, tilt up to 20 deg).
5. The partial bucket is the single correct partial state `TLT`; wrong stacks include
   large-on-medium, large-on-small and medium-on-small-on-large (half weight).
6. The two verifier allowances above (0.1 s minimum carry, release touchdown on the table).
7. Noise-bucket acceptance is lower (most rejections: the carried cube brushes the table);
   only passing episodes are kept.

## Not tested on the laptop

- Rendering (no OpenGL on Windows): camera images, videos and `--render` datasets are
  checked on the server.
- A trained policy: the policy-mode eval is exercised with a stand-in that replays expert
  actions.
