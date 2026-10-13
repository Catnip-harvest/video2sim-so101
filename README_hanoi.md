# Branch `hanoi/v1`: Tower of Hanoi on the table, one move per episode

All results are in simulation. The demonstrations come from a **scripted expert** (our
planner), not from a human video.

## The task

- **Three spots** on the table: A (left), B (middle), C (right), as seen in the front camera
  `cam_head`. They lie on an arc 0.19 m from the shoulder-pan axis at -37, 0 and +37 deg.
  Nominal centres (robot base frame, m): **A (0.1905, -0.1143), B (0.2288, 0.0000),
  C (0.1905, 0.1143)**; neighbours are 12.1 cm apart. Each spot moves inside a 1 cm disc per
  episode, so neighbours stay at least 10.1 cm apart (>= 9 cm is also enforced). A spot is a
  flat marker (dark ring, light centre) with no collision: contype = conaffinity = 0.
  There is no tray.
- **Three cubes**: large 45 mm (blue), medium 35 mm (green), small 25 mm (red). Each colour
  gets a small hue and brightness jitter per episode. Masses are 40-60, 25-40 and 15-25 g, and
  friction is 0.4-1.0. A cube sits within 3 mm of its spot (or of the cube under it), with its
  faces within +-10 deg of square to the arm.
- **Everything else is unchanged**: table, lights, front camera, servos, expert tempo and
  pauses all come from `variations.py`.
- **State**: which spot each cube is on, written as three letters for (large, medium, small).
  - "AAA" means the whole tower is on A. "ABA" means the large and small cubes are on A (small
    on top) and the medium cube is alone on B.
  - A legal stack always has the larger cube below, so all 27 states are legal. The goal is
    "CCC".
- **Solver**: breadth-first search from the goal. From "AAA" it needs 7 moves. From every
  other state the optimal next move is unique, and it always moves a top cube.
- **One episode is one move.**
  - The start state is uniform over the 26 non-goal states. The cubes start stacked and
    settle for 0.6 s before recording begins.
  - The expert makes the solver's next move: grasp the top cube with both jaws, lift it
    vertically, carry it, place it on the table or on top of the larger cube, release,
    retreat and return to the ready pose. Then 0.5 s of rest and a 1 s settled tail.
  - The instruction is always `make the next move to stack the tower on the right spot`.
  - Starting uniformly means the small cube moves in 18 of the 26 start states, the medium cube
    in 6 and the large cube in 2.
- **Physics honesty** follows the pick-and-place rules.
  - The expert only writes servo targets.
  - Cube poses are written once, before the settle.
  - There are no welds, attachments, mocap bodies, equality constraints or teleports.

## The verifier (`datagen/r2s/hanoi_verify.py`)

The verifier uses physics only and needs no planner phase labels, so the same code scores the
expert, a 10 Hz replay and a policy. It watches all three cubes on every physics step.

- **L_task**: the final arrangement is the solver's expected next state.
  - Every cube's centre is within 2 cm (xy) of its spot centre and within 6 mm (z) of its
    stack height.
  - All cubes are at rest (< 2 cm/s).
  - No jaw has touched a cube in the last 0.5 s.
  - Failure reasons: `cube_off_spot`, `wrong_arrangement`, `wrong_stack_height`,
    `not_resting`, `not_released`, `unstable`.
- **L_full** is L_task plus every check below:
  - **Carry**: the moved cube was carried by both jaws and rose at least 2 cm. It was not
    dropped, and it was not released more than 10 mm above what it landed on.
  - **Other cubes**: no other cube moved more than 1 cm or tipped more than 15 deg.
  - **Legality**: a larger cube never rested on a smaller one (`illegal_stack`).
  - **Physical honesty**, the same clauses as the pick-and-place phase-free verifier:
    - no foreign contact while carried;
    - no arm link (other than the jaws) touching any cube;
    - no fling;
    - no arm-table or self collision;
    - no joint-limit violation or torque saturation;
    - no speed-cap violation;
    - no penetration deeper than 2 mm for 3 steps;
    - two style checks: tracking error and object disturbed before the grasp.
- Every episode records which checks failed and when, plus per-cube movement, tilt, lift,
  carries and contact counts.

## Design decisions in the expert (each one was measured, not guessed)

| Problem seen | Fix |
|---|---|
| With vertical fingers the grasp site reaches only about 85 mm up at r = 0.19 m. The top cube of a full stack needs about 97 mm, and the carry heights more. | Tilt the fingertips 15 deg outward, which raises the reach to about 125 mm. The jaws close **tangentially**, on the face pair closest to tangential, so the pads stay vertical. |
| A radial tilt with radial closing turned the cube by 14 deg during the squeeze. | Same tangential closing. Cube faces start within +-10 deg of square to the arm, so the face pair is at most 10 deg off. Measured tilt in the hand is now 0-4 deg. |
| 8 g wooden small cubes sank 2.1-2.6 mm into the jaw tips, a penetration failure. | Keep cube masses in the band the pick-and-place grasp gate covered (15-41 g). MuJoCo's soft contacts sink deeper the lighter the body. |
| Grasping the large cube at its centre or 5 mm above: the moving jaw's upper part bit its top edge, failing 8/8 and 4/8. | Grasp height above the centre: large 12 mm, medium 8 mm, small 5 mm. Large-cube failures dropped to 0/8. |
| The pick-and-place home puts the jaw tips 97 mm up, right above spot B, inside a 105 mm stack. | A separate Hanoi ready pose `HANOI_HOME_Q = [0, -0.9, 0.15, 1.45, -1.55]`. Its lowest jaw point is at least 0.12 m up even with the +-0.15 rad start jitter. |
| Stacking needs a gentle landing. | Carry 20 mm above the tallest other cube. Lower the last 2 cm at 12 mm/s. Open 3 mm above the landing surface, using the cube's measured lowest corner. |

## Results (Windows laptop, physics only, no rendering)

| Check | Result |
|---|---|
| Solver | 27 states; 7 moves from AAA; all 26 non-goal states have a unique optimal move of the top cube |
| Stacks settle | AAA, BBB, CCB and ABA rest within 2 mm of their stack heights and drift < 2 mm over 2 s |
| Camera framing (geometric) | In 300 draws, all spot corners at table level and at 105 mm project inside the central 95 % of `cam_head` |
| Expert, single moves, test set (seed 1, ids 0-19) | **L_task 20/20, L_full 20/20** (moved cube: small 10, medium 10) |
| Expert, single moves, more draws | seed 1, ids 0-99: **100/100 L_full**; seed 0, ids 0-39: **40/40** |
| `generate.py --hanoi`, 12 episodes, 10 Hz, molmoact2 frame | 12/12 accepted; `validate_dataset.py` 58/58 checks (without `--variety`: 20/20 accepted) |
| Harness, `--hanoi move --mode expert` (seed 1000, ids 0-49, 10 Hz replay) | **50/50 L_full** (Wilson 95 % 92.9-100 %); native and harness verdicts agree 50/50; 0 rate-limited frames; moved cube: small 39, medium 8, large 3 |
| Full puzzle, expert, native | solved AAA -> CCC in **7 moves**, 0 illegal states, all 7 moves L_full |
| Full puzzle, expert, through the harness (`--hanoi puzzle --mode expert`, seed 1000, ids 0-4) | **5/5 solved in 7 moves**, 0 illegal states, every move-episode L_full; about 60 s wall per puzzle |
| Speed | about 5.5 CPU-s per single-move episode (about 20 s simulated) |

## How to run

The tests run on the laptop, from the repo root:

```
python -m pytest -q -s -p no:cacheprovider datagen/tests/test_hanoi.py eval/tests/test_hanoi_eval.py
```

### Generate N episodes

On the server (`export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`), from the repo root:

```
python datagen/generate.py --spec demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json \
  --hanoi --variety --record-hz 10 --seed 1 --n 600 --workers 16 \
  --render --render-quality fast --export-frame molmoact2 \
  --camera-names front=cam_head,wrist=cam_right --out <out_dir> [--dataset-dir <dataset_dir>]
python datagen/validate_dataset.py <dataset_dir or out_dir/lerobot> --lerobot
```

- Drop `--render ...` for a state-only run, which also works on the laptop.
- Only L_full episodes are kept (rejection sampling).
- Each episode's start state, move, expected end state, moved cube and spot centres are in
  `meta/datagen_episodes.jsonl` under `"hanoi"`, and also in `variation.extras.hanoi`.

### Evaluate

Eval seed 1000 does not overlap with training seed 1.

```
# (a) one move per episode from random legal start states
python eval/sim_eval.py --hanoi move --mode expert --n 20 --workers 8 --out <dir>
python eval/sim_eval.py --hanoi move --mode policy --checkpoint <ckpt>/pretrained_model \
  --n 50 --videos 4 --device cuda --out <dir>
# (b) the whole puzzle from AAA: up to --max-moves 12 move-episodes of at most --move-max-s 40 each
python eval/sim_eval.py --hanoi puzzle --mode expert --n 5 --workers 5 --out <dir>
python eval/sim_eval.py --hanoi puzzle --mode policy --checkpoint <ckpt>/pretrained_model \
  --n 10 --videos 2 --device cuda --out <dir>
```

`summary.json` gains a `"hanoi"` block:

- **Move**: L_task / L_full by moved cube, and whether the end state matches the expected one.
- **Puzzle**: solved count with a Wilson interval, solved in 7, moves used per run, and illegal
  states reached.

On the laptop add `--obs-codec none` to the expert-mode commands; they need no GL.

## Not tested here (needs OpenGL; run on the Linux server)

- Rendering anything: dataset videos (`--render`), the render test
  `datagen/tests/test_hanoi.py::test_front_camera_shows_spots_and_cube_colours` (cube colours
  and spot marks visible in `cam_head`), and
  `eval/tests/test_hanoi_eval.py::test_policy_mode_with_real_renders`.
- Policy mode with a real checkpoint (no GPU here). Policy mode was exercised with a stand-in
  runner that plays the expert's actions, with blank camera frames.
- Loading a Hanoi dataset with `LeRobotDataset` (`validate_dataset.py --lerobot`).

## What you may / may not claim

You may claim:

- Expert success rates **in simulation**, under a verifier that needs both jaws on the cube
  and forbids welds, attachments and teleports.
- Policy per-move and full-puzzle results **in simulation**, with their intervals.

You may not claim:

- Anything about a real robot.
- That Hanoi was learned from a human video.
- That a policy "plans" the puzzle. Every training episode is one move, and the solver's
  choice is in the data, not in the instruction.
- Generalisation to other cube sizes, spot layouts or more than three cubes. None was tested.
