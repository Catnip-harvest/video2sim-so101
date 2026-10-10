# Branch `eval/ood-ladder`: test scenes outside the training ranges

All results are in simulation.

## What this branch does
- Adds `--ood <rung>` to `eval/sim_eval.py`. Each rung changes **one factor** outside the training range, on the same 50 scenes (seed 4000, ids 0-49). Everything else (cube pose, mass, friction, start pose) stays identical to the `none` run, so each rung can be compared pair by pair.
- Rungs (full definitions and training ranges: `docs/OOD_LADDER.md`):
  - `camera`: front camera height +-12-16 cm and pitch +-12-15 deg, FOV 51 deg;
  - single-axis sub-rungs: `camera_height`, `camera_pitch`, `camera_yaw` (+-10-15 deg), `camera_distance` (+-8-12 cm);
  - `exposure`: brightness x0.5/x1.6, or contrast x0.6/x1.4, or white-balance gain 0.8/1.2;
  - `colour`: magenta cube, near-black bin, saturated green table, lights dimmer or brighter than any training scene;
  - `combo`: exactly the camera draws plus exactly the colour draws;
  - `distractor`: 1-2 extra cylinders or spheres (20-35 mm) on the table; putting one in the bin fails the episode (`wrong_object_in_bin`).
- The camera is fixed within an episode (no shake or drift).
- `datagen/r2s/free_bodies.py`: extra objects as plain free bodies (no weld, no mocap).

## How to run (inside container `sprint`, cwd `/work/wt-ood`, `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`)
```
python -m pytest -q -p no:cacheprovider eval/tests/test_ood.py datagen/tests/test_verify_policy.py \
  datagen/tests/test_free_bodies.py datagen/tests/test_datagen.py          # 10 + 6 + 4 + 7 passed
nice -n 19 python eval/sim_eval.py --mode expert --ood camera --seed 4000 --n 20 --workers 2 \
  --out /work/outputs_large/eval/expert_s4000_camera                       # expert ceiling, CPU
python eval/ood_sheet.py --ood camera --codec --out /work/outputs_large/eval/ood_sheets/camera.png
python eval/sim_eval.py --mode policy --checkpoint <ckpt>/pretrained_model --ood camera \
  --seed 4000 --n 50 --videos 4 --workers 1 --device cuda --out <outdir>   # one policy process at a time
```

## Results
Code gate (source: `/root/work/gates/gate_p1_ood_code.md`, PASS): all rungs ready; **expert ceiling 17/20 L_full, 19/20 L_task on every rung** (the expert plans from state, so images do not affect it). Cube and gripper visible in frame 0: 50/50 on camera, 48/50 on camera_pitch.

MolmoAct2 (ckpt 023400), seed 4000, n=50 (source: `/root/work/SCOREBOARD.md`, generated 03:02 UTC 10 Oct):

| Rung | L_full | L_task |
|---|---|---|
| none | 33/50 (52.2-77.6 %) | 39/50 (64.8-87.2 %) |
| camera | 35/50 (56.2-80.9 %) | 39/50 (64.8-87.2 %) |
| exposure | 28/50 (42.3-68.8 %) | 41/50 (69.2-90.2 %) |
| colour | 30/50 (46.2-72.4 %) | 38/50 (62.6-85.7 %) |
| combo | 27/50 (40.4-67.0 %) | 39/50 (64.8-87.2 %) |
| distractor | 32/50 (50.1-75.9 %) | 39/50 (64.8-87.2 %) |
| camera_yaw | 29/50 (44.2-70.6 %) | 35/50 (56.2-80.9 %) |
| camera_distance | 29/50 (44.2-70.6 %) | 37/50 (60.4-84.1 %) |

- **No MolmoAct2 drop vs its own `none` run is significant** (exact McNemar, Holm over the scoreboard's 204 tests). Write "not distinguishable at n=50".
- SmolVLA, ACT and ACT-wide rows for the same rungs: `SCOREBOARD.md` and the `small/smolvla-act` and `datagen/wide-camera` READMEs.

## What you may / may not claim (honest-claims list)
May claim:
- How much each model's success **drops** on out-of-range colours, camera poses, exposure and non-cube distractors, measured on the same scenes and against the expert ceiling. Evidence for *this* task only.
- Success rates **in simulation** on held-out random scenes, with intervals, under a verifier that needs both jaws on the object and forbids welds and teleports.

May not claim:
- "Generalises" beyond these tested perturbations, or from in-range seeds alone.
- That a non-significant drop means "no drop": n=50 per rung cannot see small drops.
- A difference between models without a significant paired test.
- Anything about a real robot (G0 overlay INCONCLUSIVE; sim-to-real gap FD 1329 vs floor 15).
- Laptop speed; that the JEPA judge works; any drawer, C/D, Hanoi or dexterous result.
