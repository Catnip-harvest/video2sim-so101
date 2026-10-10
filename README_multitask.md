# Branch `multitask/scene-v1`: named cube among three (P3-A-lite)

All results are in simulation. These demos come from a **scripted expert** (our planner), not from a human video.

## What this branch does
- Puts **three cubes** on the table. The instruction names one colour ("put the red cube in the tray"). Only that cube may go in the tray.
- The two extra cubes are ordinary free bodies (`datagen/r2s/free_bodies.py`): no weld, no mocap, no teleport.
- Colours: red, green, blue, yellow; no two alike. Training targets are red/green/blue only. **Yellow is never a training target** (it is the held-out test).
- The named cube takes a random slot, so position does not reveal which cube is named.
- The verifier rejects: a distractor moved >= 1 mm (`distractor_disturbed`), the wrong cube in the tray (`wrong_object_in_tray`), the wrong cube moved (`wrong_object_moved`).
- Eval adds `sim_eval.py --multitask {train,paraphrase,swap,yellow}` and named-cube scoring (named / other / none / multiple).
- Question it tests: does a model that reads language (SmolVLA) pick the named cube more often than one that cannot read (ACT, no language; expected ~33 % by chance)?

## How to run (inside container `sprint`, `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`)
Expert gate (CPU, cwd `/work/wt-multi/datagen`):
```
nice -n 19 python generate.py --spec /work/wt-multi/demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json \
  --multitask --variety --record-hz 10 --seed 2 --n 600 --max-attempts 200 --workers 6 \
  --no-render --no-dataset --out /work/outputs_large/datagen/multitask_A_gate200_v2
```
Dataset (600 episodes): same command with `--render --render-quality fast --export-frame molmoact2 --camera-names front=cam_head,wrist=cam_right --dataset-dir <dir>` and no `--max-attempts`. Then `small/prep_small_dataset.py` makes the `_small` copy, which **removes `observation.environment_state`** (it is the target cube's pose and would leak the answer).

Tests (cwd `/work/wt-multi`):
```
python -m pytest -q -p no:cacheprovider datagen/tests/test_multitask.py eval/tests/test_multitask_eval.py   # 13 passed
python -m pytest -q task/tests/test_grasp_physics.py                                                         # 4 passed
```
Policy eval (GPU, one at a time through the queue):
```
python eval/sim_eval.py --mode policy --checkpoint <ckpt>/pretrained_model --multitask swap \
  --seed 5000 --start 0 --n 20 --videos 4 --workers 1 --device cuda --out <outdir>
```
Training commands (lerobot 0.6.1): see `/root/work/queue/bin/p3_train.sh` (SmolVLA-multi 15k steps bs64; ACT-multi-no-language 30k steps bs32).

## Results (source: `/root/work/gates/gate_p3A_expert.md`)
| Check | Result |
|---|---|
| Expert acceptance, seed 2, 200 attempts | **169/200 = 84.5 %** (Wilson 95 % 78.8-88.9 %) |
| Per target (accepted/attempts) | red 54/67, green 60/67, blue 55/66 |
| Distractor movement, all 200 attempts | max **0.000 mm**; arm-distractor contacts 0 |
| No-weld physics test | 4/4 passed (neq=0, nmocap=0) |
| Wrong-object negative controls | rejected (`wrong_object_moved`, `wrong_object_in_tray`) |
| All three cubes visible in cam_head | 198/200 (ids 137, 167: one cube hidden by the arm) |
| Instruction matches target | 20/20 sampled episodes |
| Leakage eval seed 5000 vs training | 0 matches |
| Dataset | 600 episodes, 145,629 frames at 10 Hz, 200 per target colour |

**Policy results:** not available when this README was written (03:15 UTC Sat 10 Oct). Both trainings started at 03:03 UTC. The SmolVLA-multi and ACT-multi rows in `/root/work/SCOREBOARD.md` were still `-`. Read the final numbers from SCOREBOARD.md, not from here.

## What you may / may not claim (honest-claims list)
May claim (if the numbers support it):
- Success rates **in simulation** on held-out random scenes, with intervals, under a verifier that needs both jaws on the object and forbids welds and teleports.
- The named-cube rate of SmolVLA-multi vs ACT-multi-no-language, **only if** an exact McNemar test (Holm-corrected) supports it.
- P3-A demos come from a scripted expert.

May not claim:
- Anything about a real robot.
- That P3-A was learned from a human video.
- A difference between models without a significant paired test ("not distinguishable at n=..." otherwise).
- "Generalises" beyond the tested perturbations (paraphrase, swap, yellow).
- Laptop speed or training time on a 4050/4060.
- Any drawer, C/D, Hanoi or dexterous result. None was run.
