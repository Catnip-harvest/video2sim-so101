# Branch `eval/harness`: closed-loop sim eval + phase-free verifier

All results are in simulation.

## What this branch does
- `eval/sim_eval.py` runs a trained policy in the MuJoCo scene, one 10 Hz step at a time, and judges every episode. Modes: `policy`, `expert` (scripted planner, CPU), `dataset-replay`.
- `datagen/r2s/verify_policy.py` is the **phase-free verifier**: it needs no phase labels, so it can judge a policy.
  - **Carry** = both jaws touch the cube, cube > 5 mm above its rest height, touching neither table nor bin floor.
  - **L_task** = cube carried with both jaws, carry ended over the bin, cube resting in the bin, no jaw touching it in the last 0.5 s.
  - **L_full** = L_task plus physics-honesty clauses (no drop/slip, release <= 10 mm high, no arm-table/bin/self collision, no penetration > 2 mm for >= 3 steps, joint limits, torque, lift >= 2 cm) and style clauses.
- `eval/policy_io.py` loads a lerobot checkpoint and calls `select_action` through its own pre/post processors.
- Each image gets one h264 encode/decode, like the training videos.
- Outputs: `episodes.jsonl` (per episode: pass, L_task, L_full, reasons), `summary.json` (Wilson 95 % intervals), `actions/*.npz` (replayable on CPU), optional mp4s.
- `eval/bplus_run.sh` etc.: filtered-BC pipeline (scripted and CPU-tested, **not run**: dropped by the owner after multi-replica inference hung).

## How to run (inside container `sprint`, cwd `/work/wt-eval`, `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`)
```
python -m pytest -q datagen/tests/test_verify_policy.py eval/tests/test_sim_eval.py      # 11 tests
python eval/sim_eval.py --mode expert --seed 1000 --n 60 --workers 3 --out /work/outputs_large/eval/expert_n60
nice -n 5 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval/sim_eval.py --mode policy \
  --checkpoint /work/outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model \
  --seed 3000 --n 200 --videos 8 --workers 1 --device cuda --out /work/outputs_large/eval/policy_final_n200_seed3000
```
**Run ONE policy process at a time** (`--workers 1`). Multi-replica inference hung on this box (K=8 pool after 7 episodes, K=4 at 0).

## Results
Harness checks (source: `/root/work/gates/gate_eval_harness.md`, PASS):

| Check | Result |
|---|---|
| Expert through the harness, seed 1000, n=60 | L_full 58/60 (96.7 %, CI 89-99 %), L_task 60/60 |
| Dataset replay, train episodes 0-4 | 5/5 |
| Phase-free vs expert verifier agree | 59/60 expert, 20/20 zero-order-hold drops, 10/10 jaw-never-closes |
| Policy smoke, MolmoAct2 ckpt 001400, n=20 | L_full 10/20 (50 %, CI 30-70 %), L_task 15/20 (75 %, CI 53-89 %) |
| Rendered frames vs training frames (same codec) | PSNR 45-99 dB (99 = bit-identical) |

Final MolmoAct2 (ckpt 023400), 200 held-out in-range scenes, seed 3000 (source: `/root/work/gates/gate_p1_eval200.md`):
- **L_full 137/200 = 68.5 % (Wilson 95 % 61.8-74.5 %)**
- **L_task 158/200 = 79.0 % (72.8-84.1 %)**
- Most common first failure: never_carried 28, arm_table_collision 15, self_collision 8.
- 301 episodes/h, median chunk inference 360 ms on the MI300X (not a laptop number).

Known limits: single-frame h264 in policy mode (37-39 dB, vs 45-99 for a sequence encode); actions ramp ~67 ms behind the expert; episodes stop 1 s after task done, so later collisions are not judged.

## What you may / may not claim (honest-claims list)
May claim:
- Success rates **in simulation** on held-out random scenes, with intervals, under a verifier that needs both jaws on the object and forbids welds and teleports.
- How much success drops on out-of-range colours, camera poses and distractors (branch `eval/ood-ladder`), for this task only.
- Cube demos come from the one-video engine.

May not claim:
- Anything about a real robot (G0 overlay INCONCLUSIVE; sim-to-real gap FD 1329 vs floor 15).
- Laptop speed or training time on a 4050/4060.
- "Generalises" beyond the tested perturbations, or from in-range seeds alone.
- That the JEPA judge works (0.784 vs 0.802, FAIL).
- A difference between models without a significant paired test (exact McNemar, Holm-corrected).
- Any B+ (filtered-BC) result: it was not run. Any drawer, C/D, Hanoi or dexterous result.
