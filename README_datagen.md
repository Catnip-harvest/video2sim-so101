# Branch `datagen/10hz-variety`: the cube data engine (one video -> verified sim demos)

All results are in simulation. Cube demos come from the one-video engine: a task spec extracted from one human video (`demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json`) drives a scripted SO-101 expert in MuJoCo, and a physics verifier keeps only honest episodes.

## What this branch does
- `generate.py --record-hz 10`: every dataset is recorded at 10 Hz, so action chunks always span the same time.
- `--variety`: per-episode expert style (side offset, approach tilt, pause lengths, a hover hesitation in 35 % of episodes, start pose +-0.15 rad). Separate random stream, so other draws stay the same.
- `--keep-rejects`: keeps failed attempts (with their reason) for analysis.
- `--render-quality full|shadow|fast`: fast = 421 images/s vs 45 for full.
- `--export-frame sim|lerobot|molmoact2`: writes joints in the policy's own frame (units module from `calib/so101-units`).
- `use_pretrained_norm.py`: fine-tune MolmoAct2 with its checkpoint's own q01/q99 stats.
- `env/train_molmoact2.sh`: the MolmoAct2 training launcher.
- The verifier rejects an episode unless both jaws carry the cube; no welds, attaches or teleports.

## How to run (inside container `sprint`, cwd `/work/wt-datagen/datagen`, `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`)
```
nice -n 10 python generate.py --spec /work/wt-datagen/demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json \
  --n 1000 --seed 1 --render --render-quality fast --record-hz 10 --variety --keep-rejects \
  --size 256x256 --codec h264 --workers 15 --out /work/outputs_large/datagen/sugar_v10_n1000
# export for MolmoAct2: same + --finalize-only --export-frame molmoact2 --camera-names front=cam_head,wrist=cam_right \
#   --dataset-dir /work/outputs_large/datagen/sugar_v10_n1000_molmoact2
python validate_dataset.py /work/outputs_large/datagen/sugar_v10_n1000_molmoact2 --lerobot
python -m pytest -q tests/test_datagen.py                    # 7 tests
DATA=/work/outputs_large/datagen/sugar_v10_n1000_molmoact2 STEPS=23400 BS=16 NAME=molmoact2_sim_v1 bash /work/wt-datagen/env/train_molmoact2.sh
```
(The cube dataset was made with exactly these settings: `outputs_large/datagen/sugar_v10_n1000/run.json`.)

## Results
- Render throughput (source: `/root/work/gates/gate1_render.md`, PASS): 10 Hz + fast render + variety, 16 workers: **0.92 attempts/s, 96.8 % accepted** on a 60-episode bench (pass marks 0.5 eps/s, 70 %).
- Cube dataset `sugar_v10_n1000` (source: `outputs_large/datagen/sugar_v10_n1000/acceptance.json`, also `gate_p25_data.md`): **1000 accepted / 1055 attempts (94.8 %)**; rejections arm_table_collision 52, unstable 3; 187,401 frames at 10 Hz.
- MolmoAct2 smoke (source: `gate1_smoke_molmoact2.md`, PASS): bs16 1.23 s/step, 146 GB GPU peak (MI300X).
- MolmoAct2 trained on this data, 200 held-out in-range scenes (source: `gate_p1_eval200.md`): **L_full 137/200 = 68.5 % (Wilson 95 % 61.8-74.5 %)**, L_task 158/200 = 79.0 % (72.8-84.1 %).
- SmolVLA and ACT trained on the same data: see `small/smolvla-act` and `SCOREBOARD.md`.
- The friend's real data was **not** used for training: the calibration check (G0) was inconclusive.

## What you may / may not claim (honest-claims list)
May claim:
- Cube demos come from the one-video engine, kept only when a verifier sees both jaws carry the cube, with no welds or teleports.
- Success rates **in simulation** on held-out random scenes, with intervals.
- Drops on out-of-range colours, camera poses and distractors (branch `eval/ood-ladder`), for this task only.

May not claim:
- Anything about a real robot (G0 overlay INCONCLUSIVE; sim-to-real gap FD 1329 vs floor 15).
- "Generalises" beyond the tested perturbations, or from in-range seeds alone.
- Laptop speed or training time on a 4050/4060.
- A difference between models without a significant paired test; that the JEPA judge works; any drawer, C/D, Hanoi or dexterous result.
