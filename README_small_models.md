# Branch `small/smolvla-act`: SmolVLA (0.45B) and ACT (~80M) on the cube task

All results are in simulation. Cube demos come from the one-video engine.

## What this branch does
- Trains two small policies on the **same** 1000-episode cube dataset MolmoAct2 (5.4B) used, then tests them with the same harness and scenes.
- `small/prep_small_dataset.py`: makes the `_small` copy. It uses the dataset's own stats and **removes `observation.environment_state`** (the cube pose), so neither model sees the answer. ACT reads no instruction.
- `eval/policy_io.py`: camera rename (cam_head -> camera1, cam_right -> camera2) that lerobot 0.6.1 forgets to save.
- `env/capped.py`: caps PyTorch's allocator (`set_per_process_memory_fraction`) to **emulate** a 6 GB / 8 GB laptop card.
- `eval/scoreboard.py`: builds `SCOREBOARD.md` from the eval run folders (Wilson intervals, exact McNemar, Holm). Never typed by hand.
- `eval/triptych.py`: side-by-side videos of several models on the same scene.

## How to run (inside container `sprint`, cwd `/work/wt-small`)
```
lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.n_action_steps=50 \
  --dataset.root=/work/outputs_large/datagen/sugar_v10_n1000_small --dataset.repo_id=local/sugar_v10_n1000_small \
  --dataset.video_backend=pyav --batch_size=32 --steps=30000 --save_freq=10000 --log_freq=200 --num_workers=6 \
  --wandb.enable=false --output_dir=/work/outputs_large/train/act_single_bs32
```
SmolVLA: `--policy.path=lerobot/smolvla_base`, same dataset args, plus
`--rename_map='{"observation.images.cam_head":"observation.images.camera1","observation.images.cam_right":"observation.images.camera2"}' --policy.empty_cameras=1 --batch_size=64 --steps=20000 --policy.scheduler_decay_steps=20000`.
Memory cap: `python env/capped.py --cap-gb 5.0 --result-json out.json -- lerobot-train <same args> --batch_size 8`.
Scoreboard: `python eval/scoreboard.py` (rows in `eval/scoreboard_manifest.json`). Videos: see the docstring of `eval/triptych.py`.

## Results
Training (source: `/root/work/gates/gate_p2_train.md`): ACT 30k steps x 32, loss 0.083, 3.69 GB, ~52 min; SmolVLA 20k x 64, loss 0.035, 19.43 GB, ~2 h 24 min (MI300X).

In-range, seed 3000 (source: `/root/work/SCOREBOARD.md`, 03:02 UTC; small models ids 0-99 only, a time cut):

| Model | L_full | L_task |
|---|---|---|
| MolmoAct2 5.4B (n=200) | 137/200 (61.8-74.5 %) | 158/200 (72.8-84.1 %) |
| SmolVLA 0.45B | 38/100 (29.1-47.8 %) | 75/100 (65.7-82.5 %) |
| ACT ~80M | 17/100 (10.9-25.5 %) | 69/100 (59.4-77.2 %) |

- Paired on ids 0-99 (Holm over 204 tests): MolmoAct2 > SmolVLA on L_full (66 vs 38, p_Holm 0.032); MolmoAct2 > ACT on L_full (66 vs 17). On L_task: not distinguishable at n=100.
- OOD ladder (seed 4000, n=50): SmolVLA > ACT on L_full only on colour (22 vs 2) and camera_distance (24 vs 4). Full table in SCOREBOARD.md.
- ACT mostly fails L_full by pressing into the table (arm_table_collision, penetration) after finishing the task.

Laptop memory (source: `/root/work/gates/gate_p2_memcap.md`, PASS). Largest batch that fits:
- SmolVLA: batch 16 under 7.0 GiB (6.11 GB reserved), batch 8 under 5.0 GiB (3.80 GB).
- ACT: batch 32 under 5.0 GiB (4.13 GB).
- Inference: ACT 0.32 GB, SmolVLA 0.99 GB reserved. MI300X ms per chunk: ACT ~11, SmolVLA 171-201 (not laptop numbers).
- Capped and uncapped loss curves match step by step.

Videos (local): `/root/work/outputs_large/deliverables/videos/` (triptych_inrange_s3000_id2.mp4, triptych_camera_s4000_id1.mp4, narrow_vs_wide_camera_s4000_id1.mp4).

## What you may / may not claim (honest-claims list)
May claim:
- Success rates **in simulation** on held-out random scenes, with intervals, under a verifier that needs both jaws on the object and forbids welds and teleports.
- Drops on out-of-range colours, camera poses, exposure and distractors, against the expert ceiling, for this task only.
- "Fits under an emulated 5.0 GiB PyTorch allocator cap (proxy for a 6 GB laptop card; context and driver overhead on CUDA not measured)" at the batches above.

May not claim:
- Laptop speed or training time on a 4050/4060 (only MI300X timings were measured).
- A difference between models without a significant paired test ("not distinguishable at n=...").
- "Generalises" beyond the tested perturbations; anything about a real robot; that the JEPA judge works; any drawer, C/D, Hanoi or dexterous result.
