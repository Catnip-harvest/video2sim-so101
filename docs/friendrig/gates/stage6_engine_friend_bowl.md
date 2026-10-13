# Stage 6 gate: engine output in the friend scene — PASS (expert + verifier), rendered 1000-episode dataset RUNNING

## Expert on 20 episodes (no render; the bar was >= 16/20)
| run | seed | first 20 attempts accepted | rejections |
|---|---|---|---|
| v1 (first try) | 0 | **15 / 20** | ik_unreachable 3 (cubes at r ~0.29 m), not_resting 2 (cube jittering on a cylinder floor) |
| v2 (after two fixes chosen from v1's failures, fresh seed) | 1 | **20 / 20** | none |
Fixes between v1 and v2: (1) bowl contact floor = box (box-box contacts settle) + visual disc; (2) cube start capped
at 0.28 m from the pan axis (see stage 4: this drops part of the real layout distribution). Nothing in the verifier
changed.

## 1000 episodes, physics + verifier (no render)
**1000 accepted of 1021 attempts = 97.9 %**, wall **533 s** with 7 workers (3667 CPU-s, ~3.6 s per attempt).
Rejections: arm_bin_collision 20 (arm touched the bowl), ik_unreachable 1. Accepted episodes: mean 749 frames at
30 Hz (25 s; the real episodes average 626 frames), in-hand slip median 1.1 mm (p90 1.4), final cube offset from
the bowl centre median 3.9 mm (p90 7.3). Physics rules unchanged: position servos only, no welds / attaches /
mocap / teleports (asserted: model.neq == 0 and nmocap == 0), both jaws must carry the cube for every carried step,
the cube must end at rest on the bowl floor and released.

## Rendered dataset (running at the time of writing)
Same command with rendering: `/root/work/outputs_large/friendrig/gen_friend_bowl_n1000_render/`, 7 workers,
tmux `fr_gen`, log `/root/work/logs/friendrig/gen1000_render.log`. Measured render cost ~80-125 s per episode
(640x480, two cameras + one segmentation pass for the composite, OSMesa "fast" quality); first 20 attempts in
~6.5 min; **ETA ~5.5 h** from 13:46 UTC. When it ends, generate.py writes the LeRobot dataset to `<out>/lerobot`.
A 3-episode export test of the identical pipeline (`gen_export_test/lerobot`) checks the format:

| | sim export | friend dataset (friend_molmoact2) |
|---|---|---|
| fps | 30 | 30 |
| joint frame | MolmoAct2 (`--export-frame molmoact2`) | MolmoAct2 (converted) |
| state/action names | shoulder_pan.pos ... gripper.pos | same |
| cameras | observation.images.cam_head, observation.images.cam_right, 480x640x3, av1 | same |
| extra feature | observation.environment_state (cube pose, 7) | none |
| robot_type | so101_follower | so_follower |
Sim<->friend joint conversion uses the FITTED calibration (`friend_arm_calib.json`: stage-1 offsets incl. wrist_roll
-189.6 deg total, stage-2 gripper map -9.5..107.4 deg).

**Caveat found in the export test:** exported wrist_roll spans ~104-274 deg (p5-p95) while the real data spans
45-106 deg. The expert picks the wrist roll from the cube's yaw and often lands half a turn away from where the
human operated; values above ~180 may be outside the real arm's calibrated range. Constraining the expert's roll
choice to the real range (the cube's 90 deg symmetry allows it) is the obvious next fix; not done.

## Commands
```
# 20-episode checks (container sprint, cwd /work/wt-friendrig, via run.sh)
python datagen/generate.py --spec task/specs/friend_bowl_task_spec.json --profile /work/outputs_large/friendrig/camfit/friend_bowl_profile.json \
    --n 20 --max-attempts 20 --workers 7 --seed 0 --out /work/outputs_large/friendrig/gen_smoke20    --no-render --no-dataset
python datagen/generate.py ... --seed 1 --out /work/outputs_large/friendrig/gen_smoke20_v2 --no-render --no-dataset
# 1000, physics only
python datagen/generate.py ... --n 1000 --workers 7 --seed 0 --out /work/outputs_large/friendrig/gen_friend_bowl_n1000 --no-render --no-dataset
# 1000, rendered + exported (wrapper /root/work/outputs_large/friendrig/gen.sh adds:)
#   --render --size 640x480 --render-quality fast --codec av1 --export-frame molmoact2
#   --camera-names front=cam_head,wrist=cam_right --arm-calib .../friend_arm_calib.json --task "Place the green cube in the white bowl."
gen.sh gen1000_render /work/outputs_large/friendrig/gen_friend_bowl_n1000_render 1000 7
```
"First 20 attempts" counts attempts 0-19 in `attempts.jsonl` (generate.py logs a few more because it works ahead).
Logs: `/root/work/logs/friendrig/{gen_smoke,gen1000,gen1000_render,gen_export_test}.log`.
