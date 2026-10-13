# Stage 5 gate: twin replay (KINEMATIC visual check) — PASS for cam_head geometry, PARTIAL for cam_right

**This is a kinematic replay, not a physics claim.** The recorded joint angles of real held-out episodes are written
straight into the sim arm each frame (no physics). The cube is scripted: it stays at its measured start until the
real gripper closes (grasp frame found from the gripper signal), rides rigidly with the gripper, and after the
gripper opens it is placed where it was measured in the last frame.

## Episodes (held-out, picked by rule before looking: single-cube "place in bowl" tasks from the largest session 18, plus one from session 9)
| episode | session | frames | grasp / release frame | cube fit IoU (start / end) | bowl fit IoU |
|---|---|---|---|---|---|
| 404 | 18 | 579 | 300 / 429 | 0.987 / 0.997 | 0.807 |
| 461 | 18 | 558 | 232 / 366 | 0.970 / 0.998 | 0.885 |
| 217 | 9 | 533 | 290 / none (multi-cube task) | 0.980 / 0.981 | 0.752 |

Videos (side by side, real | sim, 1280x480, 30 fps), server `/root/work/outputs_large/friendrig/twin/` and Drive
`gdrive:so101-sprint/hanoi_oct13/friendrig/twin/`:
`twin_ep{404,461,217}_cam_head_composite.mp4`, `twin_ep{404,461,217}_cam_right.mp4`, and the plain-MuJoCo A/B
`twin_ep461_cam_head_plain.mp4`.

## What the videos show (looked at frames 150/260/420 of ep 461 and 320 of ep 404)
- cam_head: the sim arm lies on the real arm (upper arm, forearm window, wrist motor, gripper), consistent with the
  stage-1 held-out IoU 0.69-0.71. The cube starts in the right place and ends in the bowl.
- The sim bowl reads as a deeper, straighter "bucket" than the real paper bowl, and its panels show; position is
  close (centre within ~1-2 cm by eye).
- cam_right (wrist): jaws are at the bottom as in the real view, but the sim view is **visibly rotated (roughly
  30-40 deg) and shifted**: the cube and bowl appear further left/up than in the real frame. The stage-2 fit only
  used the jaw tips at the image bottom, which pin the camera poorly in roll about its own axis. The table texture
  is the real orthophoto where cam_head saw it and a tiled crop elsewhere; the robot's shadow from the single key
  light is harder than in the real frame.

## Command (container `sprint`, cwd /work/wt-friendrig, via run.sh; ~215 s per episode, 1 process)
```
python -m friendrig.replay --profile /work/outputs_large/friendrig/camfit/friend_bowl_profile.json \
    --episodes 404 461 217 --out /work/outputs_large/friendrig/twin
python -m friendrig.replay --profile ... --episodes 461 --plain --out /work/outputs_large/friendrig/twin
```
Log `/root/work/logs/friendrig/replay.log`; summary `twin_summary_composite.json`.
