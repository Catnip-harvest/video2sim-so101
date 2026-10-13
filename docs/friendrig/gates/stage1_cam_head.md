# Stage 1 gate: overhead camera (cam_head) fit — PARTIAL

**Verdict: PARTIAL.** Held-out arm-silhouette IoU **mean 0.686, median 0.708** (n = 500 frames from all
50 held-out episodes). That is well above October's 0.49 and above the 0.6 "keep going" line, but
below the 0.8 bar. 25 % of held-out frames reach >= 0.8.

## Numbers (all held-out unless marked; MuJoCo OSMesa render, full Menagerie meshes, 640x480, hard IoU)
| Variant | mean IoU | median | p10 | n |
|---|---|---|---|---|
| **Session cameras + fitted joint offsets (the result)** | **0.686** | **0.708** | 0.481 | 500 |
| same, cable band 21 px instead of the pre-chosen 15 px | 0.706 | 0.727 | 0.524 | 500 |
| same, cable pixels counted as arm ("raw", no ignore band) | 0.621 | – | – | 500 |
| same cameras, default joint offsets (wrist_roll -90) | 0.522 | 0.544 | 0.249 | 500 |
| one global camera, default offsets (what October assumed: one fixed camera) | 0.476 | 0.486 | 0.264 | 500 |
| TRAIN frames, the result | 0.713 | 0.736 | 0.529 | 4740 |
| October G0 (per-episode camera, 3 frames, default offsets) | 0.49 | | | 10 |

## Why October failed (and what changed)
1. **cam_head is not fixed over the dataset.** First frames show the table grain and the parked arm moving
   between recording sessions. There are two clear camera set-ups (yaw ~0.31-0.37 rad for episodes 0-295,
   ~0.17-0.19 rad for 297-523) plus smaller shifts. One camera for everything scores 0.476.
2. **The wrist_roll mapping is off by about 180 deg, not 90.** A scan of the wrist_roll offset (session cameras
   frozen) peaks at -180 total (0.592 soft IoU vs 0.514 at the default -90); the joint fit lands at
   -189.6 deg total. October's per-episode cameras could not absorb this.
3. October fitted each camera on 3 frames of one episode; here each session camera uses up to 120 frames
   from many train episodes (robust trimmed mean, worst 10 % dropped).
4. The wrist camera's USB cable is black and ~6-20 px wide; it is now an IGNORE band (left out on both sides)
   instead of counting as arm. Cam_right is the WRIST camera (verified from frames), not a side camera.

## Fitted values
- Joint offsets (deg, ADDED to calib/so101_units defaults; fitted jointly with the session cameras on TRAIN
  frames, bound +-25 around the start): shoulder_pan -0.35, shoulder_lift +2.83, elbow_flex -6.00,
  wrist_flex +1.70, **wrist_roll -99.56 (total -189.6)**. Gripper map: not fitted here (stage 2).
- 19 camera sessions (consecutive train episodes whose per-episode cameras agree within 12 px on a table
  grid). Typical camera (session 18, episodes 379-523, 134 train episodes), robot base frame:
  x -0.018 m, y 0.128 m, z 0.498 m, yaw 0.182 rad, tilt_a 0.493 rad (~28 deg from vertical),
  tilt_b -0.011 rad, **vertical FOV 27.8 deg**. Across sessions: z 0.46-0.57 m, fovy 23-31 deg.
- Caveat: height and FOV trade off (higher camera + narrower FOV look alike); the pair is less certain than
  the viewing direction. The real webcam's FOV was not measured independently.

## Remaining error, from the overlays (`overlays_cam_sessions_offsets_roll/`, 12 held-out frames)
- Big-arm frames 0.79-0.92. Low scores are the parked pose at frame 0 (small silhouette, white-taped camera
  mount is not dark so the real mask misses it) and frames where the cable is wider than 15 px near the lens.

## Method (short)
Arm = dark pixels (V < 75, or V < 130 and bluish: lit black plastic), not green, closed; thin dark structures
(opening with a 15 px disk removes them) = ignore. Sim = Menagerie SO-101 posed at observation.state via
calib/so101_units + offsets. Optimiser: a no-OpenGL point-splat silhouette (PointArm, IoU 0.94 vs OSMesa at
160x120), soft IoU, CMA-ES (pycma 4.0.0). Reported numbers re-rendered with MuJoCo OSMesa on full meshes.
Held-out episodes (friend_split.json) are never used for fitting; each uses the camera of its recording
session (by episode index).

## Commands (container `sprint`, cwd /work/wt-friendrig, `run.sh` = MUJOCO_GL=osmesa, nice 10, PYTHONPATH pylib)
```
python -m friendrig.camfit frames       --out /work/outputs_large/friendrig/camfit --workers 8
python -m friendrig.camfit fit-global   --out ... --workers 8
python -m friendrig.camfit fit-episodes --out ... --workers 8
python -m friendrig.camfit sessions     --out ...
python -m friendrig.camfit fit-sessions --out ... --workers 8 --offsets --rounds 2 --iters 60 --max-frames 120 --offset-frames 40
python friendrig/rollscan.py cam_sessions_offsets.json      # wrist_roll scan (one-off scan)
python -m friendrig.camfit fit-sessions --out ... --workers 8 --offsets --rounds 2 --iters 60 --max-frames 120 \
    --offset-frames 40 --init-offsets 0.1 4.2 -9.3 6.1 -90 --init-cams cam_sessions_offsets.json --bound 25 --tag roll
python -m friendrig.camfit eval --out ... --cams cam_sessions_offsets_roll.json --with-train --overlays 12
python -m friendrig.camfit eval --out ... --cams cam_sessions_offsets_roll.json --cable-px 21 --tag roll_cable21
python -m friendrig.camfit eval --out ... --cams cam_sessions_offsets_roll.json --no-offsets --tag roll_cams_default_offsets
python -m friendrig.camfit eval --out ... --cams cam_sessions_offsets_roll.json --global-cam --no-offsets --tag global_nooffsets
```
Logs: `/root/work/logs/friendrig/camfit.log`, `camfit_eval.log`, `rollscan.log`.
Outputs (server only, derived from the friend's frames): `/root/work/outputs_large/friendrig/camfit/`
(`eval_*.json`, `cam_sessions_offsets_roll.json`, overlays). Drive: `gdrive:so101-sprint/hanoi_oct13/friendrig/`.
