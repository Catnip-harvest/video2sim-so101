# Stage 2 gate: wrist camera (cam_right) check — PARTIAL (finding: Menagerie wrist_cam is on the other gripper face)

**Verdict: PARTIAL.** The evidence says the default sim wrist camera does NOT match: in the real cam_right the
FIXED jaw tip is on the image right and the moving tip on the left; Menagerie's `wrist_cam` sees the mirror image.
The friend's camera is mounted on the other face of the gripper. After turning the camera mount 180 deg about the
wrist_roll axis and a small fitted correction, held-out jaw-band IoU goes from **0.095 to 0.430** (median 0.000 to
0.450). Tip-level agreement is good when the jaw is closed or holding (0.57-0.87 in the overlays); with the jaw
wide open the real moving tip leaves the image and the match is poor.

## Numbers (HELD-OUT: 200 frames = 4 per held-out episode, spread over gripper values; bottom 30 % of the image;
MuJoCo OSMesa render, full meshes, 640x480, hard IoU of jaw pixels)
| Variant | mean IoU | median | frames with IoU >= 0.5 |
|---|---|---|---|
| Menagerie default (wrist_cam as shipped, fovy 61) | 0.095 | 0.000 | 0 % |
| camera on the other face, nothing fitted | 0.216 | 0.213 | 1.5 % |
| default face, pose + fovy fitted | 0.219 | 0.000 | 26 % |
| **other face + pose/fovy + gripper map fitted (adopted)** | **0.430** | **0.450** | **48 %** |

Train soft IoU (optimiser objective, 300 TRAIN frames): 0.264 -> 0.546.

## Adopted correction
camera_mount body turned 180 deg about the gripper z (= wrist_roll) axis, then: dpos (+11.2, -6.0, +9.5) mm in the
mount frame; rotation (0.148, -0.121, 0.040) rad about the camera's x, y, z; **fovy 60.7 deg** (Menagerie 61:
unchanged within noise). Gripper map: lerobot 0 % -> -9.5 deg, 100 % -> 107.4 deg (default -10 / 100).
In generated scenes the mount's two collision boxes are made non-colliding (on the other face they would overlap
the moving jaw's sweep; the real housing was not measured).

## Caveats
- Jaw mask = dark and not green-tinted in the bottom 30 %: shaded cube faces, the operator's sleeve or cables can
  leak in; the metric only sees the small jaw tips.
- Some real frames show no jaw tips while the gripper reads closed (video/state sync, or tips out of view).

## Commands (container `sprint`, cwd /work/wt-friendrig, via run.sh)
```
python -m friendrig.wristcam frames --out /work/outputs_large/friendrig/wrist --workers 4
python -m friendrig.wristcam fit    --out ... --workers 8 --n-frames 300 --iters 60
python -m friendrig.wristcam fit    --out ... --workers 8 --n-frames 300 --iters 60 --flip --gripper-map
python -m friendrig.wristcam eval   --out ... --workers 4 --overlays 8
```
Log `/root/work/logs/friendrig/wrist.log`; results `/root/work/outputs_large/friendrig/wrist/wrist_eval.json`,
`wrist_fit*.json`, overlays `wrist_overlays_*` (server/Drive only).
