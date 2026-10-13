# Stage 3 gate: objects and look — PASS (sizes, scene), PARTIAL (look)

**Cube 39.8 mm** (median over 203 clean train episodes; p10-p90 37.5-40.6 mm; median silhouette IoU 0.97).
**Bowl: rim radius 68.4 mm (137 mm across), height 59.5 mm, base radius 48.2 mm** (p10-p90: rim 59.6-78.1,
height 38.0-63.9, base 37.0-51.7 mm; silhouette IoU ~0.8). The bowl is less certain than the cube: its shaded
outer wall is barely brighter than the table, so the bowl mask is grown from the bright inner wall.

## Method
With the fitted session camera a known 3D shape projects to a known silhouette, so the shape is FITTED instead of
back-projecting blobs: cube = centre, yaw, side (hull of its 8 corners); bowl = centre, rim radius, height, base
radius (a frustum: hull of its two circles). IoU against the green mask / bowl mask in the first cam_head frame,
arm pixels ignored. Sizes = median over episodes with clean fits (both IoU > 0.6, not touching the image border).

## Scene (`friendrig/profile.py`, used by datagen when `--profile` is given)
- Cube: box 39.8 mm, colour = median real green (0.145, 0.412, 0.349).
- Bowl: base 48 mm, rim 68 mm, height 59.5 mm, real white (0.753, 0.757, 0.812): a box contact floor (box-box
  contacts settle; the first try with a cylinder floor left 2/20 cubes jittering), a visual disc, and 20 tilted
  wall panels with a real cavity. Geoms are named bin_floor / bin_wall_* so the verifier's contact rules apply
  unchanged.
- Arm: all robot materials black (0.05, 0.05, 0.06).
- Table: per-session **orthophoto** of the real empty table (1 mm/px, base frame x -0.15..0.60, y -0.45..0.45 m),
  made by resampling the clean plate through the fitted camera. The camera sees ~18 % of that area; the rest is a
  tiled crop of the seen part (seams visible). Wrist renders therefore show the real table surface in place.
- cam_head renders are **composited**: sim pixels where segmentation says robot/cube/bowl, the real clean plate
  elsewhere (`"composite": true`; plain MuJoCo with `"composite": false` or `replay --plain`, for an A/B).
  Arm shadows on the table are lost in the composite.
- Domain randomisation stays available: with `"look_dr": true` (default for generation) the generic lighting draws
  of datagen/r2s/variations.py apply and cube/bowl colours get x0.9-1.1 per channel; the twin replay uses fixed
  nominal lighting.

## Clean plates (`friendrig/plate.py`): per-pixel median over up to 240 cam_head frames per session
Arm (dark, dilated), cube, every white blob (hull, dilated) and the grown bowl outline are left out of each frame.
Session 18 (episodes 379-523): 240 frames from 119 train episodes, 0.011 % holes. Session 9: 240 frames / 58
episodes, 5.9 % holes. Small sessions (3-8 train episodes) have 7-21 % holes (inpainted). **Known defect:**
session 10 (episodes 224-295) keeps a bowl-shaped artefact: a bowl sits in the same place in most of its episodes
and is not fully caught by the white mask (white balance makes it bluish). Plates:
`/root/work/outputs_large/friendrig/camfit/plates/` (server/Drive only).

## Commands (container `sprint`, cwd /work/wt-friendrig, via run.sh)
```
python -m friendrig.objects fit --out /work/outputs_large/friendrig/camfit --cams cam_sessions_offsets_roll.json --workers 8
python -m friendrig.plate plate --out ... --cams cam_sessions_offsets_roll.json --workers 8
python -m friendrig.build_profile --camfit ... --cams cam_sessions_offsets_roll.json \
    --profile .../friend_bowl_profile.json --wrist /work/outputs_large/friendrig/wrist/wrist_fit_gmap_flip.json
```
Logs `/root/work/logs/friendrig/{objects,plate,profile}.log`; `objects.json`, `friend_bowl_profile.json`,
`friend_arm_calib.json` on the server.
