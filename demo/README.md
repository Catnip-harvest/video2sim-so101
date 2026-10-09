# Watch me once: demo extractor

**Problem.** We want the SO-101 to learn a pick-and-place from a person doing it once on video.

**What this produces.** It takes **one third-person phone video** of a person who picks up a small object and places it somewhere, for example into a bowl. From it, the extractor writes `task_spec.json`, an **object-centric** task description for a two-finger parallel gripper. The spec contains:
- the object's size;
- its start and place poses;
- its smoothed 3D path;
- grasp and release times, approach direction and pinch width;
- a suggested gripper plan.

All positions are already mapped into the robot's reach. The mjlab RL task, built by another worker, consumes only this JSON.

**Why the spec is object-centric.** A human arm hangs from the shoulder, but the SO-101 is bolted to the table, so the hand's path does not transfer to the robot. The *object's* path does. This is the X-Sim-style design. The hand is used only for four things:
- timing (grasp and release);
- approach side;
- pinch width;
- one depth cue, at low weight.

> **Status.** End-to-end runs exist on a synthetic clip with exact ground truth and on one real stock clip. Real-phone-video quality is **unverified until the owner's clip arrives**. See "Tested on".

---

## One command

```bash
cd "mi300x-rss/sprint/manip/demo"
.venv/Scripts/python.exe -I extract_demo.py path/to/your_clip.mp4 --out outputs/my_demo
```

- Expect about 6–7 min on this laptop's CPU for a 7 s clip. Most of that is SAM 2.
- Re-running with different downstream options? Add `--reuse-object-masks` to skip SAM 2.
- If the automatic object pick goes wrong (check `debug_overlay.mp4`), click the object yourself: `--object-point x,y`, or give a box with `--object-box x0,y0,x1,y1`. Both are pixels in frame 0 of the source video.
- If you know the object's dimensions, pass them: `--object-size-m 0.04,0.04,0.04`.

### Outputs (in `--out`)

| File | What it is |
|---|---|
| `task_spec.json` | The contract for the RL task (schema below) |
| `debug_overlay.mp4` | Hand skeleton, object mask (magenta), A4 corners (yellow), table axes (x red / y green / z blue, 10 cm), object path (orange), and phase label per frame |
| `trajectories_3d.png` | Object path in the human table frame and in the robot frame, with the reach half-disc |
| `trajectory_debug.json` | Per-frame object samples, rejection reasons, per-source depth estimates, hand pinch (debug only) |
| `object_masks.npz` | SAM 2 masks (packed bits); the cache that `--reuse-object-masks` reads |
| `so101_task_spec.json` | The same demo in the mjlab task's own format, `so101_pickplace_task_spec/v2`. Point `SO101_TASK_SPEC` at this file. Converted by `demo_extractor/so101_export.py`: object size clamped into the task's 20–50 mm band, ±1 cm start/place ranges, trajectory padded so grasp/release fall inside it. |

### One-time setup (already done on this laptop)

```bash
C:/Users/vieth/AppData/Local/Programs/Python/Python313/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -I tools/fetch_models.py     # each model into its own third_party/<name>/
```

---

## Filming instructions (for the owner)

**Why the rules matter.** The A4 sheet makes the camera's height, angle and focal length *measured*, not guessed. Slow, unoccluded motion keeps the object tracker locked on.

1. **Phone propped still.** Use a stand, books or a mug. Never handheld; the camera must not move during the take, not even a slow drift.
2. **Landscape, 3/4 view from the side.** Put the phone about 30 cm above the table top, 50–80 cm from the action, looking down at roughly 20–35°. Pick and place should both be in frame, with room above them for the lift.
3. **A4 sheet flat on the table, all four corners visible** for the whole clip. It may sit between the object and the bowl, but nothing may cover its corners. Plain white printer paper, no folds.
4. **Start recording with your hand out of frame for 1 second.** The sheet and object are measured from these first frames.
5. **Object: small, rigid, 3–5 cm wide**, something a two-finger gripper can hold: a small block, bottle cap, small sponge, or toy.
   - Avoid symmetric, shiny, transparent, or very dark objects; tracking breaks on them.
   - Avoid objects the same colour as the table.
   - Measure it if you can and pass `--object-size-m`.
6. **Task:** start with thumb and index visibly *open*, pinch the object with thumb and index (two fingers, not a whole-hand grab), lift it a few cm, carry it, put it into a bowl (or onto a spot), open your fingers, and move your hand away and out of frame.
7. **5–10 s, slow and deliberate, roughly constant speed.** No fast swings: motion blur gets frames rejected. Pause about 0.5 s after closing on the object and about 0.5 s before opening.
8. **Hand and object never hidden.** Keep your body, sleeve and other objects from passing in front of either. Approach from the side or above, not from behind the object as the camera sees it.
9. **Good, even light, plain table.** Daylight or a lamp, no hard shadows over the sheet. No patterned tablecloth. Keep clutter out of frame.
10. **Hold the end pose about 1 s** with the object at rest and your hand gone, so the place pose is measured cleanly.

Record at 1080p or 4K; 30 fps is fine. Send the original file, not a messenger-compressed copy.

---

## Output contract: `task_spec.json` (schema `watch_me_once.task_spec/2 (object-centric)`)

**Frame:** `robot_table`.
- Origin is at the robot base, on the table surface.
- Axes: +x forward, +y left, +z up.
- Units: metres and radians. Times are seconds of video time.

A complete example is in `outputs/synthetic/task_spec.json`.

| Key | Meaning |
|---|---|
| `object.size_m` | `[width along the grasp closing axis, depth, height]`. Physical; never scaled. |
| `object.start_pose_xyz_yaw` | Object centre at rest before the pick, plus yaw (the closing-axis direction; object faces are assumed aligned with it). |
| `object.place_pose_xyz_yaw`, `place_target_xyz` | Object centre at rest after the place. |
| **`object_waypoints`** | **Primary.** The object-centre path `[[x,y,z,t],...]`. Smoothed, densely resampled (default 30 Hz), from lift-off − 0.5 s to set-down + 0.5 s. The rest ends are held exactly at the start and place poses. |
| `object_keyframes` | start / apex / place, each with xyz and t. |
| `approach_dir` | Unit vector the gripper should travel along into the grasp: the human hand's wrist→pinch axis at the grasp. `approach_top_down_fraction` = its downward component. |
| `gripper.grasp_width_m` | Object width along the closing axis, clamped to `gripper.max_opening_m` (placeholder 0.06; set with `--gripper-max-open`). |
| `gripper.grasp_yaw_rad`, `closing_axis` | How the two fingers should close. |
| `waypoints` | *Suggested* gripper plan `[[x,y,z,gripper_width,t],...]` with `waypoint_labels`. Derived from the object path (gripper centre = object centre while held) and labelled pre_grasp, grasp_open, grasp_closed, lift, carry_mid, pre_place, place, release, retreat. **Not** the human hand path. |
| `grasp_time`, `lift_off_time`, `set_down_time`, `release_time`, `phases` | Segmentation into approach / grasp / transport / release / retreat. |
| `workspace_mapping` | The human→robot transform and its fit report. |
| `human_table_frame` | The same poses and path in the A4-centred frame, before mapping. |
| `calibration` | Intrinsics; A4 corners; camera pose and height; focal-estimate consistency; hand scale factor. |
| `quality` | Rejected frames by reason; per-source depth disagreement; hand/object tracking fractions; camera motion; event sources; **`warnings`**. |
| `progress_curve` | `null`, unless the optional Robometer hook ran (see below). |

**Human → robot mapping (planar similarity).** In words: rotate and uniformly scale the table-plane coordinates, then shift them.

```
xy_robot = s · Rz(yaw) · xy_human + offset
z_robot  = z_scale · z_human          (z_scale defaults to 1, so heights are kept)
```

- The default fit places pick on the robot's right and place on its left, both at the same radius inside a 0.25 m half-disc in front of the base, clear of a 0.10 m keep-out around it.
- It shrinks `s` until every path point is reachable.
- It never shrinks below a 0.12 m pick–place span, because the task's bin needs at least 0.11 m. A short human move is scaled *up*. `fit_ok: false` then means some path point lies beyond 0.25 m; the task itself accepts up to 0.32 m from the pan axis.
- Override any piece with `--robot-scale`, `--robot-yaw-deg`, `--robot-offset x,y`, `--robot-z-scale`, `--robot-reach`, or `--robot-min-radius`.
- Object size and gripper width are never scaled.

**Human table frame.**
- Origin is the A4 sheet's centre.
- +x runs along the sheet's long edge, in the direction that points right in the image.
- +z points up.

---

## How it works

| Step | Method | Model / licence |
|---|---|---|
| 1. Hand keypoints | 21 image + 21 metric "world" keypoints per frame; MediaPipe-style tracking (the previous frame's landmarks set the next crop) | MediaPipe Hands palm detector + landmark model, **Apache-2.0**. On this laptop it runs as the OpenCV Zoo ONNX conversion via `cv2.dnn`, because Windows Application Control blocks the `mediapipe` package's unsigned `libmediapipe.dll`. `--hand-backend mediapipe` uses the pip package where it loads. No MANO / no sign-up. |
| 2. Calibration | Find the A4 quad (bright, low-saturation, convex, line-refined corners; median over the first ~0.8 s). Then two constraints from the rectangle give the focal length; planar PnP gives the camera pose; least-squares refinement follows. Official terms: plane homography, image-of-the-absolute-conic constraints, IPPE. A quad that does not behave like A4 is rejected: reprojection over 2 px, wrong aspect ratio, or camera below the table. | — |
| 3. Object masks | SAM 2.1 video tracking, streamed one frame at a time. Prompt options: a click (`--object-point`), a box (`--object-box`), or **auto**. Auto first compares hand-free first and last frames (registered for small drift): of the changed blobs, it takes the one nearest the pinch at the first thumb–index closure and prompts frame 0 with its box. If no clean change blob exists, it asks SAM 2 what lies between thumb and index at that moment, tracks it backwards to frame 0, and seeds the forward pass with that mask. | SAM 2.1 hiera-tiny (HF transformers), **Apache-2.0** |
| 4. Object size and start pose | Fit an upright box resting on the table so its projection matches the frame-0 mask's bounding box. Yaw comes from the grasp closing axis. | — |
| 5. Events | Lift-off and set-down come from object motion (the smaller of the centroid shift and the mask-bottom shift, so finger occlusion does not count as motion). Grasp is when the thumb–index gap closes; release is when it starts to reopen. | — |
| 6. Object path | Each frame: the object's view ray from its mask, times a distance fused from three sources by inverse-variance weighting, with the worst outlier against the median dropped. The sources: **(a)** Depth Anything V2, made metric per frame by fitting it to the A4-measured table plane (σ≈4%); **(b)** a vertical transport plane through pick and place (σ≈2 cm across the plane); **(c)** the hand's distance, scaled by one hand-size factor calibrated at the grasp contact (σ≈15%). Frames with occlusion (<40% of rest-area), mask leaks, fast motion (>1 object-width/frame) or no depth are rejected. The path is then pinned to the rest poses, Savitzky–Golay smoothed and PCHIP-resampled. | Depth Anything V2 **Small**, **Apache-2.0**. Base and Large are CC-BY-NC, and Apple Depth Pro is `apple-amlr` (research-only), so none of those are used. |
| 7. Place pose | Final mask ray (hand gone) times a fused distance: depth model, hand at release, and a weak "rests near table height" prior | — |
| 8. Retarget | Gripper centre = object centre while held; width = object width along the thumb–index axis, clamped. A `--pinch-mode thumb-index-middle` variant exists. | — |
| 9. Robot mapping | Planar similarity (above) | — |

---

## Tested on (2026-10-09)

### A. Synthetic clip with exact ground truth: `test_assets/synthetic_pick_place.mp4`

- 7 s, 960×540, 30 fps.
- Rendered by `tools/render_synthetic_demo.py`, a NumPy ray caster, because this shell has no OpenGL.
- Scene: a capsule-built skin-coloured hand pinches a 40 mm red cube, lifts it 12 cm, carries it 29 cm and lowers it into a bowl. An A4 sheet lies on the table. The camera sits 0.31 m above the table at a 30° view elevation.

Reproduce:
```bash
.venv/Scripts/python.exe -I tools/render_synthetic_demo.py --out test_assets/synthetic_pick_place
.venv/Scripts/python.exe -I extract_demo.py test_assets/synthetic_pick_place.mp4 --out outputs/synthetic
.venv/Scripts/python.exe -I tools/evaluate_against_gt.py --spec outputs/synthetic/task_spec.json --gt test_assets/synthetic_pick_place_gt.json
```

Fresh end-to-end run with the auto object prompt (change-blob method); SAM 2 took about 6 min on CPU for 105 frames.

| Quantity | Result (estimated vs ground truth) |
|---|---|
| A4 corners | 1.0 px mean, 1.75 px max |
| Focal length | 703.2 vs 703.4 px (−0.02%); found from the A4 alone, no FOV prior |
| Camera centre | 4.1 mm error; height 0.312 vs 0.310 m |
| Table normal | 0.22° |
| Object size | 39.0 × 39.0 × 39.5 mm vs 40 mm cube |
| **Pick** (object start centre) | **1.5 mm** |
| **Place** (object in bowl) | **19.5 mm**, mostly along the view ray (bowl rim hides the cube's lower half) |
| **Object path while carried** | **13 mm mean, 38 mm max** |
| Grasp / lift-off / set-down / release time | −0.03 / +0.07 / −0.03 / +0.20 s |
| Grasp yaw | 8.4° |
| SAM 2 mask IoU | 0.937 mean, 0.896 min |
| Workspace fit | scale 1.0; every path point at 0.11–0.20 m from the base |

**Caveat: the transport-plane prior helps the synthetic case.** The synthetic carry *is* planar. On real demos people arc out of that plane by a few cm, so expect worse numbers than these.

Monocular hand depth was poor on the rendered hand: the ratio to ground truth swung 0.96–1.62 per frame. That is why the hand has the lowest weight and is not used for the path.

### B. Real stock clip: Pexels 7986194, "a person getting keys on the table"

- 960×540, 25 fps, 7 s, Pexels License.
- Source and licence are recorded in `third_party/pexels_7986194/SOURCE.txt`.
- The clip breaks three of the filming rules: there is **no A4 sheet**, the **camera drifts about 23 px**, and the keys **leave the frame** instead of being placed. It therefore tests the real-video front end and the failure reporting, not 3D accuracy.

Command used:
```bash
.venv/Scripts/python.exe -I extract_demo.py third_party/pexels_7986194/7986194-sd_960_540_25fps.mp4 \
    --out outputs/pexels_7986194_keys --object-box 250,305,520,355
```

What worked:
- **Hand keypoints** landed on the real hand in 49% of frames, which is every frame where the hand is in view.
- **SAM 2** followed the keys, a thin, shiny, dangling object, from rest through the lift until they left the frame.
- **Lift-off** came out at 3.92 s, matching the video.
- **Object left view** was detected at 4.8 s.

What was correctly refused or flagged:
- A false "A4" quad (31.8 px corner reprojection error) was rejected.
- Camera motion was flagged.
- The no-A4 fallback was flagged.
- The hand-scale sanity check failed (0.26).

The 3D numbers from this clip are not meaningful.

What failed:
- **Auto prompting.** The first-vs-last-frame difference is polluted by parallax from the camera drift, and in this side view the pinch point sits on the fingers. A manual `--object-box` was needed.
- **Grasp timing.** The person approached with thumb and index already nearly closed, so the gap changed by less than 1 cm. Grasp time therefore falls back to lift-off, with a warning.

### C. First real demo (chosen internet clip): Pexels 7488188, sugar cube into a teacup

- Author: Nataliya Vaitkevich. Pexels License; details and the rejected candidates are in `third_party/pexels_7488188/SOURCE.txt`.
- **Why this clip:** the camera is still (0.09 px drift); the view is elevated 3/4; one hand does a thumb–index pinch of a small, rigid, matte cube off the table, lifts it, carries it and drops it into a cup, all in about 2.2 s.
- **Weaknesses:**
  - no A4 sheet;
  - the cube is smaller than the robot's band (about 16 mm);
  - the motion is fast;
  - the hand is blurred and half out of frame at the pick (hand found in 22% of frames);
  - the cube sinks into the tea, so the place is its last visible position at the cup.
- Trimmed to 0–3 s with `tools/trim_clip.py`.

**Calibration fallback.** The metric scale comes from an assumed standard sugar cube (16 mm). The table tilt comes from the tea surface, a guaranteed-horizontal circle: its ellipse axis ratio is 73/194, so the ray is about 22° below horizontal, which gives a camera pitch of about 42° assuming a level horizon. The focal length comes from an assumed `--hfov-deg 40`. Two checks:
- The average-hand scale alone gave a 16.5 mm cube, within 3% of the assumed 16 mm.
- The depth model's zero-shift tilt disagreed by 19.8°, which is why it is not used for the tilt.

```bash
.venv/Scripts/python.exe -I tools/trim_clip.py third_party/pexels_7488188/7488188-sd_540_960_30fps.mp4 \
    test_assets/pexels_7488188_sugar_cube_0-3s.mp4 --start 0 --end 3.0
.venv/Scripts/python.exe -I extract_demo.py test_assets/pexels_7488188_sugar_cube_0-3s.mp4 \
    --out outputs/pexels_7488188_sugar_cube --max-fps 30 --object-size-m 0.016,0.016,0.016 \
    --hfov-deg 40 --camera-pitch-deg 42 --object-box 464,281,510,322
```

**Results.**
- SAM 2 tracked the cube from rest through the carry into the cup.
- Events: grasp 1.03 s (from lift-off, because the hand was not detected), lift-off 1.20 s, set-down 2.00 s, release 2.07 s. These match the video frame by frame.
- 7 of 25 carried frames were rejected (6 fast motion, 1 no depth).
- Human frame: pick-to-place distance 0.165 m, carry apex 8.7 cm.
- Robot frame (scale 0.78): pick (0.184, −0.064, 0.008), place (0.184, +0.064, 0.048).
- `so101_task_spec.json` passes the task's `load_task_spec`.

**Which numbers to trust (no A4 sheet).**
- **Trustworthy:** the timings, the object's 2D track, and the *shape* of the path (lift, carry, lower).
- **Scale-dependent:** every metric distance (pick-to-place distance, lift height) inherits the assumed cube size (16 mm ±3 mm, so ±20%) and the assumed FOV.
- **Rough:** the height of the place and depth along the view ray (several cm).
- **Not used:** the hand's 3D path.

**Feeding it to the mjlab task.** Run this only when no other heavy Python job is running, because it commits about 2.5 GB:

```bash
cd sprint
SO101_TASK_SPEC=$PWD/manip/demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json MJLAB_WARP_QUIET=1 ./.venv-train/Scripts/python.exe -I -u manip/task/scripts/smoke_env.py --device cpu --num-envs 2
```

**Research datasets considered, none downloaded.** All are far over the 300 MB budget, and most are registration-gated:
- **DexYCB:** about 1000 sequences; the licence is not stated on the aggregators checked (verify at dex-ycb.github.io).
- **HO-3D v3:** the archive is about 67 GB.
- **ARCTIC:** needs MPI registration (with SMPL-X/MANO terms).
- **OakInk v1:** CC-BY-NC-SA-3.0 per its HF card.

**Real-phone-video quality with an A4 sheet is unverified until the owner's clip is filmed.**

---

## What the extractor cannot handle

- **A moving camera.** Calibration is done once, from the first frames. Drift over 3 px is flagged, and auto-prompting tends to fail; use `--object-point` or `--object-box`.
- **No A4 sheet**, or a sheet with a hidden corner. The fallback works like this, and a warning always says so; expect several-cm errors:
  - **Focal length** is guessed from `--hfov-deg` (default 65).
  - **Metric scale** comes from `--object-size-m` if given (preferred), otherwise from an *average* hand.
  - **Table tilt** comes from `--camera-pitch-deg` with a level horizon (preferred; measure it from a round rim or liquid surface: ray elevation = asin(minor/major)), otherwise from the depth model assuming zero disparity shift. On the sugar clip that depth-model tilt was off by 20°.
- **Two hands, hand-overs, or re-grasps.** Only one grasp→release cycle is extracted.
- **Whole-hand grasps, pushing, sliding, tool use, deformable objects.** It assumes a thumb–index pinch on a rigid object.
- **Grasps without a visible open→close of thumb and index.** Grasp time then falls back to lift-off, with a warning.
- **Objects hidden for long stretches.** Gaps over 0.5 s are interpolated and flagged. Shiny, transparent and symmetric objects break SAM 2 tracking. Tiny objects (under about 15 px across) are unreliable.
- **An object that leaves the frame or is never put down.** The place becomes the last tracked position, with a warning.
- **Strongly curved carries.** The transport-plane prior pulls the path toward the pick–place vertical plane. The depth model and hand partly correct this; the per-source disagreement is in `quality`.
- **Object yaw beyond "faces aligned with the pinch",** and orientation changes of the object other than yaw.
- **Lens distortion and off-centre principal points** (phone video is normally already corrected), and portrait videos with rotation metadata. Film in landscape.

---

## Optional: progress curve (MI300X only)

`tools/progress_curve_robometer.py` gives a per-frame task-progress curve from Robometer-4B via LeRobot: the progress of growing video prefixes. It needs about 9 GB of VRAM, so it was **not downloaded or run here**. On the GPU box:

```bash
pip install -e ".[robometer]"     # inside a LeRobot checkout
python -I tools/progress_curve_robometer.py CLIP.mp4 --task "pick up the block and put it in the bowl" \
    --spec outputs/my_demo/task_spec.json --device cuda
```

It writes `progress_curve.json` and links it from `task_spec.json`. API per the LeRobot Robometer docs (checked 2026-10-09); the script is unverified until run.

## Notes for this laptop

- **CPU only.** C: had 2.8 GB free, and the 2.6 GB CUDA torch wheel failed with "No space left on device". `--device cuda` works wherever a CUDA torch is installed.
- **One heavy torch process at a time.** The extractor waits while free RAM is below 2.5 GB, and loads SAM 2 and the depth model sequentially, never together.
- **No `mediapipe` DLL** (blocked by Application Control) and **no OpenGL in agent shells**. Do not disable Application Control. The ONNX hand backend and the NumPy renderer route around both.
- `EXO_TO_EGO.md` asks whether converting the third-person video to a head or wrist camera view would help (short answer: no).
