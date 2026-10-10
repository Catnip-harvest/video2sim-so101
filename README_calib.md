# Branch `calib/so101-units`: SO-101 units and the G0 overlay check

All results here are measurements on a real dataset's video against a simulated arm. None of it shows a real robot running a policy.

## What this branch does
- `calib/so101_units.py` is **the one place** where SO-101 joint values change units. Three conventions meet:
  - `sim`: MuJoCo Menagerie SO-101, radians (what our datagen writes);
  - `lerobot`: LeRobot after PR #777, degrees, gripper 0..100 % (the friend's dataset);
  - `molmoact2`: the older LeRobot calibration MolmoAct2-SO100_101 was trained in
    (`shoulder_lift_old = 90 - shoulder_lift_new`, `elbow_flex_old = elbow_flex_new + 90`).
- `calib/overlay_gate.py` (gate G0) asks: if we put the friend's recorded joint values into the sim arm, does the sim arm land where the real arm is in the overhead camera? It compares a real arm mask (dark pixels) with a sim segmentation mask (IoU), after fitting a camera per episode on 3 frames and testing on 2 held-out frames.
- `calib/dump_real_frames.py` dumps frames from the friend's dataset.

## How to run (inside container `sprint`, cwd `/work/video2sim-so101`, `export MUJOCO_GL=osmesa`)
```
python -m pytest -q calib/tests/test_so101_units.py calib/tests/test_overlay_gate.py
python calib/overlay_gate.py frames --episodes 1 155 205 265 373 --out /work/outputs_large/calib/overlay
python calib/overlay_gate.py run --out /work/outputs_large/calib/overlay --workers 4 --seed 0
python calib/overlay_gate.py ceiling --out /work/outputs_large/calib/overlay --workers 4
python calib/overlay_gate.py fit-offsets --out /work/outputs_large/calib/overlay
```
`run --only <variant>` runs one mapping (e.g. the deliberately wrong ones). Keep `--workers` at 4 or less when a GPU job shares the box.

## Results (source: `/root/work/gates/gate_g0_overlay.md`, measured 9 Oct 10:15-11:05 UTC)
**Verdict: INCONCLUSIVE.** Pass mark was held-out IoU >= 0.8.

| Mapping | Held-out IoU |
|---|---|
| default mapping (seed 0) | **0.494** |
| default mapping (seed 1 repeat) | 0.468 |
| shoulder_lift sign flipped (deliberately wrong) | 0.368 |
| elbow_flex +20 deg | 0.459 |
| wrist_flex +20 deg | 0.466 |
| wrist_roll offset 0 instead of -90 | 0.514 (repeat 0.485) |
| offsets fitted jointly with cameras | 0.501 |
| ceiling: a separate camera per held-out frame | 0.658 |

What it means, plainly:
- The test **cannot reach 0.8** even with a free camera per frame (0.66): cables merge into the arm mask, shiny faces drop out, the camera is very close.
- It catches only gross errors: a flipped shoulder_lift sign costs 0.13. A 20-degree error costs about 0.03, which is the same as seed noise.
- Signs look right; offsets (+-20 deg), the wrist_roll -90 offset and the gripper map are **not verified**.
- `calib/arm_calib_friend.json` (fitted offsets) is marked NOT VERIFIED and is **not committed** (third party's calibration).
- Because of this, all training tonight was sim-only.

Related gates: `gate1_render.md` PASS (0.92 attempts/s, 96.8 % accepted at 10 Hz fast render), `gate1_smoke_molmoact2.md` PASS (bs16 1.23 s/step, 146 GB GPU peak).

## What you may / may not claim (honest-claims list)
May claim:
- The unit conversions are in one tested module, and the overlay test detects a flipped shoulder_lift sign.
- Success rates **in simulation** on held-out random scenes, with intervals (other branches), under a verifier that needs both jaws on the object and forbids welds and teleports.

May not claim:
- **Anything about a real robot.** G0 overlay is INCONCLUSIVE; the sim-to-real gap is large (FD 1329 vs floor 15, see `gate_jepa.md`).
- That the friend's calibration offsets are correct.
- Laptop speed, "generalises" beyond tested perturbations, model differences without a significant paired test, JEPA judge working, any drawer/C-D/Hanoi/dexterous result.
