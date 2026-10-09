INCONCLUSIVE: held-out mean IoU 0.49 (seed 0; 0.47 on a repeat). The pass mark of 0.8 is NOT met, but this test cannot reach 0.8 even when every frame gets its own camera (0.66), and it cannot tell the default mapping from a 20-degree error.
# Gate G0: overlay of the friend's joint data on the real arm. Measured 9 Oct 2026, 10:15-11:05 UTC.

## What was measured
- Data: 5 table-to-bowl episodes (1, 155, 205, 265, 373) from MinhDuk/final_dataset_annotated, using cam_head
  and observation.state. Each episode gives 5 frames chosen automatically: the arm is visible, it has left the
  rest pose, it moves less than 20 deg/s, and the 5 poses are as different as possible. In each episode,
  3 frames are used to fit the camera and 2 are held out (10 held-out frames in total).
- Real mask: a pixel counts as arm if it is dark (max RGB < 70) and not green. A 5 px closing and a 7 px opening
  clean it up. Sim mask: MuJoCo segmentation render of the Menagerie SO-101, with qpos = lerobot_to_sim(state)
  and kinematics only.
- Camera: 7 numbers per episode (aim point on the table, distance, azimuth, elevation, roll, fovy). They are fitted
  by random search and then Nelder-Mead on the fit frames only. Implausible cameras (inside the arm, or below
  0.2 m) are rejected.

## Numbers (IoU at 640x480)
| mapping | fit frames (in-sample) | held-out | held-out, thick structures removed from both masks |
|---|---|---|---|
| default ArmCalib (seed 0) | 0.542 | **0.494** | 0.499 |
| default ArmCalib (seed 1 repeat) | 0.555 | 0.468 | 0.475 |
| shoulder_lift sign flipped | 0.456 | 0.368 | 0.370 |
| elbow_flex +20 deg | 0.519 | 0.459 | 0.466 |
| wrist_flex +20 deg | 0.546 | 0.466 | 0.462 |
| wrist_roll offset 0 (instead of -90) | 0.564 | 0.514 | 0.530 |
| wrist_roll offset 0 (seed 1 repeat) | 0.548 | 0.485 | 0.464 |
| offsets fitted (+-15 deg) jointly with cameras | 0.561 | 0.501 | 0.511 |
| ceiling: a separate camera for every held-out frame | n/a | 0.658 | n/a |

Default held-out IoU per episode (seed 0): ep1 0.45, ep155 0.46, ep205 0.59, ep265 0.61, ep373 0.36.

## What it means, plainly
1. The pass mark is not met: 0.49 held-out against 0.8 required.
2. That does not prove the mapping is wrong. Even with a free camera per frame (the "ceiling"), the masks only
   reach 0.66 on average and 0.81 at best. The limit comes from the measurement: black cables merge into the
   arm mask, shiny arm faces look grey and drop out of it, the cube hides the gripper, and the camera is very close
   to the arm, so lens distortion and small sync errors cost a lot of pixels.
3. The test catches only gross errors. A flipped shoulder_lift sign drops held-out IoU by 0.13, which is clear.
   But +20 deg on elbow or wrist_flex costs only 0.03, and re-running the default with another seed already moves
   the result by 0.03. Wrist_roll offset 0 versus -90 cannot be told apart (it scored 0.51/0.49 vs 0.49/0.47).
4. Fitting per-joint offsets jointly with the cameras barely helps (0.50). The fit moved wrist_roll by +14.6 deg,
   close to its +-15 bound, and moved everything else by less than 2 deg. This is not evidence for any offset.
   The result is saved in calib/arm_calib_friend.json, marked NOT VERIFIED.
5. What we can say: the signs of shoulder_lift are consistent with the default (identity). The default puts
   the arm in roughly the right place: in the overlays the forearm and wrist land in the right area, often offset
   by a few cm. The offsets (+-20 deg), the wrist_roll -90 offset and the gripper map are still unverified.

## To make this gate decisive (not done; needs owner decision)
- A better real-arm mask (SAM2 with box prompts, or hand-labelled masks for about 10 frames) so the ceiling gets
  close to 0.9.
- One camera shared by the many episodes that show the same view (the bowl sits at the same pixel). This gives
  many more poses per camera and makes errors of +-10-20 deg visible.
- Optionally, a test that does not use pixels: put the real arm in a known pose (e.g. all-zero/middle) and
  compare the reading with the sim zero.

## Files
- Code: video2sim-so101/calib/overlay_gate.py (CLI: frames | run | ceiling | fit-offsets), test calib/tests/test_overlay_gate.py
- Outputs: /root/work/outputs_large/calib/overlay/ : results_default.json, results_seed1.json, results_wrong.json,
  results_fitted.json, results_ceiling.json, overlays/*.png (green = real mask outline, magenta = sim outline),
  sheet_*.png (contact sheets), real_masks_sheet.png (mask check: green kept; magenta removed only by the thick opening)
- Logs: /root/work/logs/g0_*.log
