# Branch `datagen/wide-camera`: does wider data help? (P2.5, owner's "data engine is the lever")

All results are in simulation. Cube demos come from the one-video engine.

## What this branch does
- Adds `generate.py --wide`. Per episode (fixed for the whole episode, no shake), it widens:
  - the front camera: height +-15 cm, pitch +-15 deg, yaw +-15 deg, distance +-10 cm, FOV 45 +-8 deg;
  - image exposure on both cameras: brightness x0.75-1.3, contrast x0.8-1.2, white balance 0.9-1.1, gamma 0.85-1.15, mild noise, blur in ~30 % of episodes.
- These draws use their own random streams, so **the scenes, physics and trajectories are identical** to the narrow dataset. Only what the camera sees changes. That makes a clean paired experiment.
- Before rendering it rejects camera draws where the cube or gripper covers < 20 px in the first frame (segmentation mask) and redraws.
- Tools: `datagen/tests/test_wide.py`, a contact sheet and `compare_runs.py` (narrow vs wide arrays).

## How to run (inside container `sprint`, cwd `/work/wt-wide/datagen`, `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`)
```
nice -n 10 /work/manip/datagen/.venv/bin/python generate.py \
  --spec /work/wt-wide/demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json \
  --n 1000 --render --render-quality fast --record-hz 10 --variety --wide --size 256x256 --codec h264 --seed 1 \
  --workers 8 --out /work/outputs_large/datagen/sugar_v10_n1000_wide
# export: same command + --finalize-only --export-frame molmoact2 --camera-names front=cam_head,wrist=cam_right \
#         --dataset-dir /work/outputs_large/datagen/sugar_v10_n1000_wide_molmoact2
python validate_dataset.py /work/outputs_large/datagen/sugar_v10_n1000_wide_molmoact2 --lerobot
```
Then `small/prep_small_dataset.py` (branch `small/smolvla-act`) and train ACT with the P2.2 recipe on `sugar_v10_n1000_wide_small`.

## Results
Data (source: `/root/work/gates/gate_p25_data.md`, PASS): 1000 accepted / 1055 attempts (94.8 %, same as narrow); 187,401 frames; camera redraws 19/1019 = 1.9 %; 0 physics differences vs narrow; tests 9/9.

Narrow ACT vs wide ACT, same scenes, paired (source: `/root/work/gates/gate_p25_eval.md`, PASS):

| Test set | n | L_full narrow -> wide | L_task narrow -> wide |
|---|---|---|---|
| in-range s3000 | 100 | 17 -> 29 | 69 -> 76 |
| s4000 none | 50 | 7 -> 16 | 31 -> 39 |
| camera | 50 | 10 -> 21 | 30 -> 44 * |
| exposure | 50 | 10 -> 17 | 30 -> 43 |
| colour | 50 | **2 -> 18 \*\*** | 18 -> 34 * |
| combo | 50 | 3 -> 13 | 19 -> 33 |
| distractor | 50 | 1 -> 7 | 12 -> 17 |
| camera_pitch | 50 | 5 -> 19 * | 25 -> 36 |
| camera_distance | 50 | **4 -> 20 \*\*** | 25 -> 37 |

\*\* significant after Holm over the whole scoreboard (p_Holm 0.027 each; the official verdict). \* significant only within the owner's 22-test narrow-vs-wide family. Unmarked: not distinguishable. Wilson intervals for every cell: `/root/work/SCOREBOARD.md` (ACT and ACT-wide rows).

Read before quoting:
- Wide >= narrow on all 22 comparisons, but only the starred ones are significant.
- The camera rungs are **mostly inside the wide training ranges**. Gains there show coverage, not out-of-range generalisation.
- Colours were **not** widened, so the colour gain (2 -> 18) is the cleaner evidence that breadth helps.
- Wide data did **not** fix distractors: ACT-wide's own distractor drop (L_task 39 -> 17) is significant.

Video: `/root/work/outputs_large/deliverables/videos/narrow_vs_wide_camera_s4000_id1.mp4` (one scene; ACT fails, ACT-wide passes).

## What you may / may not claim (honest-claims list)
May claim:
- On this task, in simulation, ACT trained on wider camera/exposure data beat narrow-data ACT on colour and camera_distance L_full (Holm-significant), on the same scenes.
- Success rates **in simulation** with intervals, under a verifier that needs both jaws on the object and forbids welds and teleports.

May not claim:
- "Generalises" beyond the tested perturbations; camera gains as out-of-range generalisation.
- A difference on any unstarred row ("not distinguishable at n=50").
- Anything about a real robot; laptop speed; that the JEPA judge works; any drawer, C/D, Hanoi or dexterous result.
