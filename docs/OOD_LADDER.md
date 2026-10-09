# Out-of-distribution (OOD) ladder

Question: has the policy learned the pick-and-place *behaviour*, or memorised the *look* of
the training scenes? Every test so far drew its scenes from the same random ranges as the
training data. The ladder draws them from **outside** those ranges, one factor at a time, on
the same 50 scenes (seed 4000, ids 0-49) so each rung can be compared pair-by-pair against
the `--ood none` run (exact McNemar on the discordant pairs).

All results are **simulation only**.

## 1. Training ranges (what "in distribution" means)

Source: `datagen/r2s/variations.py` (`sample_variation`, with `variety=True`, as datagen and
the eval harness both call it). The MolmoAct2 training set `sugar_v10_n1000_molmoact2` used
datagen seed 1; eval seeds 3000/4000 draw from the **same** ranges.

| Factor | Training range | Code |
|---|---|---|
| Cube colour | HSV: hue 0-1 (any), saturation 0.50-0.95, value 0.50-0.95 | `object_rgb` |
| Bin colour | HSV: hue 0-1 (any), saturation 0.30-0.90, value 0.30-0.90 | `bin_rgb` |
| Table colour | 2-colour checker, same hue; hue 0-1, saturation 0.00-0.35, value 0.35-0.90 (2nd square value +-0.12, clipped 0.2-1.0) | `table_rgb1/2` |
| Table texture | MuJoCo built-in checker only, repeat 2-12 per side | `table_texrepeat` |
| Key light | directional, elevation 35-80 deg, any azimuth, diffuse 0.35-0.90 | `light_dir`, `light_diffuse` |
| Ambient | 0.08-0.35 | `ambient` |
| Headlight | diffuse 0.15-0.50 | `headlight` |
| Front camera height | nominal z 0.34 m, +-8 cm (`cam_dz`) | `cam_pos` |
| Front camera x/y | +-3 cm each, aim point moves with it (so at most ~+-3.4 deg of implicit yaw at 0.53 m) | `cam_pos`, `cam_lookat` |
| Front camera pitch | +-8 deg about its horizontal axis (`cam_dpitch_deg`), after re-aiming at the nominal point | `cam_lookat` |
| Front camera yaw | **not randomised explicitly** (only the implicit ~+-3.4 deg above) | - |
| Front camera distance | **not randomised explicitly**; height and x/y change it by roughly +-5 cm around 0.53 m | - |
| Front camera FOV | 45 deg +-3 deg | `cam_fovy` |
| Wrist camera | fixed to the gripper (Menagerie `wrist_cam`), never randomised | - |
| Image exposure | **none**: frames are rendered (quality "fast", no shadows) and h264-encoded, nothing else | - |
| Other objects on the table | **none**: only the cube and the bin | - |
| Cube | 20 mm +-3 mm box, mass x0.61-1.65, friction 0.4-1.0 | physics, unchanged by the ladder |

Nominal front camera: position (0.56, -0.24, 0.34) m, looking at (0.17, 0.00, 0.03) m in the
robot base frame, ~0.53 m away.

## 2. The rungs (`eval/sim_eval.py --ood <rung>`)

`--ood` edits the Variation **after** `sample_variation`, with its own random stream
(`numpy.random.default_rng([episode_seed, <rung tag>])`), so:

* `--ood none` gives byte-identical Variations to the harness before this change (tested);
* every rung changes only its own factor; the physics draws (cube pose, mass, friction,
  servo gains, start pose, expert tempo) stay identical to `none` on the same id;
* `combo` = exactly the `camera` draws plus exactly the `colour` draws of the same episode.

Each episode records what was changed under `variation.extras.ood` in `episodes.jsonl`.
The camera is fixed within an episode (no shake or drift).

| Rung | What changes (per episode, random sign / choice, deterministic from the episode seed) | vs training |
|---|---|---|
| `none` | nothing | in range |
| `camera` | front camera height +-12-16 cm and pitch +-12-15 deg (both signs random), FOV 51 deg; the training x/y draw is kept | height 1.5-2x, pitch 1.5-1.9x, FOV 2x the band |
| `camera_height` | height only: +-12-16 cm | 1.5-2x |
| `camera_pitch` | pitch only: +-12-15 deg | 1.5-1.9x |
| `camera_yaw` | yaw only: camera orbits the aim point about the vertical axis by +-10-15 deg | 3-4x the implicit +-3.4 deg |
| `camera_distance` | distance only: camera moves along its view axis by +-8-12 cm (+ = farther) | ~1.6-2.4x the implicit +-5 cm |
| `exposure` | one image factor per episode, applied to every rendered frame of **both** cameras before the h264 round-trip (i.e. "in the sensor"): brightness x0.5 or x1.6, or contrast x0.6 or x1.4 (pivot mid-grey 0.5), or white-balance gains R,B in {0.8, 1.2} (G = 1) | training had none |
| `colour` | cube magenta (HSV 0.83, 1.0, 1.0), bin near-black (value 0.05), table saturated green checker (saturation 0.85, values 0.55/0.45); lights either dim (0.5x the training minima: diffuse 0.175, ambient 0.04, headlight 0.075) or bright (1.6x the maxima: diffuse 1.44, ambient 0.56, headlight 0.80), 50/50 | see note |
| `combo` | `camera` + `colour` (no distractor) | |
| `distractor` | 1-2 extra free bodies (cylinder or sphere, 20-35 mm, colour hue >= 0.25 away from the cube's), >= 60 mm from the cube centre and from the bin's outer walls, >= 50 mm apart, within the arm's reach band; ordinary contacts (no weld, no mocap). New verifier clause `wrong_object_in_bin` (task failure) | training had none |

Notes.

* **Colour.** Training cube hues cover the whole circle, so a cube cannot be outside in hue;
  magenta at saturation 1.0 / value 1.0 is outside in saturation and value only (just above
  the 0.95 maxima). The large shifts are the table (saturation 0.85 vs at most 0.35) and the
  bin (value 0.05 vs at least 0.30). No new texture: skipped to stay inside the time box.
* **Expert ceiling.** The scripted expert plans from the true state and never looks at an
  image, so on the camera, exposure and colour rungs its ceiling only checks that the scene
  still builds and the verifier still agrees. Only the distractor rung can change physics.
* **Camera visibility.** Checked by eye on the 4x4 contact sheets in
  `/work/outputs_large/eval/ood_sheets/<rung>.png` (front camera, first frame of ids 0-15).
