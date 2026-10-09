# object3d: make any mesh simulable, then prove the physics path on real scans

This is the physics half of "object from video (v2)" (`docs/proposals/object-from-video.md`
on branch `proposal/object-from-video`). There is **no image-to-3D model here yet**. The
point of this step is that *any* mesh, including one a model will produce later, can be
turned into a MuJoCo object, checked, and used by the data generator. The check runs first
on ground-truth scans (YCB) so that later failures can be blamed on reconstruction, not on
the physics path.

Status, 9 Oct 2026 (laptop, CPU MuJoCo 3.11, no rendering): the converter, the harness and
the datagen hook all work. A 30 mm cube that goes through the mesh path behaves like the
built-in cube (30/30 vs 30/30 on the same seeds). Of the three YCB objects, **only the
Rubik's cube can be grasped by the current top-down planner, and only half the time**. The
can and the mug cannot be grasped at all yet. The reasons are measured below.

## What each step does, and why

| Step | File | What it does | Why |
|---|---|---|---|
| 1. Convert | `mesh_to_mjcf.py` | Mesh (OBJ/GLB/PLY/STL) + one measured size (or a scale factor) + mass in kg → a folder with `object.xml`, the meshes and `report.json`. | The simulator needs a closed collision shape, a mass and a size in metres. A raw scan gives none of these reliably. |
| 1a. Turn and centre | | Turns the mesh about the vertical so its footprint's smallest rectangle lines up with x and y (longer side on x), then puts the body origin at the centre of the bounding box. | The grasp planner squeezes along the object's own x or y axis, and "half height" must be exact for the place check. |
| 1b. Scale | | One measured side in metres (`--target-size`, `--target-dim`) or `--scale`. Written as MJCF `<mesh scale="s s s">`, so the files stay in source units. | No image-to-3D model gives real size; one ruler or A4-sheet measurement fixes it. Rescaling is then one attribute (used by the scale test). |
| 1c. Repair | | Merges duplicate vertices (texture seams), drops zero-area faces and "fins" (the same triangle twice with opposite winding), caps open holes with a fan. Records whether the result is closed (`watertight`). | Volume, mass and convex decomposition need a closed surface. All three YCB scans needed this: as loaded, none was closed; after repair, all three were. |
| 1d. Convex pieces | | CoACD (MIT) splits the repaired copy into convex pieces. Fallbacks: V-HACD (`vhacdx`), then a single hull. A result must hold ≥ 90 % of the mesh volume or the next backend is tried. | MuJoCo collides each mesh as its convex hull. A mug as one mesh would be a solid lump. |
| 1e. Mass and inertia | | Mass split across the pieces by volume (`--inertia convex`, solid objects) or by surface area (`shell`, thin or hollow objects). `auto` = convex if the repair closed the mesh. | Uses MuJoCo's own `<mesh inertia="convex/shell">` modes. Inertia is the least trusted number (see Limits). |
| 1f. Visual mesh | | The original textured mesh is kept as a geom that is only drawn: `contype=0 conaffinity=0`, group 1, mass 0. | Renders keep the scan's look; it never touches anything and weighs nothing. |
| 2. Datagen hook | `datagen/r2s/mesh_object.py` + small hooks in `scene.py`, `planner.py`, `rollout.py` | A task spec may say `"object": {"kind": "mesh", "mjcf": "...", "mass": ...}`. The scene attaches the mesh body; the planner and the verifier read its half sizes instead of the cube size. The cube stays the default and its code path is unchanged (all 12 existing tests pass). | Lets the existing physics-verified generator, with its unchanged verifier, run on any object. |
| 3. Validate | `validate_object.py` | `settle`, `parity`, `scale`, or `all`. | Pass marks fixed before the runs, from the proposal. |
| 4. Fetch scans | `fetch_ycb.py` | Downloads three YCB scans, checks sha256, unpacks with the safe tar filter. | Ground truth to test against. Licence: CC BY 4.0 (see `third_party/README.md`). |

### The three validation tests

- **Settle.** Drop the object 5 mm onto the table and wait 2 s. Pass marks: resting height
  within 2 mm of its half height, penetration ≤ 1 mm, in the last 0.5 s speed < 5 mm/s and
  spin < 3 deg/s, drift over the last 1 s < 1 mm, tilt < 3°.
- **Grasp parity.** Run the existing scripted generator (`datagen/r2s/rollout.py`, the same
  verifier that gates every dataset episode) on the same seeds with object A and object B.
  A seed fixes the pose, servos, friction and mass factor, so only the object differs.
  Reported per object, with 95 % Wilson intervals:
  - grasp + lift (both jaws on it and ≥ 1 cm up after the lift);
  - carried without a drop;
  - full episode accepted;
  - in-hand slip and lift height;
  - the paired disagreements.

  Pass mark: within 10 points.
- **Scale sensitivity.** The same batch with the object scaled ×0.95, 0.97, 1.03 and 1.05.
  The mass stays the same, because a kitchen scale is right even when the size estimate is
  wrong.

All runs use the sugar-cube task spec (`demo/outputs/pexels_7488188_sugar_cube`), seed 0 and
30 seeds per object or scale. The pass marks for size accuracy and for the unseen back side
need a reconstruction, so they belong to the next step.

## Commands

```bash
PY=/c/Users/vieth/AppData/Local/Programs/Python/Python313/python.exe   # mujoco 3.11, trimesh, coacd
$PY -m pip install trimesh coacd          # MIT; vhacdx (MIT) optional
$PY -I object3d/fetch_ycb.py              # 24 MB of archives -> object3d/third_party/ycb (gitignored)
$PY -I object3d/mesh_to_mjcf.py object3d/third_party/ycb/025_mug/google_16k/textured.obj \
    --out object3d/out/ycb_025_mug --mass 0.118 --target-size 0.082 --target-dim height --name ycb_025_mug
$PY -I object3d/validate_object.py all object3d/out/ycb_077_rubiks_cube --ref box:0.057:0.094 \
    --n 30 --workers 5 --out object3d/results/ycb_077_rubiks_cube
$PY -I -m pytest -q -p no:cacheprovider object3d/tests     # 12 tests, ~15 s
```

`--plan '{"open_max_rad": 1.6, "grasp_height_max": 0.03}'` passes planner overrides (both
default to the cube planner's behaviour). Generated objects go to `object3d/out/`, which is
gitignored. Reports go to `object3d/results/`, which is committed.

## Measured numbers (laptop, 9 Oct 2026)

### Conversion (CoACD, threshold 0.10)

| Object (official mass, size) | Scale source → factor | Size after (mm) | Closed after repair | Volume (cm³) | Pieces | Pieces / mesh volume | Time |
|---|---|---|---|---|---|---|---|
| synthetic 30 mm cube (from mm) | x = 30 mm → 0.001 | 30.0 × 30.0 × 30.0 | yes | 27.0 | 1 | 1.00 | 1.8 s |
| 077 Rubik's cube (94 g, 57 mm) | height 57 mm → 0.984 | 58.1 × 57.1 × 57.0 | yes | 174.1 | 21 | 1.02 | 61 s |
| 005 tomato soup can (349 g, 66 × 101 mm) | height 101 mm → 0.992 | 67.0 × 66.9 × 101.0 | yes | 311.4 | 9 | 1.08 | 54 s |
| 025 mug (118 g, 80 × 82 mm) | height 82 mm → 1.009 | 106.2 × 93.8 × 82.0 | yes | 141.7 | 32 | 1.69 | 275 s |

- **Scale.** The YCB scans are already in metres, so the factors are within 1.6 % of 1. That
  gap is the scan-versus-official-size error.
- **Mug footprint.** The mug's narrowest footprint width is 92.5 mm (handle side excluded).
  The YCB sheet's "80 mm" is presumably the cup body without its rim flare or handle.
- **Mug pieces.** The pieces hold 1.69× the mug's material volume. Convex pieces of a thin
  wall include some air. The mass is still exactly 118 g, but its centre of mass and inertia
  are approximate.
- **Decomposition time.** CoACD at threshold 0.05 gave 65 pieces for the cube (82 s) and
  37 for the can (96 s). A 32-piece cap made CoACD merge pieces, which took 160–230 s. 0.10
  with no cap is the default.

### Mesh path vs built-in cube (same 30 seeds)

| | grasp + lift | full episode accepted | in-hand slip median / p90 | lift |
|---|---|---|---|---|
| built-in 30 mm cube (`box:0.030:0.025`) | 30/30 (100 %, CI 89–100) | 30/30 | 0.20 / 0.30 mm | 60.1 mm |
| same cube through `mesh_to_mjcf` | 30/30 (100 %, CI 89–100) | 30/30 | 0.16 / 0.28 mm | 60.0 mm |

Paired: the two agree on all 30 seeds. **The mesh path itself does not change the result.**

### Settle (2 s on the table)

| Object | Result | height err | speed / spin, last 0.5 s | drift, last 1 s | jitter peak-to-peak |
|---|---|---|---|---|---|
| built-in 30 mm cube | PASS | −0.11 mm | 0 / 0 | 0 | 0 µm |
| 30 mm cube via mesh | PASS | −0.11 mm | 0.35 mm/s / 0.9 deg/s | 0.001 mm | 8 µm |
| Rubik's cube | **FAIL** (spin) | −0.19 mm | 1.9 mm/s / **3.8 deg/s** | 0.002 mm | 18 µm |
| tomato soup can | **FAIL** (speed, spin) | −0.19 mm | **7.7 mm/s / 8.3 deg/s** | 0.019 mm | 65 µm |
| mug | PASS | +0.36 mm | 1.4 mm/s / 1.9 deg/s | 0.005 mm | 17 µm |

- **What the failures are.** None of the objects sinks or wanders: drift is ≤ 0.02 mm and
  tilt ≤ 0.13°. The failures are contact chatter, where the velocity flips sign every few
  steps and the body moves 18–65 µm.
- **What would fix it.** Switching to pyramidal friction cones removes it (all three YCB
  objects pass, speed ≤ 1 mm/s). More solver iterations, condim 3/4/6 and torsional friction
  do not.
- **Why it is not switched.** The cube gate and every dataset so far use elliptic cones.
  Changing the cone type for mesh objects is a decision for the owner. The marks were not
  loosened.

### Grasp batch with the current top-down planner (30 seeds each)

| Object | grasp + lift | full episode accepted | What fails |
|---|---|---|---|
| Rubik's cube, YCB scan | 15/30 (50 %, CI 33–67) | 15/30 (50 %, CI 33–67) | `grasp_failed` 13, `dropped_or_slipped` 2 |
| Rubik's cube as a 57 mm primitive box (parity reference) | 11/30 (37 %, CI 22–55) | 10/30 (33 %, CI 19–51) | `grasp_failed` 19, `ik_unreachable` 1 |
| tomato soup can | 0/30 (0 %, CI 0–11) | 0/30 | `ik_unreachable` 30 |
| mug | 0/30 (0 %, CI 0–11) | 0/30 | `ik_unreachable` 19, `grasp_failed` 11 |

- **Rubik's cube parity: FAILS the 10-point mark.**
  - The scan does better than the box: 50 % vs 33 %, a 17-point gap. On the same seeds,
    5 succeeded only with the scan and none only with the box.
  - Both are poor because 57 mm is outside the 20–50 mm band this jaw was validated for.
    In-hand slip is 5.7 mm (scan) and 8.2 mm (box), against 0.2 mm for the 30 mm cube.
  - Success depends on servo torque. With the 12 V servo: 10/15 (scan) and 8/15 (box). With
    the kit's 7.4 V servo: 5/15 and 2/15.
  - The scan's rounded edges probably help. That is not proven.
- **Rubik's cube scale sensitivity.** Grasp + lift success at each scale:

  | Scale | ×0.95 | ×0.97 | ×1.00 | ×1.03 | ×1.05 |
  |---|---|---|---|---|---|
  | Success | 57 % | 57 % | 50 % | 33 % | 17 % |

  Fully accepted episodes follow the same trend: 47 %, 53 %, 50 %, 33 %, 13 %. **Too big
  breaks first.** +3 % already costs 17 points and +5 % costs 33. Shrinking by up to 5 %
  costs nothing. For a 57 mm object near the jaw's limit, the scale estimate must not be
  high by more than about 2 %.
- **Can and mug, scale sweep.** 0/30 at every scale. The cause is the planner, not the size.

### Why the can and the mug cannot be grasped, and what the planner needs

1. **The approach point is out of reach.**
   - The planner grasps at the object's mid-height and starts its straight-down approach
     7 cm above that: 121 mm for the can and 111 mm for the mug. With the fingers pointing
     straight down, the arm misses those points by 45–63 mm.
   - Measured at the can's position (r ≈ 0.2 m): the arm reaches a vertical-finger pose
     only up to about 0.09 m site height. At 0.10 m it is already 11 mm short.
   - Starting lower does not help. A straight-down approach must begin above the object's
     top, about 101 + 15 + 10 mm for the can, and that is out of reach.
   - **Needed:** a side grasp. The fingers point horizontally and come in along the arm's
     radial direction, low on the object. A 5-joint arm can tilt its fingers only in its own
     vertical plane, so the approach must be radial.
2. **The mug is wider than the opening the planner commands.**
   - The planner opens to at most 1.0 rad, an 87 mm mouth. The mug is 93.8 mm across its
     body and 106 mm across the handle.
   - In a probe with the opening raised to 1.6 rad and a lower approach (10 seeds), 6 plans
     went through, but none lifted. The best lift was 8.6 mm, and two runs touched the mug
     with the arm.
   - At 1.36–1.48 rad, the moving jaw's tip is 55–65 mm above the fixed jaw's tip, so the
     jaws close as a wide V and pinch near the top.
   - The planner also picked the 106 mm handle axis in 4 of those 6 plans, because it
     chooses the side nearest the arm's natural closing direction.
   - **Needed:** choose the narrowest side first and skip sides wider than the jaw. Better,
     use a **rim grasp**: one jaw inside the cup, one outside, across a wall a few mm thick.
     A **handle grasp** is the other option. Both need a new grasp-point rule, not just new
     sizes.
3. **The can's weight is still untested.** At 349 g, it is 14× the cube's 25 g. Whether the
   jaw (0.8–1.5 N·m) can hold it is unknown until a side grasp exists.

## Contact findings (why mesh scenes differ slightly from cube scenes)

These are on by default for `kind: "mesh"` only (`scene.enable_mesh_contacts`, `scene.add_table`):

- **The object rests on a box table top, not on the plane.**
  - MuJoCo gives a plane-against-mesh pair only 3 contacts. A 30 mm single-piece cube then
    rocked forever, at 25 mm/s and 68 deg/s in the last 0.5 s.
  - The plane is still there for the robot and for rendering. A hidden 1 cm box top, which
    only world objects touch, carries the object. With it the cube passes settle.
- **multiccd stays on.** With it off, box-against-mesh gives one contact point.
- **Native CCD is off.**
  - With MuJoCo's native CCD, the mesh cube ended 2–4 mm deep in the 4 mm bin floor, in a
    jitter cycle. Only 3 of 8 episodes were accepted, all as `not_resting` failures.
  - With the libccd path, 8/8 were accepted.
- **Memory.**
  - Each episode leaves its model copies in reference cycles. With mesh objects that is
    about 250 MB per episode until Python's cycle collector runs.
  - Six workers exhausted Windows' commit limit, and qhull's `malloc` failed inside MuJoCo's
    compile. Workers died silently and the pool hung for 40 minutes.
  - Now `gc.collect()` runs after every episode (harness and `generate.py`), and harness
    workers restart every 10 episodes.

## Environment notes (Windows laptop)

- **CoACD.** Windows Application Control blocked CoACD's DLL on the first import
  (`WinError 4551`), then allowed it on later imports. Every run above used CoACD.
- **V-HACD.** `vhacdx` 0.1.0 loads but returns broken pieces. A unit box came back as a
  tetrahedron, and a sphere as 39 % of its volume. On the scans, the pieces held 11–21 % of
  the volume. The coverage check rejects it.
- **Licences.** No GPL code is used: no PyMeshLab, no CGAL. Licences: trimesh MIT, CoACD
  MIT, vhacdx MIT (V-HACD BSD-3), MuJoCo Apache-2.0, YCB data CC BY 4.0.
- **Disk.** About 106 MB in total: 24 MB of archives, 59 MB unpacked, 23 MB of generated
  objects, all gitignored. Results add 0.4 MB.

## Limits, stated honestly

- **Ground truth only.** Size accuracy and the unseen back side need the image-to-3D step.
- **One task spec.** Each batch is 30 seeds of one task spec, so the confidence intervals
  are ±15–20 points.
- **Inertia is approximate.**
  - The mass is exact.
  - The inertia comes from the convex pieces, so concave objects (the mug) carry too much
    of it far from the centre.
  - `shell` mode exists but was not needed: all three scans closed after repair.
- **Reference shapes.** The parity reference for the Rubik's cube is a sharp-edged box. The
  can and the mug have no primitive reference, because a datagen cylinder is not wired.
- **The mjlab RL task** (`task/`) does not know `kind: "mesh"`. Only CPU datagen does.
