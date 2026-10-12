# Plan: object from video (v2), one page

> **PROPOSAL, NOT ACTIVE.** Build this after the weekend run. Written 9 Oct 2026.
> Facts are tagged [V] when checked on the primary page and [I] when inferred; sources are at the bottom.

## Why

Today the simulator uses a cube sized from the video, so the engine only handles simple shapes. R2R2R needs a separate phone scan of the object.

Turning **the same single video** into a textured 3D object that the simulator can grasp gives three things:
- real shapes (mug, toy, sponge);
- renders that look more like the real camera;
- a clean difference from R2R2R: **one video, no scan, physics-checked**.

## Pipeline

1. **Best frame + mask.** Pick the frame where the object is largest and least hidden. SAM 2 already gives the mask.
2. **Image → 3D.** Produces a textured mesh (GLB/OBJ). No model gives real-world (metric) scale [I].
3. **Scale.** Multiply the mesh so one measured dimension matches: the A4 sheet, or a ruler. Use MJCF `<mesh scale="s s s">` [V].
4. **Make it physical.**
   - Keep the textured mesh as the **visual** geom only (`contype="0" conaffinity="0"`).
   - Build a closed copy for **collision** (alpha-wrap repair), then split it into convex parts with **CoACD** (MIT) [V]. MuJoCo collides each mesh as its convex hull, so a mug needs several parts [V].
   - Package it with **obj2mjcf** (MIT), which can call CoACD [V].
   - Mass comes from a kitchen scale. Inertia uses `inertia="convex"` for solid objects and `"shell"` for thin hollow ones [V]. Inertia is the least reliable number: one published pipeline reports roughly 359% inertia error [V].
5. **Same gates as now.** The existing grasp-physics test must pass on the new object before any demo is generated.

## Model choice (in this order)

| Step | Model | Licence | Runs where | Why / risk |
|---|---|---|---|---|
| 0 Baseline | **TripoSR** | MIT, ungated [V] | Laptop RTX 4060 (about 6 GB) [V]; MI300X with its compiled marching-cubes step swapped for a CPU one [I] | Fewest dependencies. Coarse geometry and a smoothed back side. |
| 1 Main | **Pixal3D** via `Pixal3D.cpp` (ggml) | MIT code and weights; the bundled DINOv3 encoder has its own licence [V] | MI300X through ROCm. Reported working on other AMD GPUs, untested on MI300X [I] | Built to keep the visible front faithful [V]. About 12 GB of weights. |
| 1b Fallback | **TRELLIS.2** via `trellis.cpp` | MIT [V] | ROCm/HIP supported, tested only on a Strix Halo iGPU [V] | High quality, but slow on small GPUs |
| Skip | SAM 3D Objects | Gated, NVIDIA GPU with ≥32 GB required [V] | — | Doesn't fit our hardware |
| Skip | Hunyuan3D 2.1 | Excludes the EU/UK/Korea and bans training other models on its outputs [V] | — | Licence |
| Skip | SF3D, InstantMesh | SF3D is gated under a community licence; InstantMesh uses old CUDA pins [V] | — | Friction for little gain |

## Validation (pass marks fixed in advance)

| Check | Pass mark |
|---|---|
| **Size accuracy** on 3 objects measured with a ruler or calipers (cube, can, mug) | Bounding-box error ≤5% or ≤3 mm, whichever is larger. Also report the volume ratio. |
| **Settle test**: drop the object on the table and wait 2 s | No jitter or sinking; resting pose matches the photo |
| **Grasp parity**: same seeds and poses on the true-shape model and on the reconstructed mesh | Success within 10 points over 50+ randomised poses, reported with a confidence interval, plus slip and lift |
| **Scale sensitivity**: rescale by ±3% and ±5% | Report which scale breaks first. This shows how much error the SO-101 jaw tolerates. |
| **Back side** (never seen by the camera): mug handle facing away versus facing the camera | Report it as a known limit |

## Effort and compute

About 1–2 working days after the weekend run. Each model run takes seconds (TripoSR) to minutes (Pixal3D), so GPU cost is small. The laptop can do step 0. The main model needs the MI300X, or an NVIDIA card with 24 GB or more.

## Closest prior work

**HUG-Bench** (arXiv 2606.17054) is the nearest match [V]:
- SAM 3D from **five views**, scaled **by hand**, alpha-wrapped, CoACD, exported to MuJoCo.
- 73% policy success.
- It does not report how accurate the reconstructed sizes are.

What would be new here: **one video, automatic scale, and a measured grasp-parity gate**. That is unverified until a fuller literature search.

## Risks

- The hidden back side is guessed.
- Thin parts (handles) get lost.
- ROCm builds fail on MI300X. Fall back to TripoSR on the laptop.
- Textures look plausible but not real. The JEPA realism score measures whether the renders got closer to the real camera.

**Sources:** TripoSR, Pixal3D, `Pixal3D.cpp`-GGUF, TRELLIS.2, `trellis.cpp`, SAM 3D Objects, Hunyuan3D-2.1, CoACD, obj2mjcf, MuJoCo XML reference, HUG-Bench (2606.17054), Scalable Real2Sim (2503.00370), R2R2R (2505.09601). Full URLs are in the research notes, under `docs/research/` once committed.
