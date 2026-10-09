# video2sim-so101

**One human video in; a physics-checked robot dataset and policy out.** The target robot is the LeRobot SO-101 arm.

I film myself picking up a small object and dropping it into a bowl. The pipeline:

1. Extracts **what happened to the object**: its path, plus when it was gripped and released. It does not copy my hand.
2. Replays that task thousands of times in MuJoCo on an SO-101, varying the object, the motors, the lighting and the camera height.
3. Keeps only episodes where **real physics** says the grasp was genuine: both jaws carry the object, nothing else touches it, it lands in the bowl and is released.
4. Writes a LeRobot dataset.
5. Uses that dataset to fine-tune a pretrained VLA (π0.5), and improves the policy with a practice round in simulation that keeps only its own physics-checked successes.

Status: work in progress, as of the 9 Oct 2026 sprint on one AMD Instinct MI300X.

## Why it is built this way

Every part comes from published work; the new part is the glue between them.

| Piece | Source |
|---|---|
| Follow the **object**, not the hand: a human arm and a table-mounted arm move differently | X-Sim (Cornell, 2025) |
| One video → many synthetic demos; smooth and interpolate the extracted path | Real2Render2Real (R2R2R) |
| Randomisation ranges for the SO-101 (gripper stiffness, joint noise) | Squint (arXiv 2602.21203; 73/80 real SO-101 transfers) |
| SO-101 model | MuJoCo Menagerie `robotstudio_so101` (Apache-2.0, commit 0059d43) |
| Base policy | π0.5 via LeRobot (`lerobot/pi05_base`) |

Full reasoning: [docs/DESIGN.md](docs/DESIGN.md).

## Layout

| Folder or file | What it holds |
|---|---|
| `demo/` | Video → task spec. MediaPipe hand ONNX, SAM 2.1 object track, Depth Anything V2 Small, A4-sheet calibration. Run `python demo/tools/fetch_models.py` to download the model weights. |
| `datagen/` | Task spec → N physics-verified SO-101 episodes → LeRobot v3 dataset (`run_droplet.sh`). |
| `task/` | mjlab RL task `SO101-PickPlace-State`, and `tests/test_grasp_physics.py`: the gate that rejects fake grasps. |
| `third_party/menagerie_so101/` | Vendored Menagerie model, Apache-2.0, LICENSE inside. |
| `env/` | Everything needed to set up the AMD MI300X droplet. |
| `CLAUDE.md`, `HANDOFF.md` | Rules and current plan for the agent running on the droplet. |

## Quick start on an AMD MI300X droplet

Use AMD Developer Cloud with the "PyTorch 2.10.0 – ROCm 7.2" image.

```bash
git clone <this repo> /root/work/video2sim-so101
bash /root/work/video2sim-so101/env/start_container.sh
docker exec sprint bash /work/video2sim-so101/env/setup_container.sh
docker exec -it sprint hf auth login      # once; your own token, typed by you
cp .env.example .env                      # then edit the repo names
docker exec sprint bash -lc "cd /work/video2sim-so101 && INSTALL_LEROBOT=0 PY=python N=20 SPECS=sugar bash datagen/run_droplet.sh"
```

**Notes for this machine**

- **Rendering is CPU-only.** It runs through OSMesa; EGL does not work here.
- **Protect the ROCm build of PyTorch.** Install every Python package with `PIP_CONSTRAINT=env/constraints.txt`, so pip never swaps it for a CUDA build.

## Measured so far

| Check | Result |
|---|---|
| Grasp-physics gate | 12 pick cases and a negative control pass; contact force carries 0.293 N against a 0.294 N object weight |
| Generator, local runs (50 episodes each) | 50/50 accepted (sugar-cube spec), 50/51 (synthetic spec) |
| Generator ablation: step servo commands | Acceptance drops to 72.5%, mostly dropped cubes. The verifier does reject failures. |
| Demo extractor on a synthetic clip with ground truth | Pick position off by 1.5 mm, place position off by 2 cm |

## Limits, stated honestly

- **The stock-clip demo has no calibration sheet,** so its scale is uncertain by about ±20%.
- **The generated demos all follow one scripted pattern;** variety is being added.
- **Simulation to real hardware is untested.**

## Licences

- **This repo's code:** MIT, to be confirmed.
- **Menagerie model:** Apache-2.0.
- **Pexels test clip:** Pexels License; source in `demo/third_party/`.
- **Model weights** are not included. They are fetched from their own repos under their own licences.
