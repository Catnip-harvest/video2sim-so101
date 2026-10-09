# Third-person demo to head/wrist-camera view?

**Short answer:** it can be done as a generated *picture*, but the geometry can't be trusted, and it does not help this extractor. The research term is exocentric-to-egocentric ("exo→ego") view translation.

## What exists (sources read 2026-10-09)

- **Exo2Ego** ([ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5558_ECCV_2024_paper.php)) works from one third-person video and produces first-person video.
  - It first maps structure across the two views, then paints pixels with diffusion guided by a hand-layout prior.
  - Trained on H2O, Aria Pilot, Assembly101 and Ego-Exo4D.
- **Exo2Ego-V** ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f5a8b5e5d007e66c929b971c2bc21d76-Abstract.html)) needs four third-person views placed around the scene.
  - Diffusion model on Ego-Exo4D, about 35% better LPIPS than earlier methods.
- **From Synchrony to Sequence** ([arXiv 2604.13793](https://www.alphaxiv.org/overview/2604.13793)) needs camera poses for both views.
  - Reaches about 16–17 dB PSNR on Ego-Exo4D.
  - Fails where the third-person camera never saw the scene.

## Does it help here? No.

1. **It adds no measurement.** We need metric object poses, and the A4 sheet plus the third-person view already give them. On the synthetic test that is about 2 mm pick error and about 2 cm place error. A generated first-person view redraws the same pixels, with hallucinated content at roughly 16 dB PSNR.
2. **It is the wrong camera.** These models output a human head view, not an SO-101 wrist camera, and need extra views or camera poses we don't have.
3. **There is a better route to wrist images.** Replay the extracted object path in mjlab and render the robot's own wrist camera, which is exact by construction.

**Revisit only if** a wrist-camera visual policy needs human-appearance pretraining data and a single-view method ships code.
