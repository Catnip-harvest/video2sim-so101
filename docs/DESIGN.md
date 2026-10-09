# Watch me once: design (locked 9 Oct 2026)

One phone video of a person doing a pick-and-place becomes an SO-101 policy, trained in mjlab on one MI300X.
This is a portfolio sprint over one weekend. Everything is simulation; a real rollout on a friend's SO-101 is an optional follow-up.

## Prior work it is built on (checked 9 Oct)

No published pipeline goes from human video to an SO-101. We borrow the parts that worked elsewhere:

| Source | What we take | Licence |
|---|---|---|
| X-Sim (portal-cornell/X-Sim) | Object-centric rewards: the robot must move the **object** the way the human did. This avoids retargeting a shoulder-hung human arm onto a table-mounted arm. | No licence file, so we take the idea and write our own code |
| R2R2R (uynitsuj/real2render2real) | One RGB video plus object tracking. Smooth **and** interpolate the trajectory: removing interpolation took their mug task from 80% to 0%. | MIT |
| Squint (aalmuzairee/squint, arXiv 2602.21203) | SO-101 sim-to-real DR recipe (73/80 real, zero-shot): proprio noise σ≈5°, gripper stiffness 500–2000 and damping 50–200, colour jitter matters (−15/80 without it), tight camera DR | MIT |
| so-frame (livekit-examples/so-frame) | Read only: SO-101 in mjlab; wrist_roll offset +90°; delta-on-previous-target actions | No licence, so reference only |
| Human2Sim2Robot (tylerlum/human2sim2robot) | Reward and tracking design; warning that RL exploits sim friction | MIT |

## Pipeline

1. **Demo extraction** (laptop, `demo/`):
   - Hand keypoints: MediaPipe.
   - Object mask: SAM 2.
   - Metric table frame from an A4 sheet.
   - Output is an object trajectory, smoothed and interpolated, plus grasp/release timing and pinch width from the thumb and index finger.
   - The hand path is **not** retargeted.
2. **Sim task** (`task/`, mjlab):
   - Robot: table-mounted SO-101.
   - Reward: track the object along the demo trajectory, plus reach/grasp/place shaping.
   - Actions: delta joint targets at 10 Hz, capped at real servo speed.
   - Domain randomisation: Squint's ranges, gains ±50%, 0–1 step latency, object-pose observation noise with dropouts.
   - Hard gate: a physics-honest grasp test (no welds or latches).
3. **Train**: state-based PPO on the MI300X (MuJoCo Warp ROCm fork). Judge in a second simulator, CPU MuJoCo.
4. **Progress judge (the JEPA ask)**:
   - Use a frozen scorer for task progress and success on both the human video and the robot rollouts.
   - Robometer-4B (in LeRobot; reported 0.916 progress correlation on real SO-101 data) is the verified option.
   - A frozen V-JEPA 2 ViT-L (or DINOv2) feature probe is the JEPA comparison, at 1–2 GPU-h.
   - No JEPA planning, no JEPA as the RL reward, and no JEPA training. None of these fits in a weekend.

## Camera height

- The state policy uses no camera, so camera height does not affect it.
- The real deploy path gets object xy from an AprilTag or colour blob plus a table homography, so any camera height works.
- If a vision student is added later: published low-cost-arm setups that transferred used **tight** camera DR (±2–2.5 cm, FOV ≤2–5°) around a camera pose that had been measured and aligned. We randomise height within a band around the measured pose, and evaluate outside the band to show robustness.
- Rendering on ROCm is CPU-only (no GPU batch renderer). Measure fps first.

## Out of scope this weekend

- Cloth or dish washing.
- Exo-to-ego video generation; lifting to 3D is viewpoint-free.
- Hand-to-gripper reward from embeddings.
- V-JEPA 2-AC planning on SO-101.
