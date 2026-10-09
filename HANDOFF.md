# Handoff: "1 video → robot dataset → policy" sprint (written 9 Oct 2026)

## Goal (locked by the owner)

**A. Data engine.** One human video in, a physics-checked SO-101 dataset out. A JEPA model then scores it as a second judge.

**B. Fine-tune π0.5.** The base model (`lerobot/pi05_base`) is already downloaded. Train it on the engine's data plus the friend's real episodes, then measure it in simulation.

**B+. Practice round in simulation.** This is a light, reinforcement-learning-style loop:
1. Run the fine-tuned policy many times in simulation.
2. Keep only the attempts the physics verifier passes.
3. Fine-tune on those again. This is filtered behaviour cloning.

Online reinforcement learning only happens if `FINAL_DESIGN.md` says it is feasible. That file is coming from the laptop session.

**Deliverables**
- A Hugging Face dataset.
- The fine-tuned policy.
- Simulated success rates over 50 or more randomised trials.
- A side-by-side video: the human clip next to robot episodes.
- A plain-language README for the owner.

## What already exists in /work/video2sim-so101

- **`demo/`**: turns a video into a task spec, meaning the object's path plus when the grasp and release happen.
  - The specs are in `demo/outputs/*/so101_task_spec.json`.
  - The sugar-cube spec comes from a stock clip with no A4 sheet in it, so its scale is uncertain by about ±20%.
- **`task/`**: the mjlab training task `SO101-PickPlace-State`, plus `tests/test_grasp_physics.py`, which passes.
- **`datagen/`**: the generator.
  - Run it with `bash datagen/run_droplet.sh`.
  - Environment variables: `INSTALL_LEROBOT=0`, `PY=python`, `N`, `SPECS=sugar|synthetic`, `WORKERS`.
  - Laptop results: 100% and 98% of episodes accepted.
  - It writes LeRobot v3 format, but that is not yet checked by actually loading it with `LeRobotDataset`. Check with `validate_dataset.py --lerobot`.
  - A rendered test run with N=20 started here at about 09:10 UTC. Its log is `/work/logs/datagen_n20.log`.
  - **Known weakness:** every demo follows one scripted pattern. Add variety (random approach angle, speed, pauses) before the full N≈1000 run.
- **`DESIGN.md`**: why the pieces are built this way. Sources: X-Sim, R2R2R, Squint, and the MuJoCo Menagerie arm model.
- **`friend_data/DIAGNOSIS.md`**: analysis of the friend's dataset.

## The friend's real data

**Source**
- `MinhDuk/final_dataset_annotated`, already downloaded into `HF_HOME`. Apache-2.0 licence.
- 524 episodes at 30 fps.
- Joint positions are in LeRobot degree units. The gripper uses a 0–100 scale.
- Cameras: `cam_head` and `cam_right`.
- About 334 episodes are the table-to-bowl task: a green cube into a white bowl.

**Before mixing it with simulated data**, convert it to the simulator's radians and check the conversion:
- Check each joint's zero point and direction. A +90° offset on `wrist_roll` has been reported for this arm model.
- Check that the gripper position computed from the joints at grasp time matches where the cube appears in the camera.
- Never mix the two datasets without that check.

## Order, gates and budget (stop and report when a gate fails)

1. **Render check.** Measure frames per second on OSMesa with 16–19 workers, and confirm the dataset loads in `LeRobotDataset`.
2. **π0.5 smoke test.** Run `lerobot-train` with the pi05 policy for a few steps on the friend's data or our simulated data.
   - Check that the loss is finite.
   - Record peak memory and steps per second.
   - From those, estimate how many hours a full run takes.
3. **Full data generation.** Run N≈1000 episodes with the added variety, then push to Hugging Face.
4. **JEPA judge.**
   - Model: a frozen V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`) with a small classifier trained on its features. Compare against Robometer-4B (`lerobot/robometer-4b`).
   - Calibrate it against the physics labels. That needs failed episodes as negatives, so record the rejected episodes too.
   - Report the AUC.
   - Then use it only where no physics label exists: the friend's real episodes and real rollouts.
5. **π0.5 fine-tune** on simulated data plus the converted real table-to-bowl episodes. Budget about 6–12 GPU-hours. Evaluate on 50 or more rendered simulation trials.
6. **B+.** Roll out, keep the verified successes, fine-tune one more round, and evaluate again.
7. **Package.** README, video, Hugging Face uploads and the cost ledger. Then tell the owner to DESTROY the droplet.

**Budget:** stay within about 40 hours of the droplet being alive in total. It was created at about 08:57 UTC on 9 Oct.

## Notes added at 09:11 UTC on 9 Oct

- **Two copies of the code exist on the droplet.**
  - This repo is checked out at `/work/video2sim-so101`; use this copy from now on.
  - An older copy at `/root/work/manip` was used for the first N=20 render test.
- **Environment.** Container `sprint` was set up with `env/start_container.sh` and `env/setup_container.sh`.
  - torch 2.10.0+rocm7.2 sees the GPU.
  - lerobot 0.6.1, mujoco 3.11.0, transformers 5.5.4.
  - `lerobot/pi05_base` and the friend's dataset are already downloaded into `/work/hf_home`.
- **Render test (N=20, 16 workers, OSMesa).**
  - Both test suites pass.
  - Front-camera and wrist-camera MP4s are written.
  - Load average is about 52 on 20 vCPUs: the OSMesa threads are oversubscribed.
  - Throughput is roughly 20 episodes in 6+ minutes, which puts N=1000 at about 5 hours.
- **Before the big run, try these one at a time and measure each:**
  - `LP_NUM_THREADS=1` with about 19 workers.
  - Recording images at 10 Hz instead of 30 Hz.
  - Smaller frames.
