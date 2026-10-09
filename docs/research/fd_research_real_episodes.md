**Bottom line.** Real episodes can seed the engine, co-train the VLA and serve as a weak offline eval set. The blocker is the joint mapping. Zero offsets are per-arm and unknown, so the mapping must be fitted, not assumed.

## 1. Units of the friend's data
- **VERIFIED — robot type:** `so_follower` is one class for SO-100/101/10X. The docstring reads "SO-100/101/10X", and `SO101Follower = SOFollower` is an alias. `use_degrees` defaults to True. The name does not prove it is an SO-101 (INFERRED risk), so confirm with the friend.
- **VERIFIED — normalisation (`motors_bus.py`):**
  - Arm joints in DEGREES: `(val - mid) * 360 / max_res`, with `mid=(range_min+range_max)/2`. STS3215 resolution is 4096, so `max_res`=4095 (about 0.088° per tick).
  - Gripper always uses RANGE_0_100: `(val-min)/(max-min)*100`.
  - RANGE_M100_100 clips to ±100. DEGREES does not clip.
- **VERIFIED — which mode your data is in (`meta/stats.json`):** state shoulder_lift min is -105.7, wrist_flex max is 103.5 and wrist_roll max is 134.2. Action shoulder_lift min is -107.9. These exceed ±100, so the data is DEGREES, not M100_100.
- **INFERRED — calibration generation:** the ranges do not match the pre-PR #777 convention, where LeRobot's docs say lift = -(lift-90) and elbow -= 90.
- **VERIFIED — what zero means:**
  - For four joints, 0° is the midpoint of the user's swept range.
  - wrist_roll is forced to range 0..4095, with `homing_offset = pos - 2047`. Its 0° is wherever the wrist was held in the "middle" step, so it is arbitrary per arm.
  - The gripper is a percentage of the calibrated jaw travel.

## 2. Mapping to Menagerie `robotstudio_so101`
- **VERIFIED — the model:** it is derived from `so101_new_calib.xml`, where zero is mid-range. Limits are pan ±110°, lift ±100°, elbow ±96.8°, wrist_flex ±95°, roll -157..+163°, gripper -10..+100°. The FK site is `gripperframe`.
- **Formula:** `q_rad = sign * deg2rad(v) + off`.
  - Start with sign=+1 and off=0 for pan, lift, elbow and wrist_flex.
  - **VERIFIED — wrist_roll:** so-frame `bridge.py` sets `OFFSET_REAL["wrist_roll.pos"]=90.0`, so sim = real - 90°. Its comment says the real wrist sat 90° from where sim showed it, and every other joint is identity. This was measured on their arm and must be re-checked after re-homing.
  - **INFERRED — consistency:** your wrist_roll mean of 76.6° fits a roughly +90° convention. It is not universal.
- **Gripper — VERIFIED, two conventions that disagree by up to 10° at closed:**
  - leisaac maps 0..100 linearly onto -10..100° (`motor_degree/motor_range*joint_range + limit_min`).
  - so-frame treats the gripper like the other joints, with θ° = g and no offset.
  - Fit with two anchors: closed on nothing, and your hold value of 24.8 against the cube width.
- **INFERRED — offsets exist:** the data exceeds Menagerie limits by 5.7° on lift (-105.7), 8.5° on wrist_flex (103.5) and 1.1° on elbow (97.9), so offsets of that size exist. 1° at about 0.3 m reach is roughly 5 mm. This is arithmetic from the XML link lengths of about 0.116, 0.135 and 0.16 m. Fit to 1–2° or less.
- **VERIFIED — use `observation.state` for FK, not `action`:** the action-minus-state mean gap is about 2.5° on lift (-34.2 vs -31.7) and 3.8° on elbow (22.4 vs 26.2). That is leader/follower sag.

## 3. Validation (INFERRED procedures; only the overlay tool exists in a source)
1. **Table-plane test (no camera needed).** At each grasp, FK z of `gripperframe` over about 335 table→bowl episodes (64% of 524) must be constant at table height plus cube half-height. Fit lift, elbow and wrist_flex offsets to minimise the variance of z.
2. **Homography test.** Fit FK (x,y) at grasp to the green-cube centroid in `cam_head` (same fixed-camera assumption as test 3). Residuals above a few pixels mean the pan or reach mapping is wrong.
3. **Overlay test.** Render the sim over real frames and compare, as in so-frame's calibrate tool (REAL/SIM/OVERLAY). It needs the head-camera extrinsics.
4. **Gripper and rest anchors.** Use the gripper anchors from section 2. The folded rest pose sits on hard stops, so rest readings against the ±100° stops give offsets directly.

## 4. Methods (all VERIFIED from arXiv)
- **MimicGen (2310.17596):**
  - Each subtask is a contiguous segment referenced to one object, with target poses re-expressed in the new object frame. A linear interpolation segment joins them.
  - It starts from about 10 source demos and generates 1,000 per task. Only successful rollouts are kept.
  - It assumes a known subtask sequence, observable object poses, rigid objects and quasi-static tasks.
  - Pose-noise ablation: 5 mm/5° dropped Square D0 generation rate from 73.7% to 60.9%, and 10 mm/10° dropped it to 30.5%. Policy success held at 89.3% and 84.7%.
- **SkillMimicGen (2410.18907):** skills are replayed and transit moves are motion-planned. Generation success is 75.4% vs 40.7% for MimicGen. In the real world it used 3 source demos and 100 generated per task.
- **DexMimicGen (2410.24185):** adds bimanual and dexterous support. Fingers are replayed rather than transformed. It has no collision handling. It generated 21K demos from 60 source demos.
- **DemoGen (2502.16932):**
  - It uses one demo with 3D point-cloud editing, and the paper reports full SE(3) transforms.
  - Real-world success was 74.6% vs 11% for the source demo alone.
  - It synthesises point clouds, not RGB, so it does not feed an RGB VLA directly.
- **CP-Gen (2508.03944):** one demo, keypoint constraints solved with IK, then motion planning. It averaged 77% vs 50% for the best baseline.
- **R2R2R (2505.09601):**
  - Its input is a human video plus an object scan, with no robot data. The hand is used only to pick the grasped part.
  - It uses differential IK with no dynamics. Throughput is 51 demos per minute on an RTX 4090.
  - It matched 150 teleop demos only on some tasks. The paper calls the results "similar but not provably equivalent."
- **Real-to-sim replay for SO-101:** I found no documented mapping.
  - A Hugging Face replay of a 535-episode real SO-101 dataset into MuJoCo exists, with no units or fidelity stated.
  - The only calibration-style tool I found is so-frame's overlay.

## 5. Decomposition, retargeting, accuracy
- **INFERRED — segmentation:** use the gripper channel. The close event with a plateau near 24.8 marks grasp, and the later rise marks release. That gives approach, grasp-lift-transport, place and retreat segments.
- **INFERRED — object poses from FK:** the cube pose is the `gripperframe` FK pose at grasp, and the bowl pose is the FK pose at release. The arm localises the objects, so no perception is needed.
- **Retargeting (MimicGen rule):** `T_new = T'_obj · T_obj⁻¹ · T_src`. Then interpolate, solve IK and run the physics verifier.
- **INFERRED — 5-DoF limit:** the SO-101 has 5 DoF, so only top-down grasps with a free yaw (wrist_roll) are reachable.
- **INFERRED — grasp tolerance:** about half of (open jaw width - cube width). MimicGen's drop with 5 mm of pose noise is the nearest evidence.
- **INFERRED — benefit:** human timing and curvature from real segments address the low-diversity weakness of the scripted phases.

## 6. Your three uses
- **(a) Seeds — yes.** Only trajectories carry over, because the original camera frames are invalid once objects move. RGB must be re-rendered on CPU (ROCm), as in R2R2R.
- **(b) Co-training — yes, directly.**
  - Write sim data in the real units: degrees, gripper 0..100, `.pos` names and camera keys `cam_head`/`cam_right`. This gives one normalisation for pi0.5 (INFERRED).
  - Sim-and-Real Co-Training (2503.22634) reports up to 2–7x when real data is scarce, and it is sensitive to the real-sampling probability α. It only tested Diffusion Policy on planar pushing.
- **(c) Offline eval — weak.**
  - CI-MSE (2606.29898): checkpoint-to-rollout Spearman is -0.87 for task-critical-interval MSE vs -0.61 for raw MSE.
  - Hold out whole episodes, score only about ±1 s around grasp, and add closed-loop sim from FK-derived layouts.
  - The SIMPLER abstract claims strong sim-real correlation but gives no numbers.

**Not verified:** whether LeRobot trains on mixed datasets, and whether Isaac Lab Mimic handles joint-space SO-101 actions. The Isaac Lab docs say Mimic environments implement `target_eef_pose_to_action` for EEF poses, so joint-space data would need FK.

**PAYWALLED:** none. All sources were arXiv or GitHub.