LEGEND: [V] = read in a primary page this session, through a fetch summariser, so re-check any number you will cite. [I] = my inference.

**VERDICT**
- "Judged by JEPA" has no off-the-shelf form. V-JEPA 2 and 2.1 ship encoders and a planner, with no success or progress head [V].
- V-JEPA 2-AC's planning energy is an L1 distance to a goal image. The paper never calls it a reward [V].
- The only JEPA-as-judge precedent I found is arXiv 2510.16240. It puts a frozen V-JEPA 2 ViT-H and a 4-block attentive probe (~77M parameters) on 2,310 hand-labelled 32-frame clips. It reports ICC 0.836 against humans, where two humans agree at 0.811 [V].
- So our judge is a frozen encoder plus a small probe we train ourselves. It must beat the zero-shot reward models in the same table, or we drop it.

**CANDIDATES** (all fit in 192 GB)
- **V-JEPA 2.1** (arXiv 2603.14482, released 2026-03-16) [V]:
  - Sizes are ViT-B 80M, L 300M, g 1B and G 2B, all at 384 px, hosted at `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_*_384.pt`.
  - The code is MIT. The README states no licence for the 2.1 weights [V].
  - The transformers docs list only V-JEPA 2, so 2.1 loads through the official repo or torch.hub [V].
  - The V-JEPA 2 HF checkpoint `facebook/vjepa2-vitl-fpc64-256` is MIT [V].
- **Robometer-4B** (Qwen3-VL-4B, LeRobot `lerobot/Robometer-4B`):
  - The HF card says Apache-2.0 and the repo says MIT [V].
  - Progress is 10 discrete bins, and the LeRobot default is `max_frames=8` [V].
  - Value-order correlation (VOC) is 0.92 in-distribution and 0.95 out-of-distribution. Failure-detection F1 is 0.81, measured on 100 DROID trajectories [V].
  - SO-101 was not in its training data [V].
  - The community fine-tune `aliangdw/robometer-4b-fft-so101` is self-reported at Pearson 0.78 on Armnet and 0.90 on held-out MolmoACT. It reports no success metrics [V].
- **TOPReward** (zero-shot Qwen3-VL-8B, log p("True")) [V]:
  - VOC is 0.942.
  - Success ROC-AUC is 0.939 (8B) and 0.965 (32B), on 279 LeRobot episodes (150 successes, 129 failures). Robometer scores 0.930 on the same set.
  - A chat template cuts VOC from 0.945 to 0.500.
  - It is in LeRobot 0.6.
- **VLAC** (InternVL 2B/8B, CC-BY-NC-SA-4.0, non-commercial) [V]. Its VOC-F1 is 0.89 on successes and 0.44 on RoboFAC failures. Robometer's table gives VLAC failure F1 of 0.16. It is weak on negatives.
- **GVL** (Gemini-1.5-Pro, paid API) [V]. Success detection is 0.71 accuracy zero-shot and 0.75 one-shot. On open Qwen3-VL-8B its VOC is only 0.316.

**(a) CALIBRATING AGAINST THE PHYSICS VERIFIER**
1. **Labels.** Take two flags from sim state: L_task (object released and resting in the bowl) and L_full (every verifier check). Torque, limit and collision flags are invisible in RGB [I]. Report AUC for both, plus the share of L_full failures that video cannot show.
2. **Negatives.** Aim for at least 200 positives and 200 negatives, which gives about ±0.03 on a 95% CI at AUC 0.9 (my Hanley-McNeil calculation) [I].
   - Natural rejects: the step-servo ablation already fails 27.5%.
   - Injected failures: early release, 2-4 cm grasp miss, never releasing, a 1-3 cm near-miss outside the rim, idle stall, time-reversed playback.
3. **Splits.** Split by scene seed and hold out a camera-height range. Never split by frame.
4. **Contenders on identical clips:**
   - V-JEPA 2.1 ViT-L plus attentive probe on 16 frames at 384 px, in overlapping chunks as in 2510.16240 [V].
   - Robometer success and progress heads.
   - TOPReward-8B.
   - A trivial baseline: logistic regression on last-frame object and bowl pixels.
   - A shuffled-label control.
5. **Metrics.**
   - ROC-AUC.
   - Balanced accuracy at a threshold fixed on validation.
   - Per-failure-type recall.
   - Spearman correlation of predicted progress against physics progress, and VOC on successes.
   - Proposed gate [I]: the probe reaches AUC of at least 0.95 on L_task and beats the best zero-shot model by at least 0.02. If not, drop JEPA as the judge.
6. **First, check the OSMesa renders.** Look for the arm hiding the object at release, which would add label noise [I].

**(b) HUMAN VIDEO VERSUS SIM RENDER**
- A pooled-embedding cosine mostly measures hand-versus-gripper appearance [I].
- XIRL needed task-specific temporal-cycle-consistency training to get an embedding distance that survives embodiment change [V, 2106.03911].
- Test retrieval instead. Take at least 30 human clips, or one clip with perturbed specs, and render each.
  - Mask the hand and arm with SAM, keep the object and bowl region, and align features over time with DTW.
  - Report top-1 and MRR against chance of 1/N.
  - Controls: a render from the wrong spec, and a time-reversed render.
- Our task_spec path error (1.5 mm pick, 2 cm place) is a stronger fidelity metric. JEPA only adds "same activity" semantics [I].

**(c) THE FRIEND'S 524 REAL EPISODES**
- The dataset card has no success field and a single `task_index` [V], so ground truth does not exist.
- Score each camera separately, since Robometer takes one image key [V]. Use Robometer-FFT-SO101 and TOPReward. Rank the episodes and flag the lowest 10% and any non-monotonic ones.
- Hand-label 100 episodes (80 random, 20 lowest-scored). Expect a wide CI of roughly ±0.05-0.1 [I].
- If the demos are mostly successes [I], VOC checks that the judge is sane. It does not measure data quality.
- Do not apply a sim-trained probe to real data, because of domain shift [I].

**(d) SUCCESS DETECTION ON REAL ROLLOUTS**
- Use the Robometer success head (LeRobot `reward_output="success"`, threshold swept) or TOPReward.
- Or train a probe on 150-300 hand-labelled real rollouts, against the 2,310-clip precedent [V].
- Fix the threshold on labelled rollouts and report false-positive rate at 0.95 recall. A false "success" is the costly error [I].

**COMPUTE AND ROCm**
- I found no ROCm report for V-JEPA 2 or Robometer [V]. Both are plain PyTorch, so they are probably fine [I]. Run a 10-minute smoke test with sdpa attention.
- The official V-JEPA 2 repo needs decord, which its README calls unmaintained [V]. Avoid it:
  - Sim frames are arrays.
  - Real AV1 decodes through PyAV or torchcodec.
  - The HF VJEPA2Model path already uses torchcodec [V].
- The LeRobot robometer extra is transformers, qwen-vl-utils and peft. It needs no flash-attn or decord, torch>=2.7,<2.12, and Python>=3.12 [V].
- Qwen3-VL on ROCm is documented only through vLLM and SGLang. The plain transformers path is untested [V].
- VRAM [I]: V-JEPA ViT-L about 1 GB, ViT-G about 5 GB, Robometer about 9 GB, TOPReward-8B about 17 GB.
- GPU time for 10k episodes [I]:
  - V-JEPA ViT-L: about 0.3 h.
  - Robometer: about 1.5 h.
  - TOPReward (about 10 prefix passes): about 3 h.
  - Probe training: minutes.
  - Whole calibration: roughly 5 GPU-h, about $10 of the 40 h budget. CPU rendering is the bottleneck, not the GPU.

**WHAT WOULD BE DECORATION**
- A JEPA accept/reject filter on sim episodes. The verifier is exact, so it adds no information. Its real uses are as an instrument checking that renders show what the physics says, and as a scorer where no verifier exists (real data).
- V-JEPA energy to a goal image as a reward. It needs a viewpoint-matched goal image and is only shown for planning [V].
- Pooled cosine between human video and render with no retrieval control.
- GVL (paid, weak on open models) and VLAC as a failure detector.
- A single AUC with no trivial baseline and no shuffled-label control.

**NOT READ**
- Robometer appendices B-3 (compute) and C (preference accuracy).
- The V-JEPA 2.1 paper body (abstract only).
- The split and AUC tables in 2510.16240.
- PAYWALLED: none encountered.

**Sources**
- arXiv: 2603.14482, 2506.09985, 2603.02115, 2602.19313, 2509.15937, 2411.04549, 2510.16240, 2106.03911.
- HF: LeRobot Robometer docs and SARM docs, LeRobot pyproject, LeRobot v0.6.0 blog, the model cards named above, dataset card MinhDuk/final_dataset_annotated.
- GitHub: facebookresearch/vjepa2 README, robometer/robometer.