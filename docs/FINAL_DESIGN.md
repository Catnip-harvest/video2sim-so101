> **Checked by hand on 9 Oct, after the workflow run.**
>
> - **The verifier depends on the scripted demo's phase labels.** This is confirmed in `datagen/r2s/rollout.py`:
>   - `CARRY_PHASES`/`PRE_GRASP_PHASES` at line 64, `if phase in CARRY_PHASES` at line 151, and the holding check at line 231.
>   - So it cannot judge a policy's own attempts until a version without phase labels is written.
> - **π0.5 needs a gated tokenizer.** The preprocessor in `lerobot/pi05_base` loads the `google/paligemma-3b-pt-224` tokenizer.
>   - On Hugging Face that model is gated with **manual** approval, so the owner's account must request access before any π0.5 run.

# Final design: one video → verified sim dataset → π0.5 → self-improvement in sim

[V] verified in a source, [I] inference, [?] measure on the droplet.

## 1. Your three questions

**What does RL add over fine-tuning?** Fine-tuning (SFT, behaviour cloning) copies demos: it never sees the states its own mistakes create, and it optimises "look like the demo", not "succeed". Two method families differ:
- **Filter or reweight** (Ilia's AWR+RECAP, our filtered BC): only sharpens behaviour already present (Ilia §10.4 [V]).
- **Exploring RL** (PPO/GRPO): can find new strategies; big gains on execution shifts even from strong SFT (RL4VLA: unseen robot pose 33.9→79.7, object moved mid-episode 28.6→74.5 [V]).
- Our twist: the scripted expert makes unlimited verified demos, so "more expert data" is a strong rival. Prior for our weekend loop: **0 to +10 points [I]**. The +40-point results used PPO/GRPO on 8×H100 (πRL: 320 envs, 16,384-episode SFT [V]).
- Sim-trained RL has transferred without a photoreal twin (RL4VLA Franka pick-place 0→27%; SimpleVLA-RL Piper 17.5→38.5% [V]), at low absolute success.

**How did Ilia reward, and how will we?** No marker on the shirt: his simulator checked six cloth points and paid 0.5, then 1.0, at fold checkpoints, all withdrawn on failure. Ours: pass/fail from the physics verifier (§3).

**Can we feed real episodes in?** Yes, after calibration: as seeds (trajectories re-rendered in sim) and as co-training data. Never as reward. This weekend: laptop calibration with numbers; co-training after (§6).

**What "judged by JEPA" means.** Sim episodes are judged by the physics verifier, which is exact. The frozen V-JEPA 2 video encoder gets two jobs the verifier cannot do: a **realism score** (distance between our renders and the friend's real clips) and a **success probe** tested against exact sim labels, to learn whether JEPA could later judge real rollouts. Not a reward, not a filter.

## 2. Pipeline

video → `demo/` task spec (object path, grasp/release timing, pinch width) → `datagen/` scripted expert under randomisation → physics verifier keeps passes → CPU (OSMesa) render at 224 px, LeRobot v3 in radians, exact segmentation-mask visibility check (replaces a learned "render audit") → JEPA scores (§4) → π0.5 SFT → filtered-BC round vs controls (§5). The sim task is "cube into an 11×11 cm tray", not a bowl.

## 3. Reward

**Blocker.** The verifier cannot score a policy rollout as built: clauses read the expert's phase labels (`rollout.py:64,151,169`), and "holding" needs a target unchanged to <1e-7 for two frames (`:231`). On policy rollouts it silently degrades to "cube ends in tray", losing the anti-pushing guard.

**Phase-free policy verifier** (laptop, before any rollout):
- grasp/carry = both jaws touching AND cube z > rest + 5 mm; release = gripper opening; holding = torque/sag window only;
- new: cube bottom ≤10 mm above the floor at release (a 10 cm drop lands at 1.40 m/s, under the old 1.5 m/s fling limit); fling limit lowered to 0.5 m/s;
- kept: no foreign contact, collision, limit or torque violation, penetration < −2 mm;
- test: 50 expert episodes with labels stripped get identical verdicts; zoh drops and a scripted push-into-tray still fail.

**Weekend reward = binary pass**, like Ilia's return. Primary metric L_task (released, in tray); L_full secondary; style-only rejections reported separately. Shaping (+0.1 grasp-and-hold per RL4VLA [V], +1.0 success, withdrawal on failure) matters only for AWR or PPO (stretch); the mjlab dense reward is unused. No learned model in the reward; winners re-verified at half timestep and held-out friction/mass. SimpleVLA-RL's emergent "pushcut" is excluded by task definition (both-jaw clause).

**Ilia's** [V, paper + checker code]:
- The organisers' checker reads six cloth-mesh points (`check_points`) from simulator state; some pairs must close (`<= thr`), some spread (`>= thr`).
- The first 0.5 is paid gradually for tops only; return = 1[success].
- Not PPO: frames sampled ∝ exp(clip(A,−2,2)), advantage conditioning (prefix + AdaRMS), CFG 7–9, best-of-2–3; GAE (γ 0.999, λ 0.99) from the model's own success/completion heads; asynchronous rollouts, stale data decayed 0.98 per iteration.
- 79.63% vs 73.50% second; no RL-vs-BC ablation, so his RL gain is unmeasured.
- Real round: no real reward, but it started from a late RL-trained sim checkpoint plus sim success replays (10%) and DAgger.

## 4. JEPA

**Model:** frozen V-JEPA 2 ViT-L, `facebook/vjepa2-vitl-fpc64-256` (MIT [V]). V-JEPA 2.1's torch.hub points at localhost [V]; load its `.pt` directly if wanted. **Precedent:** arXiv 2510.16240 (ViT-H, 32-frame clips, 2,310 human labels, surgical video) reached ICC 0.836 vs human–human 0.811 [V]; our ViT-L/16-frame setup is untested.

**Realism score [I]:** Fréchet distance in V-JEPA features between renders and the friend's `cam_head` clips, with real-vs-real and sim-vs-sim split-half floors. Picks camera pose and randomisation ranges.

**Success probe:** labels from sim state. Negatives: natural rejects (zoh: 19 + 6), widened randomisation, 2–3 injections (no close, early release); no reversed clips. Split by scene seed, hold out a camera-height band. Contenders: probe, a *trained* last-frame logistic, shuffled-label control. Gate: beat last-frame by ≥0.02 AUC, paired DeLong (CI ≈ ±0.022 at AUC 0.95, 200/200). Report per failure type. Fail → JEPA lost.

Stretch: Robometer-4B, TOPReward on Qwen3-VL-8B (lerobot 0.6.x [V]); a real success detector needs labelled real failures.

## 5. Weekend method: filtered BC (not RL)

Name it **"self-improvement by filtered BC (rejection-sampling fine-tuning)"**; say "RL" only if AWR or PPO runs.
1. Roll out SFT π0.5 on ~500 seeds (N = hours × measured episodes/s), 4–8 processes with own bf16 replicas (~7 GB); CPU MuJoCo.
2. Keep verifier passes; re-render at 224 px, 10 Hz held frames; fine-tune `train_expert_only`.
3. **Control A:** same count of fresh expert episodes, same scenes, same steps. **Control B:** SFT + inference-time action smoothing.
4. One seed list, paired McNemar, CIs. Claim a gain only over the best control.

Cut: expert-restart corrections (DAgger). SFT <10% → no loop, fix data; ≥90% → perturbation eval instead.
Reserve-only: RLinf LIBERO smoke ≤2 h (official π0.5 ROCm image for gfx942 [V]; PPO throughput unbenchmarked; wrapping our env ~0.5–1 day). Warp ROCm physics smoke ≤1 h ("CUDA-only" is now [?]).

## 6. Real episodes (laptop)

- DEGREES mode confirmed (lift −105.7, wrist_flex 103.5, wrist_roll 134.2 exceed ±100); 0° = midpoint of the calibration sweep, wrist_roll 0° = homing pose [V].
- Primary source: the friend's calibration JSON; offset = recorded mid-range − MJCF mid-range. Count frames outside MJCF limits before seeding.
- **G0:** PnP-placed sim camera; render the arm at converted joints over ~20 real frames; mask IoU ≥0.8, residuals in px/mm. Table-plane fit is a cross-check only.
- One conversion module, round-trip tested; sim stays in radians.
- Co-training later needs: matched action−state gaps (gripper 15.7 vs 21.0), keys `cam_head`/`cam_right` (no wrist cam), a bowl, one merged dataset.
- Seeding: grasp = gripper closing to ~24.8; MimicGen retarget `T_new = T'_obj·T_obj⁻¹·T_src` → IK → verifier. 5 DoF limits approach to the radial vertical plane. At 5 mm/5° noise MimicGen's yield fell 13–49 points while policy success held [V].

## 7. Build order (wall clock)

Droplet alive since ~08:57 UTC Fri 9 Oct; hard stop at 40 alive hours (~00:57 UTC Sun); powered-off still bills; no snapshots. Destroy **Sat ~14:00 UTC (~29 h ≈ $58)**, ~11 h slack.

| UTC | Step | h | Gate / kill |
|---|---|---|---|
| Fri 09:30–11:30 | `lerobot==0.6.1` (0.6.2 doesn't exist), PIP_CONSTRAINT; PaliGemma tokenizer (gated, request now); π0.5 20-step smoke (bf16), latency at batch 1 and 8–16; render A/B (LP_NUM_THREADS=1, smaller/no shadows, offsamples 0–1) | 2 | Render ≥0.5 eps/s (0.03–0.055 now). Tokenizer refused, latency >300 ms or <2 epochs in 10 h → SmolVLA, stated |
| 11:30–~14:00 | 300 episodes + judge negatives + default-spec trajectories; LeRobotDataset loads; push to HF | 1.5–3 | Acceptance ≥70%; mask check |
| → Sat 00:00 | π0.5 SFT unattended; steps = 10 h ÷ s/step = LR decay; checkpoints to HF every 30 min; 4 cores trainer, 15 render; JEPA work alongside | ~10 | dataloading_s ≪ update_s |
| Sat 00–02 | SFT eval, fixed seeds (≥400 if throughput allows) | 2 | Picks §5 branch |
| 02–09 | Filtered BC + Controls A, B | 7 | <0.05 eps/s → skip |
| 09–14 | Cards (Gemma licence), eval table, pull all, destroy | 5 | — |

Laptop, in parallel: phase-free verifier, eval harness, calibration.

## 8. Claims

**Will claim** (numbers, CIs): N episodes per video verified in simulation (MuJoCo contacts), acceptance rate, render cost; task-spec error 1.5 mm / 2 cm on a synthetic clip with A4, ±20% scale without; the video's contribution as trajectory statistics vs a default spec; π0.5 sim success on held-out seeds and stated range edges; filtered-BC delta vs the best control; JEPA results, or that JEPA lost; calibration residuals.

**Will not claim:** real-robot success; "RL" for filtered BC; generalisation beyond the ranges; that the robot imitates the hand; JEPA as reward or filter; borrowed RL gains; VLA-JEPA results unless run; a bowl task.

## 9. Objections narrowed or rejected

n≥1,000 eval "in minutes" (measured speed says hours); pure-DAgger control (expert restart cut); second SFT on default-spec data (no budget); RLinf as early go/no-go (a pass cannot wrap our env this weekend).

**PAYWALLED:** none. Unread: Ilia's references, RECAP past 100k chars, Robometer appendices, V-JEPA 2.1 body, 2510.16240 split tables, pi.website (HTTP 429).
## 10. Model choice update (9 Oct, after the owner asked about MolmoAct 2)

**Facts checked on 9 Oct:**
- **LeRobot 0.6.1 ships the policies we need.** `src/lerobot/policies` at tag v0.6.1 contains `molmoact2`, `pi05` and `vla_jepa`.
- **`allenai/MolmoAct2-SO100_101` exists.** It is already fine-tuned on an SO-100/101 data mixture: absolute joint-pose control, language-annotated tasks such as "grasp the lemon … drop it into the red bowl".
  - Not gated. 21.8 GB in fp32 (about 5.4B parameters).
  - Inference needs about 24–26 GB in fp32. bf16 is supported and needs roughly half.
  - Its state vector is in LeRobot SO-101 joint units (degrees), not our sim's radians.
  - The licence is not stated on the model card. Check before publishing.
- **π0.5 (`lerobot/pi05_base`) is a generalist base.** It is about 3.3B parameters, needs the gated PaliGemma tokenizer, and AMD has reported fine-tuning it on MI300X.

**Decision: smoke-test both before committing.** The table has a time box for each.

| Step | Pass if | Time box |
|---|---|---|
| π0.5: 20 training steps in bf16 on our sim data | finite loss; record s/step and peak memory | 20 min |
| MolmoAct2-SO100_101: 20 training steps in bf16 via LeRobot `molmoact2`, CUDA graphs OFF on ROCm | finite loss; record s/step and peak memory | 30 min |

**Which model becomes primary:**
- **MolmoAct2-SO100_101,** if it trains on ROCm and fits roughly 2 epochs in about 10 h. It has already seen real SO-101 data, which is a better prior for later transfer to the friend's arm, and the friend's team uses Molmo too.
- **π0.5 otherwise.**

Either way, the sim data must be exported in the chosen model's joint units through the single tested conversion module (§6). A deployment GPU for MolmoAct 2 needs at least 16 GB (bf16). The laptop's RTX 4060 (8 GB) cannot run it.
