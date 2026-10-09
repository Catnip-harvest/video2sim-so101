**rl_benefit: what RL adds over SFT on a pretrained VLA, and what is feasible this weekend**

**Bottom line**
- RL adds three things SFT cannot: it trains on the policy's own failure states, it optimises task success instead of imitation likelihood, and it reduces the demos needed. The big gains in the papers are in weak-SFT or few-shot regimes.
- When SFT is already strong, RL adds roughly +4 to +8 points.
- For our budget, the safe weekend plan is iterative filtered BC using our physics verifier as the reward. PPO via RLinf on π0.5 is a stretch goal (details below).

**Evidence (SFT → SFT+RL)**

| Method | Setting / base | Result | Compute / images |
|---|---|---|---|
| RL4VLA 2505.19789 | ManiSkill3 sim, OpenVLA 7B | PPO beats GRPO and DPO. In-distribution 0.781→0.938. Unseen robot pose 0.339→0.797, object position 0.568→0.807. Strong texture change only 0.557→0.630. | VERIFIED. ~42 h on one A100. 640×480 RGB at every step. |
| SimpleVLA-RL 2509.09674 | OpenVLA-OFT, GRPO, binary reward | LIBERO full SFT 91.0→99.1. One-demo SFT 48.9→96.9. RoboTwin2.0 38.3→68.8. Sim-to-real 17.5→38.5. | VERIFIED. 8×A800, batch 64 × 8 samples. Hours not reported. |
| VLA-RL 2505.18719 | OpenVLA-7B LoRA, learned process reward | LIBERO 76.5→81.0 (+4.5). | VERIFIED. 48 GPU-h. |
| RLinf-VLA 2510.06710 | OpenVLA / OFT | ManiSkill OFT 28.1→97.7 (PPO). LIBERO 42.1→98.1 (GRPO). RoboTwin 24.5→84.6. PPO beats GRPO on ManiSkill. | VERIFIED. GPU type and hours not stated. |
| πRL 2510.25889 | π0 / π0.5, PPO (Flow-Noise / Flow-SDE), VLM frozen, 300M action expert trained | LIBERO few-shot: π0.5 77.1→98.3, π0 57.6→97.6. ManiSkill π0.5 40.1→90.9. Real: 20 demos + 100 RL iterations gave 40% zero-shot, using a Gaussian-splat twin. | VERIFIED. 8×H100. 64 envs × 8 rollout epochs per step, 500 steps on LIBERO. That is ~256k episodes at most (INFERRED). |
| π*0.6 RECAP 2511.14759 | Real robot only, advantage-conditioned, offline iterations | About 2× throughput, about half the failures. Two iterations of 300–600 trajectories each. | VERIFIED. No PPO, no simulation. |
| RLT 2604.23073 | π0.6 frozen, small actor-critic on an "RL token" | Screw critical phase 20→65%, up to 3× faster. 400–1000 episodes. | VERIFIED. Real robot, human-labelled reward. |
| DSRL 2506.15799 | π0 frozen, RL on the latent noise | LIBERO ~20%→~100% in ~10k online samples. Real toaster 5/20→18/20 in 80 episodes. | VERIFIED from the paper's text. Compute not reported. |
| ConRFT 2502.05450 | Octo-small, real Franka | 96.3% average, +144% over the SFT baseline, 45–90 min online. | VERIFIED. One A6000. Uses a reward classifier and human takeovers. |
| PA-RL 2412.06685 | Critic-ranked best-of-N, then distill | OpenVLA real 40→70%. | VERIFIED. The authors say sampling is expensive for big models. |
| ReinFlow 2505.22094 | Small flow policies | +135% reward on locomotion, +40% success on manipulation. | VERIFIED. No π0. The paper calls scaling to flow VLAs future work. |
| Flow-GRPO 2505.05470 | Text-to-image | GenEval 63→95%. | VERIFIED. Not robotics. πRL's Flow-SDE is the VLA analogue (INFERRED). |
| Filtered BC / rejection sampling | Self-Improving EFMs 2509.15155 | Abstract says SFT plus self-improvement beats collecting more imitation data. | The abstract has no numbers. A search snippet claimed ≥1.5× with <2% extra episodes (secondhand, INFERRED). |

**Filtered BC caveat.** I found no matched-budget filtered-BC versus PPO paper. A search summary claiming one was wrong: 2608.11363 is MiDAS, an online RL method. So filtered BC has thinner evidence than PPO.

**RLinf, ROCm and MuJoCo**
- VERIFIED from the README: π0/π0.5 on LIBERO and ManiSkill carries a ROCm tick. It states "Every hardware backend fully supports all the environments listed in its row."
- The ROCm Docker image is `rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-rocm7.2`, and the build args include `gfx942`, which is the MI300X architecture.
- On AMD, LIBERO uses CPU rendering (`MUJOCO_GL=osmesa`). The docs warn software rendering eats CPU and to scale env counts down. ManiSkill uses CPU simulation on non-CUDA backends.
- The README algorithm list includes PPO, GRPO, DSRL, RLT and RECAP (CFG).
- MuJoCo is not a listed env. A custom env goes in through the `gym.Env` guide (`new_env.rst`). The doc names no MuJoCo or custom-env example for π0.5, so wrapping our scene is real work (INFERRED).
- Nobody has shown RLinf π0.5 running on an MI300X. It is a README claim, so treat the first run as a smoke test.
- VERIFIED: Warp's install page lists NVIDIA CUDA GPUs only, with CPU as the other option. mjlab says "requires an NVIDIA GPU for training."
- INFERRED: mjlab physics will not run on the droplet's GPU. Any RL there has to use classic CPU MuJoCo on the 20 vCPUs.

**Feasibility for us (budget figures INFERRED, not benchmarked)**
- Our 10–15 GPU-h is about a quarter to a third of the smallest published runs (RL4VLA 42 h, VLA-RL 48 h). It is roughly 1–5% of πRL's LIBERO run.
- Render only at policy queries (every 5–10 physics steps), not at every 30 Hz step. That is about 30–60 render calls per episode instead of 300, so CPU rendering stops being the bottleneck.
- Guess: 1–3 episodes/s. That puts the ceiling around 36–100k episodes in 10 h. After PPO updates, evals and restarts, plan on 10–30k usable episodes.
- Measure episodes per second in the first hour. That one number decides which option below is viable.
- Memory is not a constraint. πRL freezes the VLM and tunes only the 300M action expert, which fits easily in 192 GB.

**Options, ranked for this weekend**
1. **Iterative filtered BC.**
   - Roll out the SFT π0.5 on 2–4k episodes per round, keep verifier passes, and fine-tune on demos plus successes. Do 2–3 rounds.
   - It reuses the SFT code, needs no new framework and has no critic.
   - Its weakness is that it discards failures.
2. **RLinf PPO with π0.5** (PPO, not GRPO, because the π0.5 published recipes and the ManiSkill results favour PPO).
   - Smoke-test LIBERO on the droplet first, which should take under 2 hours.
   - Wrapping our own env is a stretch goal, roughly 0.5–1 day (INFERRED).
3. **DSRL-style latent steering of a frozen π0.5.** This is the cheapest true RL: about 10k steps, and RLinf lists DSRL in its README. The evidence is one paper, on π0 only.
4. **Skip:** RECAP from scratch, ConRFT, PA-RL and ReinFlow. They are real-robot, small-policy or critic-heavy.

**Reward**
- Our physics verifier is the reward. Use binary success on verifier pass.
- Add a small shaping term for a grasp-and-lift, as ManiSkill does (+0.1 for gripper attachment).
- Keep RLinf's `success_once`-style success flag as the headline metric.

**Realistic gain (INFERRED)**
- If SFT is at or above 85–90% on our sim, expect +4 to +8 points at most.
- If SFT is 40–70% on a narrow task, plan on +10 to +20 points absolute in-sim. Anything higher is a bonus.
- Real-robot transfer is unlikely. Every sim-to-real result above used a digital twin (a photoreal twin such as Gaussian splats, or a matched RoboTwin scene), and our CPU-rendered scene is not one.

**Real episodes**
- They enter at SFT time, and the demos can stay in the mix during filtered BC (the anchoring idea MiDAS also uses).
- I did not research this question in depth.

**PAYWALLED:** none. Every source I used was open.

**Not read:**
- pi.website pages (HTTP 429, rate limit, so I used arXiv HTML for RLT).
- RECAP baseline figures.
- RL4VLA's parallel-env count.
- πRL's GPU-hours.
- Self-Improving EFMs' figures.