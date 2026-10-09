**Reward research for the SO-101 pick-and-place RL stage**

Tag meanings:
- **VERIFIED**: extracted from the primary arXiv HTML. A small summariser model sat between me and the page, so formulas were not read raw. Long pages were cut at about 100k characters.
- **SNIPPET**: seen only in search excerpts.
- **INFERRED**: my reasoning.

**1. Ilia's reward (arXiv 2606.27163, LeHome)**
- **Success is geometric.** VERIFIED: "Success is binary and defined geometrically." Pairs of garment keypoints that should meet (for example two sleeves) must come closer than a threshold. Tops have 5 conditions, short pants 4, long pants 7. These use privileged simulator state.
- **The "marker" is probably these keypoint pairs.** INFERRED. I found no visual marker on the shirt.
- **Reward:** VERIFIED: "Reaching the intermediate fold checkpoint is worth 0.5 reward," and full success brings it to 1.0. The first checkpoint on tops is graded: 0.5·clip((d₀−m_t)/(d₀−d_t1),0,1), where m_t is a running minimum.
- **Failure withdrawal:** VERIFIED: "all reward withdrawn on failure so the episode return stays binary", so ∑r = 1[success].
- **Why shape at all:** VERIFIED: binary success is "too sparse for efficient RL".
- **Method is not PPO.** VERIFIED. It combines advantage-weighted regression through sampling (frames drawn in proportion to clipped advantage) with RECAP-style advantage conditioning. The value is the policy's own success head (BCE loss) plus a completion head (t/T on successes only). No VLM reward model is used.
- **Results:** 79.63% in sim, 1st of 62 teams.
- **Real robot:** VERIFIED: no RL. The real policy was behaviour cloning plus DAgger, because there was no real-side reward or value function. It placed 2nd (865 vs 895).

**2. Object-centric tracking from a human demo**
- **X-Sim (2505.07096).** VERIFIED. The object is tracked with FoundationPose, and the teacher is PPO on privileged state, distilled into a diffusion policy.
  - r_goal = (1 − tanh(α_d·d_pos)) + (1 − tanh(α_θ·d_rot)) + 2·i_waypoint
  - r_approach = 1 − tanh(k·d_obj)
  - r_obj = r_approach + r_goal, plus r_static and a +1 r_success.
  - The goal advances when the object is within ε of the current waypoint. The paper gives no values for ε, k, α or the weights.
  - PPO settings: lr 3e-4, γ 0.8, 1,024 environments.
  - Results: +30% task progress over baselines. 90% success from 1 minute of human video, against 70% from 10 minutes of robot demos. The 20 s versus 60 s per-clip figures in the text conflict with those totals.
  - The paper does not mention reward hacking.
- **Human2Sim2Robot (2504.12609).** VERIFIED. There is a single term, r = exp(−α·d(T_target, T_obj)), with d = Σ‖T1·k_i − T2·k_i‖ over 3 anchor points at L = 0.2 m and α = 10.
  - Ablations: a fixed or interpolated target gets stuck in a local minimum. Downsampled waypoints learn more slowly with jerky motion. Adding a hand-tracking reward gives no gain at convergence.
  - Episodes reset if the object is more than 0.25 m from its target.

**3. Learned progress rewards**
- **GVL (2411.04549).** VERIFIED. Gemini-1.5-Pro is asked for completion percentages on shuffled frames.
  - VOC is 0.74 on RT-1 but −0.01 on DROID and −0.85 on RoboNet.
  - Without shuffling, predictions collapse into a linear ramp.
- **VLAC (2509.15937).** VERIFIED. It outputs a pairwise progress delta, c = VLAC(o_i, o_{i+Δt}; task), plus a done head. Real-robot PPO went from about 30% to about 90% in 200 episodes. The paper does not discuss false "done" signals or reward hacking.
- **RoboReward (2601.00675).** VERIFIED. It gives a 1–5 score; the 8B model has MAE 0.665.
  - Real-robot RL success: human reward 75%/90%, RoboReward-8B 50%/80%, Gemini 10%/45% (monkey/drawer).
  - Reward accuracy correlates with downstream RL at r = 0.83.
- **Robometer (2603.02115).** VERIFIED. It is a 4B Qwen3-VL trained on RBM-1M (over 1M trajectories, 21 embodiments).
  - Loss is L = L_pref + L_prog + L_succ.
  - Online RL with DSRL on π0: 20% to 85%.
  - Its own limitations say it "lacks direct access to latent physical state such as contact forces, grasp stability" and "may fail to recognize... certain failure cases."
  - The paper has no reward-hacking analysis.
- **Hacking evidence (SNIPPET only):** CLIP similarity reward on FetchReach was gamed through 2D appearance (a Stanford class project). Dream2Reward (2608.18787) argues progress rewards stay high after a wrong transition. I have not read it.
- **V-JEPA as a reward: no primary source found.** V-JEPA 2-AC uses L1 distance between the predicted and goal embeddings as a planning energy, not an RL reward. VERIFIED.
  - Pick-and-place success: cup 80%, box 65%. It needs a goal image and is sensitive to camera pose.

**4. VLA RL: how reward is defined**
- **π*0.6 / RECAP (2511.14759).** VERIFIED.
  - Reward is −1 per step, 0 at success, −C_fail at failure (value not given).
  - The value function is distributional over 201 bins. The indicator is I = 1(A > ε), with ε at the 30th percentile.
  - Throughput more than doubled and the failure rate roughly halved.
- **RL4VLA (2505.19789).** VERIFIED. Reward is sparse: 0.1 for grasp-and-hold, 1.0 for placement. PPO beat GRPO and DPO. On out-of-distribution objects and tables, RL beat SFT by 42.6%.
- **SimpleVLA-RL (2509.09674).** VERIFIED. Reward is binary 0/1, trained with GRPO.
  - LIBERO 91.0% to 99.1%. One-trajectory SFT 48.9% to 96.9%.
  - "Pushcut": the policy pushed the can into place instead of grasping it.
- **πRL (2510.25889).** VERIFIED. PPO on π0/π0.5 with reward 1.0 for placement plus 0.1 for gripper attachment.
  - ManiSkill π0.5: 40.1% to 90.9%. LIBERO π0.5: 77.1% to 97.9%.

**5. Reward spec for us (all weights INFERRED, built from the sources above)**

The verifier is the only authority on success. All shaping is subordinate to it.

- **Success:** +1.0, terminal, only if every verifier clause passes (both-jaw lift, lands in bowl, released, no violations).
- **Milestones, paid once per episode:**
  - +0.1 for a both-jaw contact lift held longer than 0.5 s (the RL4VLA/πRL 0.1 value).
  - +0.05 for cube over bowl while still jaw-held.
- **Dense terms (teacher only):** potential-based, so the optimal policy is unchanged, with cumulative shaping capped at 0.35 so it cannot outweigh success.
  - Φ = 0.15·(1 − tanh(10·d_ee→cube)) + 0.20·exp(−10·d_cube→waypoint).
  - Waypoints come from the human object path. Advance when within 2 cm.
  - Cube is symmetric, so use position only (no anchor points).
- **Withdrawal:** if the verifier fails, subtract the shaping accrued, so the return equals 1[success] (Ilia).
- **Small penalties:** action rate 1e-3·‖Δa‖² and torque 1e-4·‖τ‖². The verifier clauses (collision, limits, torque) become hard terminations instead of weights.

**Which signal for which RL variant**

| Variant | Signal |
|---|---|
| A. State-based PPO teacher (X-Sim style), then verifier-gated demos into the engine | Full dense spec above. Best fit for our data engine, and it fixes the low-diversity scripted-demo weakness. |
| B. Online PPO on π0.5 in sim (πRL style) | Sparse only: 1.0 success + 0.1 grasp-lift. It needs an SFT start with non-zero success. Rendering on CPU (OSMesa/EGL) will be the bottleneck, so this is a stretch goal. |
| C. Offline filtered or advantage-conditioned (RECAP/AWR) | Per-step −1, 0 on success, −C_fail on failure. Set C_fail to the maximum episode length. Simplest version: keep verifier-passing rollouts and run filtered behaviour cloning. |
| D. Real episodes (the friend's 524, no verifier) | Robometer or VLAC score as a filter or proxy value only. Never optimise against it. Ilia's real-world case shows that without a real-side reward, behaviour cloning plus DAgger is the safe route. |

JEPA is best used as the data engine's judge and a monitoring metric, not an optimised reward. That is INFERRED, because no primary source supports V-JEPA as an RL reward.

**6. Hacking guards (INFERRED, mapped to known failures)**
- **Pushing instead of grasping:** the verifier's both-jaw-lift clause already blocks it (SimpleVLA-RL pushcut).
- **Flinging:** terminate if cube speed exceeds about 1.5 m/s without jaw contact. Require a jaw-contact-free landing speed below a threshold.
- **Penetration and soft-contact abuse (SNIPPET: IndustReal, Midas):** terminate if contact distance is below −2 mm. Re-verify winners at half the timestep.
- **Friction and weld tricks:** re-evaluate every accepted episode under held-out friction and mass draws. Keep the existing no-weld grasp test as a regression test.
- **Farming a shaping term near a threshold (SNIPPET):** pay milestones once, and use running-min progress (Ilia's m_t).
- **Learned-reward exploitation:** never put a learned model in the online reward. Use it for offline scoring, and check against the verifier.

**PAYWALLED:** none. Every source was open on arXiv. Unread tails of the long pages: RECAP after 100k characters, Ilia's paper after 140k, X-Sim, Human2Sim2Robot (last 11.8k), Robometer (165k), RoboReward and V-JEPA 2.