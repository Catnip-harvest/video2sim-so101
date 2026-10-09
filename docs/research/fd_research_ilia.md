**ILIA / LeHome 2026: findings for the RL question.** I read arXiv 2606.27163v3 up to about char 133k of 140k; the remainder is references. WebFetch summarises the page, so short quotes are as the fetcher returned them. I also read his repo (`IliaLarchenko/lehome_solution` and the challenge fork's success checker) directly through the GitHub API.

**(1) What the "RL loop" is**
- VERIFIED: It is not PPO/GRPO. It is an asynchronous flywheel of on-policy rollouts followed by supervised updates. Quote: "PPO and GRPO are log-probability policy-gradient methods that transfer poorly to flow-matching VLAs."
- VERIFIED: Each iteration the trainer downloads new rollouts, recomputes advantages, trains about 1000 steps, and uploads a checkpoint about every 500 steps. Rollout workers (3–5 parallel Isaac Sim instances each), the trainer and a human DAgger station talk only through the HuggingFace Hub (§2.1).
- VERIFIED: The policy update uses two methods together:
  - **AWR through the sampler.** Frames are sampled with probability proportional to exp(clip(A,−2,2)). The flow-matching loss itself stays plain unweighted MSE (§2.3).
  - **RECAP-style advantage conditioning.** A learned advantage token is an input, masked in training. P(neutral) ramps from 0.5 at A=0 to 0.1 at A≥2, and A<0 is always masked. At inference the token is always on, with classifier-free guidance scale 7–9 (§4.3, §7).
- VERIFIED: Inference-time candidate selection is also used. Best-of-N scores candidates with the Δsuccess head and uses N=2–3, with no gain beyond 3.
- VERIFIED: Warm-up is 20k steps of BC on a scripted dataset, from π0.5 weights. Periodic checkpoint rollbacks were used (3 in round 1, 1 in round 2).
- So it is closest to iterated advantage-weighted and advantage-conditioned BC with fresh on-policy data. It is neither offline RL nor rejection sampling.

**(2) Reward, success and progress signals**
- VERIFIED (code): There is no physical marker on the shirt. Success is computed from privileged simulator state in the organizers' checker (`success_checker_chanllege.py`). Six pre-chosen cloth-mesh particles per garment (`check_points`) are read in cm. Success is a set of pairwise-distance conditions, for example `dist(p[0],p[4]) <= thr`, `dist(p[2],p[3]) <= thr`, and `dist(p[0],p[1]) >= thr` for spreading. A top has 5 conditions, short pants 4, and long pants 7 in his expanded set. The paper says "Each garment carries a set of keypoints…". It does not say where the points come from, but the code answers that.
- Your recollection of "reward when the marker is reached" matches the checkpoints, with the correction that the markers are mesh vertices read from the simulator.
- VERIFIED: The reward is the binary success densified into per-garment checkpoints (§6.1, `compute_dense_reward`):
  - The first checkpoint is worth 0.5 and the full fold brings the total to 1.0 (short pants have a single checkpoint worth 1.0).
  - For tops the first 0.5 is paid gradually: 0.5·clip((d₀−m_t)/(d₀−d_t1), 0, 1), where the distance is the collar/sleeve pair `p[2],p[3]`.
  - If the episode fails, all accumulated reward is withdrawn, spread over the frames after the peak. The return therefore stays equal to the binary outcome.
- VERIFIED: No separate critic. The VLA is its own value function through linear heads on one image-only query token (§5):
  - **Success head:** BCE on the episode outcome, loss weight 0.05, label smoothing 0.05.
  - **Completion head:** MSE on t/T, successful episodes only, weight 0.02.
  - **Keypoint-distance heads:** 21 outputs, plus 30-frame-ahead versions.
- VERIFIED: Advantage is GAE with γ=0.999, λ=0.99 (computed offline each iteration):
  - Value baseline V̂ = α·(S̄−R_cum) with a dampened α=0.5, success EMA-smoothed at 0.2.
  - Completion is added as potential-based shaping with Φ=0.5·C̄.
  - Stale rollouts blend toward a per-garment segment baseline.
  - The top 20% of successes per garment get +0.3 "precision boost", and advantage is clipped to [−2, 2] (§6.5–6.6).
- VERIFIED: The repo has no jury-scoring proxy and no learned reward model. The only learned signal is the self-predicted success, completion and keypoint heads. He calls reward engineering "still an art".

**(3) How much RL improved results over BC**
- VERIFIED: There is no controlled RL-vs-BC ablation. In the §8 and §10 text I read, none of RL, AWR, conditioning, best-of-N or guidance has a before/after number. The paper says online bandit tuning replaced full ablations. The one project-page line: "Behavior cloning on top of it works, but the resulting policy isn't robust enough."
- VERIFIED final numbers:
  - Online round, 62 teams, 20 garments per type × 10 episodes each: 79.63% overall (long top 74.5%, short top 70.0%, long pants 80.5%, short pants 93.5%). Second place was 73.50% (+6.1 pp).
  - Scale: about 12,500 retained rollout episodes, about 4.3M frames, about 140 collection sessions, about 300k training steps on one H200.
  - He says "the approach is not sample-efficient" and that a much smaller dataset might suffice.
- VERIFIED: He calls inference-time tuning (execution length, guidance, N via Thompson-sampling bandit) "a big win" that needed no training. The FM head's correlation with outcomes was about zero, yet N=2–3 still helped (his guess: it avoids bad chunks at rare multimodal states).
- INFERRED: Treat "RL adds X points" as unmeasured. His evidence is a leaderboard result plus engineering judgement.

**(4) Simulation vs real**
- VERIFIED: All RL ran in Isaac Sim. The real round was a one-week sprint that ran as plain BC "because there was no real-side reward or value".
- VERIFIED: The real policy was fine-tuned from a slightly older sim checkpoint with the keypoint, world-model and advantage parts removed, and with no CFG or best-of-N at inference. The data mix was:

| Source | Episodes | Share |
|---|---|---|
| Organizer BC | 500 | 60% |
| Own teleop (321) + DAgger (471), 288k DAgger frames | 792 | 30% |
| Sim success-replays | 1,723 | 10% |

- Result: 2nd place, 865/1080 (first place 895). He guesses "90%+ is achievable" with a real-side value model plus value-weighted DAgger.
- VERIFIED: Real episodes were not turned into reward or aggregated through the engine. They were mixed at the dataset-sampling level, and real DAgger frames carry human-vs-policy labels. `scripts/replay_real_in_sim.py` replays a real action stream in sim only to check sim/real alignment ("we are NOT trying to reproduce the real trajectory"). A unit mismatch (degrees vs normalised) cost him about two days.

**(5) Lessons for rigid pick-and-place on SO-101**
- VERIFIED, his own words:
  - "the only reliable source of recovery was human intervention"
  - RL is "great at reproducing and sharpening behavior it has seen" but does not explore out-of-distribution states.
  - His scripted BC set had little diversity and no recovery behavior. This is the same weakness as your single scripted expert.
  - "keep interventions short" (short DAgger).
  - Heavy augmentation and rig randomization drove robustness. "Cheap image augmentations are underused in robotics."
  - The model could be "much smaller".
  - Check units early.
- INFERRED, mapping to your project:
  - Your physics verifier is already his success checker. A dense reward would be checkpoints such as grasped (both jaws in contact), lifted, over bowl, released and in bowl. Paying a gradual reward for the distance closed on the hardest checkpoint and withdrawing everything on failure keeps the return binary.
  - Use his RL form (success-head value, AWR/conditioning), not PPO, on a flow-matching π0.5.
  - RL's likely gain for you is robustness: failure-state retries and recoveries from your sim rollouts. This is unmeasured in his work.
  - Real episodes enter as BC data with an explicit sampling share. Real-side reward needs a value model that he did not build.
  - Your 30 fps rendered data at 20 vCPU is CPU-bound, which is the opposite of his GPU Isaac setup.

**PAYWALLED:** none. Unread: the last ~7.5k characters of the arXiv HTML (references).

Sources:
- https://arxiv.org/html/2606.27163v3
- https://github.com/IliaLarchenko/lehome_solution
- https://github.com/IliaLarchenko/lehome-challenge (success checker `source/lehome/lehome/utils/success_checker_chanllege.py`)
- https://ilialarchenko.com/projects/lehome2026/

Repo files I downloaded for the code checks are in `C:\Users\vieth\AppData\Local\Temp\claude\C--Users-vieth-Documents-lablab-hackathon\95967455-696f-4378-93c2-58499e16ccc5\scratchpad\ilia`.