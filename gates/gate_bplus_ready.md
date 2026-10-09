PASS: the filtered-BC pipeline (FINAL_DESIGN §5) and its controls are scripted and CPU-tested end to end with the expert standing in for the policy (`TEST=1 bash eval/bplus_run.sh`: 2 min 6 s, all 4 written datasets pass `validate_dataset.py --lerobot`, 17/17 tests). GPU parts not yet run: K-replica throughput, BlendSource's real-model chunk call, the two round-2 trainings.

# Gate: B+ (filtered BC + controls) ready to run (9 Oct, branch eval/harness, commit 5ff6b47)

All code is in /root/work/wt-eval. Nothing in this gate used the GPU.

## The one command (after training ends)

```
docker exec -it -w /work/wt-eval sprint bash
SFT=/work/outputs_large/train/molmoact2_sim_v1/checkpoints/<last>/pretrained_model \
K=8 N_EVAL=400 N_ROLL=500 STEPS_R2=<steps> SMOOTH=ema bash eval/bplus_run.sh
```

- Outputs go to `/work/outputs_large/bplus/`.
- Each stage logs to `/work/logs/eval_bplus_<stage>.log`, ending with `EXIT=`.
- Re-running resumes:
  - eval runs skip episode ids already in `episodes.jsonl`;
  - dataset stages skip when their output exists;
  - training skips when the final checkpoint exists.
- `STAGES="probe"` runs a single stage. Run the probe first to choose K.

| stage | what | GPU | CPU (measured) |
|---|---|---|---|
| probe | 8 episodes at K=1, then 2K episodes at K → episodes/h | yes | — |
| sft_eval | SFT on seed 1000, ids 0..399 (the paired list for every condition) | yes | about 3 s CPU per 18 s-sim episode. Flat from 1 to 6 workers (2.73 → 2.96 s), so 8 replicas are GPU-bound, not CPU-bound |
| ctrlB_eval | control B: SFT + `--smooth ema` (alpha 0.5) or `blend` (re-plan every 5 steps, linear cross-fade; 2× inference), same 400 seeds | yes | same |
| rollouts | SFT on seed 2000, ids 0..499. Every rollout saved (state, commanded goal, raw action, cube pose, full qpos, verdict) | yes | same |
| kept_ds | L_full-passed rollouts → LeRobot v3. Re-rendered from qpos with datagen's renderer, then `use_pretrained_norm.py`. Writes `kept_ids.json` | no | render 11.7–12.7 s per episode per worker. About 11 min for 300 kept on 6 workers |
| ctrlA_ds | control A: `generate.py`'s own `_attempt` and `finalize` on the kept seed-2000 ids. Any id the expert fails is topped up from not-kept ids, so the count stays equal. Then pretrained norm | no | about 17 s per episode per worker (8 episodes in 34.5 s on 4 workers). About 14 min for 300 on 6 workers |
| merge | SFT data + kept, and SFT data + control A (videos hard-linked, exact image stats) | no | 1,008 episodes in 2.1 s |
| train | `env/train_round2.sh` for both arms from the SFT checkpoint, same `STEPS_R2`, batch 16, lr 1e-5, sequential | yes | — |
| r2_eval | both round-2 checkpoints on the 400 paired seeds | yes | same as sft_eval |
| compare | `eval/compare.py` (exact McNemar + Wilson) for: SFT vs r2_kept, r2_ctrlA vs r2_kept, SFT+smooth vs r2_kept, SFT vs SFT+smooth, SFT vs r2_ctrlA | no | seconds |

**Rough GPU timeline:**
- 1 replica measured 116 episodes/h while sharing the GPU with training.
- If K=8 gives about 4× (unmeasured), the 4 × 400 eval episodes plus 500 rollouts ≈ 4.5 h.
- Training adds 2 × `STEPS_R2` × about 1.2 s.
- Example: 3000 steps ≈ 2 h, so the full pipeline is about 7 h.
- If the probe is slow, cut `N_EVAL` (e.g. 200) before cutting `N_ROLL`.

## What was checked on CPU

1. **Rollout → dataset format.**
   - `info.json` features are identical to `sugar_v10_n1000_molmoact2`: same fps 10, MolmoAct2 frame, cam_head/cam_right 256×256 h264, same task string.
   - `stats.json` has the same keys, and action q01/q99 equal the SFT set's (the pretrained norm).
   - Expert-stand-in rollouts against Control A on the same 8 seed-2000 scenes:
     - action columns are identical (max diff 0.0°);
     - states differ by median 0.09° (max 7.0°), from the harness's 67 ms ramp lag.
2. **Control A top-up.**
   - kept = {30, 31, 32} on seed 1000: the expert fails 32, so id 0 is added to keep the count at 3.
3. **Merge.**
   - SFT (1,000 episodes) + 8 gives 1,008 episodes and 188,807 frames.
   - Merged frames and decoded images equal the sources (max diff 0.0 on episodes 65 and 1003).
   - The image mean is the count-weighted combination.
4. **Round-2 launcher.**
   - Plain `lerobot-train --policy.path=<molmoact2 checkpoint>` fails in lerobot 0.6.1 with `KeyError: override keys ['normalizer_processor'] do not match any step`. This was reproduced in a test; it would fail after the 11 GB model load.
   - `env/train_round2.py` drops the override keys that the saved pipeline does not have. That keeps the checkpoint's own normaliser, which equals the dataset's pretrained stats (checked by `--check`) and also carries the gripper mask.
   - The test shows the unpatched call raises and the patched call loads.
   - `DRY=1` parses and validates the full config (policy.path, dataset, steps).
5. **Control B.**
   - EMA and blend are unit-tested.
   - Through the harness with the expert as the source: 20/20 for both.
   - The postprocessor gives identical results on a (1,10,6) chunk and per step.
6. **McNemar.** p for b=10, c=2 is 0.0386, equal to `scipy.stats.binomtest`.

## Still needs the GPU (cannot be tested now)

- **Throughput at K=4–8:** run the `probe` stage. 8 × 12–14 GB ≈ 110 GB of VRAM.
- **`PolicyRunner.chunk`** (`predict_action_chunk` + postprocessor) with the real model. It is only needed for `SMOOTH=blend`; the default `ema` does not use it.
- **The two round-2 trainings.** Model loading uses the same `from_pretrained` path the eval already uses, but a real step has not been run.

## Choices to confirm

1. **Control B smoothing:** default `ema` with alpha 0.5. `blend` doubles the inference cost.
2. **Round-2 STEPS_R2:** no default; the same value is used for both arms. The suggestion is 3000, about 1 h per arm at batch 16.
3. **Rollout labels:** each kept rollout's action label is the goal the harness commanded (the policy action, clipped to joint range). The SFT labels are the expert's 30 Hz target at that frame. Both are absolute joint goals.
4. **Early stop:** rollouts stop 1 s after the task is done, so kept episodes end before the arm goes home. Expert episodes end at home.
