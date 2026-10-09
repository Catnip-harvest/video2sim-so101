# Ledger
AFTER ANY WAKE, RESTART OR COMPACTION: re-read /root/work/OVERNIGHT_BRIEF.md and the last 20 LEDGER lines.
- Droplet created ~08:57 UTC 9 Oct 2026 (MI300X, $1.99/h).
- 09:34 UTC 9 Oct: ~0.6 h alive (~$1.2). Step 1 render A/B done (see below). π0.5 blocked on gated PaliGemma tokenizer.
  - Render, 19 workers, LP_NUM_THREADS=1, 2 cams: full 45 img/s (0.03 eps/s), shadow 117 (0.11), fast 421 (0.41); 224 px fast 438.
  - N=20 dataset: 20/20 accepted, LeRobotDataset load 18/18 checks pass.
- 09:59 UTC 9 Oct: ~1.0 h alive (~$2.0).
  - MolmoAct2-SO100_101 smoke (bf16, cuda graphs off, sim n20/n60 data): PASS, loss finite.
    bs8: 0.73 s/step update, 81 GB (lerobot) / 90.8 GB GPU peak. bs16: 1.23 s/step, data 0.09 s (8 workers), 131 GB / 146 GB peak.
  - End-to-end datagen, --render-quality fast, 15 workers, 30 Hz, 2 cams: 0.34-0.36 eps/s, 60/60 accepted.
  - PaliGemma tokenizer still 403 gated at 09:59.
- 10:18 UTC 9 Oct: ~1.4 h alive (~$2.8).
  - Gate 1a render: PASS with --record-hz 10 --render-quality fast --variety: 0.92 att/s (16 workers), 96.8 % accepted.
  - Datagen N=1000 (10 Hz, variety, rejects kept) running -> outputs_large/datagen/sugar_v10_n1000.
  - Overlay gate G0 (mask IoU >= 0.8) running in a subagent. Branches pushed: calib/so101-units, droplet/step1-render-smoke, datagen/10hz-variety (PRs to be opened by owner; no gh token on box).
- 10:55 UTC 9 Oct: ~2.0 h alive (~$4).
  - Datagen N=1000 done: 1000/1055 accepted (94.8 %; 52 arm_table_collision, 3 unstable), 187,401 frames @10 Hz.
    Export (molmoact2 frame, pretrained norm, cams cam_head/cam_right): 68/68 + 18/18 checks.
    Pushed (private): datasets VietHwang/so101-sim-sugarcube-10hz-molmoact2, VietHwang/so101-sim-sugarcube-10hz.
  - Training molmoact2_sim_v1 started 10:48: 23,400 steps bs16 (2 epochs), ckpt every 1400 -> VietHwang/molmoact2-so101-sim-v1.
    Step 50: updt 1.30 s, data 3.18 s (CPU contention with overlay-gate fits; asked subagent to cap at 4 workers).
- 11:05 UTC: training steady at 1.33 s/step (updt 1.19 + data 0.13) after overlay-gate fits ended; ETA ~19:35 UTC. Loss 5.77 (step 50) -> 0.97 (step 300).
- 11:15 UTC 9 Oct: ~2.3 h alive (~$4.6).
  - Gate G0 overlay: INCONCLUSIVE. Held-out IoU 0.494 (repeat 0.468) vs 0.8; ceiling with per-frame cameras 0.658; cannot tell default from +-20 deg errors. -> sim-only training (owner rule). Code: calib/overlay_gate.py (pushed on calib/so101-units).
  - Started subagents: eval harness + phase-free verifier (worktree wt-eval), JEPA judge (wt-jepa). Both nice 19, <= 6 CPU workers, <= 15-30 GB VRAM.
- 11:50 UTC: 11:37-11:44 full GPU stall (~6.5 min) from 2 policy replicas + training + V-JEPA near VRAM limit; rule now 1 replica, <170 GB. First ckpt 001400 pushed 11:31:54. Early smoke of 001400: 7/15 pass (eval agent, not final).
- 11:55 UTC: Eval harness gate PASS (expert via harness 58/60, replay 5/5, verifier agreement 59/60+20/20+10/10). Ckpt 001400 smoke: 10/20 L_full (50 %, CI 30-70), 15/20 L_task. ~116 eval eps/h per replica. Branch eval/harness pushed.
- 11:59 UTC: JEPA gate FAIL (pre-registered probe 0.784 vs DINOv2 last-frame 0.802, diff -0.018 p=0.59). Secondary (not pre-registered): time-ordered 0.866, wrist video 0.912. Realism FD sim-vs-real 1329 vs floor 15 (viewpoint/scene mismatch). Judge outputs pushed to VietHwang/so101-jepa-judge; branch jepa/judge pushed.
- 12:20 UTC: HF private storage limit hit (002800 refused). Deleted throwaway repo molmoact2-so101-dryrun; HF now keeps newest ckpt only (002800 on HF 12:16:54, 001400 removed + history squashed). Owner decision pending: paid HF plan (default: no).
- 12:20 UTC: B+ pipeline ready (gate PASS, 15 tests pass on re-run; eval/bplus_run.sh). Defaults applied: SMOOTH=ema, STEPS_R2=3000, kept-rollout labels = commanded 10 Hz goal, rollouts end 1 s after task done. Note: lerobot 0.6.1 --policy.path crashes for molmoact2 (processor override name mismatch) -> env/train_round2.py wrapper.
- 19:47 UTC 9 Oct: ~10.8 h alive (~$21.5).
  - Training molmoact2_sim_v1 DONE 19:42:54, EXIT=0: 23,400 steps (2 epochs), 8 h 52 min, loss 5.77 (step 50) -> 0.31 (epoch 1) -> 0.283 (end).
  - HF: every checkpoint push since 12:44 failed (storage limit, 130 failures); HF holds only 002800. Owner (relayed via laptop/watcher): "still do not touch Hugging Face" -> pusher stopped. Final model kept locally: outputs_large/final_models/molmoact2_sim_v1_023400 (+ checkpoints/022400, 023400).
  - Running owner's 20-trial smoke of the final checkpoint (seed 1000) -> outputs_large/eval/policy_final_n20. Then B+ (eval/bplus_run.sh).
- 19:58 UTC: FINAL ckpt 023400, 20-trial smoke (seed 1000): L_full 15/20 (75 %, Wilson 53-89 %), L_task 17/20 (85 %). Vs 001400 (10/20): +6/-1 discordant, McNemar exact p=0.125 (n.s. at n=20). ~210 eps/h single replica on free GPU.
- 20:25 UTC: B+ probe: K=8 pool hung after 7 eps (19:57), K=4 pool hung at 0 eps; workers spin ~180 % CPU with 0 GPU CU occupancy. Killed both. Testing 4 independent single-worker processes (--start shards).
- 20:45 UTC 9 Oct: ~11.8 h alive (~$23.5).
  - Owner (via laptop session): B+ round DROPPED. Stopped all B+ jobs cleanly (code + logs kept: wt-eval eval/bplus_run.sh, logs/eval_bplus_*.log, outputs_large/bplus/). GPU freed.
    B+ probe finding (for the record): multi-replica policy inference hangs on this box (K=8 pool after 7 eps; K=4 pool at 0 eps; 4 independent procs at 0 eps; K=4 pool with OMP=4 after 7 eps). Single process reliable; 2 independent procs with OMP=8 finished 8/8.
  - Started 200-trial eval of FINAL ckpt 023400 on fresh held-out seed 3000 (unused anywhere), 1 process, videos 8 -> outputs_large/eval/policy_final_n200_seed3000.
- 20:55 UTC: Another session (likely laptop, per owner's Google Drive plan) ran rclone: copied a full pretrained_model/ (7 files, model.safetensors 10.137 GiB) 20:51-20:54, EXIT=0 (logs/drive_upload_molmoact2.log). Source/destination not in log; not started by main session. 200-trial eval at 57/200 at 20:59.
- 21:21 UTC 9 Oct (laptop handoff, brief executing): ~12.4 h alive ($24.58). REPORT.md v1 on gdrive:so101-sprint/REPORT.md (3931 B, size-verified). gates/gate_drive_molmo_final.md PASS (rclone check --size-only, 7 files, 0 diffs). Uploader loop started in tmux session 'uploader' (uploader.sh, copy-only, log logs/uploader.log). Queue and watcher extension pending coordination with work-22.
