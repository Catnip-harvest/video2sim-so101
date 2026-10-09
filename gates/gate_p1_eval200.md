PASS (result, no pass mark): final MolmoAct2 (ckpt 023400) on 200 held-out in-range scenes (seed 3000): L_full 137/200 = 68.5 % (Wilson 95 % 61.8-74.5 %), L_task 158/200 = 79.0 % (72.8-84.1 %)
# Gate P1.1: MolmoAct2 200-trial in-range eval (headline level)
- Checkpoint: outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model (on Drive: gdrive:so101-sprint/models/molmoact2_sim_v1_023400, gate_drive_molmo_final.md).
- Seed 3000 ids 0-199: never used in training (seed 1), the 20-trial smoke (1000) or probes (9000).
- Harness flags (config.json): interp=lag, obs_codec=h264, max_sim_s=40, workers=1, OMP 8.
- Failure reasons among the 63 L_full failures (an episode can have several): never lifted / never carried 42,
  arm_table_collision 16, self_collision 11, torque_saturation 10, penetration 5, object_disturbed_before_grasp 4,
  release_too_high 4, dropped_or_slipped 3, tracking_error 1. First reason: never_carried 28, arm_table_collision 15,
  self_collision 8, penetration 5, object_disturbed_before_grasp 3, dropped_or_slipped 3, release_too_high 1.
- 21 episodes put the cube in the tray (L_task) but broke a physics clause; 2 of those were style-only.
- Throughput 301 episodes/h (1 process, GPU otherwise mostly idle); median chunk inference 360 ms; 4.1 % of frames rate-clipped.
- Command (CONTAINER, cwd /work/wt-eval): nice -n 5 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval/sim_eval.py --mode policy
  --checkpoint /work/outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model --seed 3000 --n 200 --videos 8 --workers 1
  --device cuda --out /work/outputs_large/eval/policy_final_n200_seed3000
- Log: logs/eval_policy_final_n200.log (EXIT=0). Artefacts: outputs_large/eval/policy_final_n200_seed3000 (summary.json,
  episodes.jsonl, videos/ 8 mp4). Drive: gdrive:so101-sprint/eval/policy_final_n200_seed3000 via the uploader loop
  (rollouts/ excluded); rclone check pending on the uploader's next pass.
