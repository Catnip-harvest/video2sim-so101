# Gate: final MolmoAct2 checkpoint on Drive - PASS

- Checked at: 21:18 UTC Fri 9 Oct 2026 (read-only check; no copy was needed).
- Local: /root/work/outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model (7 files, 10.137 GiB)
- Drive: gdrive:so101-sprint/models/molmoact2_sim_v1_023400/pretrained_model
- Command: `rclone check <local> <drive> --one-way --size-only`
- Result: "0 differences found", "7 matching files", exit code 0.
- Listing (rclone lsf -R): model.safetensors, config.json, train_config.json, policy_preprocessor.json, policy_postprocessor.json, and the two step-normalizer safetensors files.
- Upload itself: 20:51-20:54 UTC, rclone copy, EXIT=0 (logs/drive_upload_molmoact2.log). Its source and operator are not in the log; the likely source is the local final_models path above.
