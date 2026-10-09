#!/usr/bin/env bash
# 20-step MolmoAct2-SO100_101 training smoke test (FINAL_DESIGN.md §10). Run inside `sprint`.
#   BS=8 DATA=/work/manip/datagen/runs/sugar_n20/lerobot bash env/smoke_molmoact2.sh
set -euo pipefail
BS="${BS:-8}"; STEPS="${STEPS:-20}"
DATA="${DATA:-/work/manip/datagen/runs/sugar_n20/lerobot}"
OUT="${OUT:-/work/outputs_large/smoke_molmoact2_bs${BS}}"
rm -rf "$OUT"
export HF_HOME=/work/hf_home
lerobot-train \
  --dataset.repo_id=local/so101_sim_smoke --dataset.root="$DATA" --dataset.video_backend=pyav \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
  --policy.norm_tag=so100_so101_molmoact2 --policy.model_dtype=bfloat16 \
  --policy.enable_inference_cuda_graph=false --policy.push_to_hub=false \
  --batch_size="$BS" --steps="$STEPS" --log_freq=1 --num_workers="${NW:-4}" \
  --save_checkpoint=false --wandb.enable=false --output_dir="$OUT"
