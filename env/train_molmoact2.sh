#!/usr/bin/env bash
# MolmoAct2-SO100_101 fine-tune on a dataset exported with generate.py --export-frame molmoact2
# (+ use_pretrained_norm.py). Run inside `sprint`. Logs end with EXIT=<code>.
#   DATA=/work/outputs_large/datagen/<run>_molmoact2 STEPS=22000 SAVE=1500 bash env/train_molmoact2.sh
# Choices (see LEDGER / gates):
#   chunk_size 10 at 10 Hz = 1 s, the same time span as pretraining (30 steps at 30 Hz).
#   No norm_tag: it would force chunk_size=30; setup/control prompt strings are copied
#   from the checkpoint's norm_stats.json metadata instead. normalize_gripper as in pretraining.
#   Normalisation stats: the checkpoint's own (written into the dataset by use_pretrained_norm.py).
set -uo pipefail
DATA="${DATA:?set DATA}"; STEPS="${STEPS:-22000}"; SAVE="${SAVE:-1500}"; BS="${BS:-16}"
OUT="${OUT:-/work/outputs_large/train/molmoact2_sim_$(date -u +%m%d_%H%M)}"
NAME="${NAME:-$(basename "$OUT")}"
LOG="/work/logs/train_${NAME}.log"
export HF_HOME=/work/hf_home HF_HUB_OFFLINE=1
{
lerobot-train \
  --dataset.repo_id=local/"$(basename "$DATA")" --dataset.root="$DATA" --dataset.video_backend=pyav \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
  --policy.model_dtype=bfloat16 --policy.enable_inference_cuda_graph=false --policy.push_to_hub=false \
  --policy.setup_type="single so100/so101 robotic arm in molmoact2" \
  --policy.control_mode="absolute joint pose" \
  --policy.chunk_size=10 --policy.n_action_steps=10 --policy.normalize_gripper=true \
  --batch_size="$BS" --steps="$STEPS" --save_freq="$SAVE" --log_freq=50 --num_workers=12 \
  --save_checkpoint=true --wandb.enable=false --output_dir="$OUT" --job_name="$NAME"
echo "EXIT=$?"
} > "$LOG" 2>&1
