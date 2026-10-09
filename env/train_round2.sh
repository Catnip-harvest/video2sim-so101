#!/usr/bin/env bash
# Filtered-BC round 2 (FINAL_DESIGN section 5): continue from the SFT checkpoint on a merged
# dataset (SFT data + kept rollouts, or SFT data + control-A expert episodes). Both arms must
# use the SAME STEPS/BS/LR (equal budget). Run inside `sprint`. Logs end with EXIT=<code>.
#   INIT=/work/outputs_large/train/molmoact2_sim_v1/checkpoints/<last>/pretrained_model \
#   DATA=/work/outputs_large/bplus/ds_merged_kept STEPS=3000 SAVE=1000 NAME=r2_kept bash env/train_round2.sh
# DRY=1 only parses/validates the config (CPU, no model load).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
INIT="${INIT:?set INIT (SFT pretrained_model dir)}"; DATA="${DATA:?set DATA}"
STEPS="${STEPS:?set STEPS (same for every arm)}"; SAVE="${SAVE:-1000}"; BS="${BS:-16}"
NAME="${NAME:?set NAME}"; OUT="${OUT:-/work/outputs_large/train/$NAME}"
LR_ARGS=()
[ -n "${LR:-}" ] && LR_ARGS=(--policy.optimizer_lr="$LR")
LOG="/work/logs/train_${NAME}.log"
export HF_HOME=/work/hf_home HF_HUB_OFFLINE=1
python "$HERE/train_round2.py" --check "$INIT" "$DATA" || { echo "stats check failed"; exit 1; }
ARGS=(
  --policy.path="$INIT" --policy.enable_inference_cuda_graph=false --policy.push_to_hub=false
  --policy.device=cuda
  --dataset.repo_id=local/"$(basename "$DATA")" --dataset.root="$DATA" --dataset.video_backend=pyav
  --batch_size="$BS" --steps="$STEPS" --save_freq="$SAVE" --log_freq=50 --num_workers=12
  --save_checkpoint=true --wandb.enable=false --output_dir="$OUT" --job_name="$NAME" "${LR_ARGS[@]}"
)
if [ "${DRY:-0}" = 1 ]; then
  python "$HERE/train_round2.py" --dry-run "${ARGS[@]}"
  exit $?
fi
{
python "$HERE/train_round2.py" "${ARGS[@]}"
echo "EXIT=$?"
} > "$LOG" 2>&1
