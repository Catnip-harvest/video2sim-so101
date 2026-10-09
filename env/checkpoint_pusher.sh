#!/usr/bin/env bash
# Push every new lerobot checkpoint (pretrained_model/ only) of a training run to a PRIVATE
# HF model repo, then prune local checkpoints to the newest KEEP. Run inside `sprint`:
#   RUN=/work/outputs_large/train/<run> REPO=VietHwang/<repo> nohup bash env/checkpoint_pusher.sh &
# Logs to /work/logs/ckpt_pusher_<run>.log; each success adds a PUSHED line to logs/hf_push.log.
set -uo pipefail
RUN="${RUN:?}"; REPO="${REPO:?}"; KEEP="${KEEP:-2}"; HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="/work/logs/ckpt_pusher_$(basename "$RUN").log"
export HF_HOME=/work/hf_home
unset HF_HUB_OFFLINE
{
while true; do
  for ck in $(ls -d "$RUN"/checkpoints/[0-9]* 2>/dev/null | sort); do
    step=$(basename "$ck")
    [ -f "$ck/.pushed" ] && continue
    [ -f "$ck/pretrained_model/model.safetensors" ] || [ -n "$(ls "$ck"/pretrained_model/*.safetensors 2>/dev/null)" ] || continue
    sleep 60  # let the trainer finish writing
    echo "$(date -u +%T) pushing $step"
    if bash "$HERE/push_hf.sh" model "$REPO" "$ck/pretrained_model" "checkpoints/$step"; then
      touch "$ck/.pushed"
    else
      echo "$(date -u +%T) push of $step FAILED; retry next loop"
    fi
  done
  # prune: keep the newest $KEEP checkpoints, only ones already pushed
  ls -d "$RUN"/checkpoints/[0-9]* 2>/dev/null | sort | head -n -"$KEEP" | while read -r old; do
    [ -f "$old/.pushed" ] && rm -rf "$old" && echo "$(date -u +%T) pruned $(basename "$old")"
  done
  grep -q "^EXIT=" "/work/logs/train_$(basename "$RUN").log" 2>/dev/null && \
    [ -z "$(for c in "$RUN"/checkpoints/[0-9]*; do [ -f "$c/.pushed" ] || echo x; done 2>/dev/null)" ] && break
  sleep 120
done
echo "EXIT=0"
} >> "$LOG" 2>&1
