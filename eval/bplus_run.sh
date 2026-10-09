#!/usr/bin/env bash
# FINAL_DESIGN section 5: "self-improvement by filtered BC" + controls, as one resumable script.
# Run inside `sprint` from the repo root (/work/wt-eval). Every stage logs to
# /work/logs/eval_bplus_<stage>.log (ends with EXIT=<code>) and is resumable/skippable.
#
#   SFT=/work/outputs_large/train/molmoact2_sim_v1/checkpoints/<last>/pretrained_model \
#   K=8 STEPS_R2=3000 bash eval/bplus_run.sh                 # all stages in order
#   STAGES="probe" ... bash eval/bplus_run.sh                 # one or more stages
#   TEST=1 bash eval/bplus_run.sh                             # CPU dry run: expert stands in
#                                                             # for the policy, tiny N, no training
# Stages (default order):
#   probe      throughput probe: 16 eval episodes at K=1, then at K (pick K before the big runs)
#   sft_eval   SFT policy on the paired eval seeds (seed 1000, ids 0..N_EVAL-1)
#   ctrlB_eval control B: SFT + inference-time smoothing (SMOOTH=ema|blend) on the same seeds
#   rollouts   SFT policy on the data seeds (seed 2000, ids 0..N_ROLL-1), every rollout saved
#   kept_ds    verifier-passed (L_full) rollouts -> LeRobot dataset (re-rendered), kept_ids.json
#   ctrlA_ds   control A: same number of fresh expert episodes on the same seed-2000 scenes
#   merge      SFT data + kept  /  SFT data + control A   (one dataset each, pretrained norm)
#   train      round 2 for both arms from the SFT checkpoint, same STEPS_R2 (sequential)
#   r2_eval    both round-2 checkpoints on the paired eval seeds
#   compare    paired McNemar + Wilson for every pair of conditions
set -uo pipefail
cd "$(dirname "$0")/.."
TEST="${TEST:-0}"
B="${B:-/work/outputs_large/bplus}"
SFT_DS="${SFT_DS:-/work/outputs_large/datagen/sugar_v10_n1000_molmoact2}"
K="${K:-8}"; N_EVAL="${N_EVAL:-400}"; N_ROLL="${N_ROLL:-500}"
EVAL_SEED="${EVAL_SEED:-1000}"; ROLL_SEED="${ROLL_SEED:-2000}"
SMOOTH="${SMOOTH:-ema}"; EMA_ALPHA="${EMA_ALPHA:-0.5}"; REPLAN="${REPLAN:-5}"
CPU_W="${CPU_W:-6}"
STEPS_R2="${STEPS_R2:-}"; SAVE_R2="${SAVE_R2:-1000}"
STAGES="${STAGES:-probe sft_eval ctrlB_eval rollouts kept_ds ctrlA_ds merge train r2_eval compare}"
if [ "$TEST" = 1 ]; then
  B="${B_TEST:-/work/outputs_large/eval/test_bplus/pipeline}"
  N_EVAL="${N_EVAL_TEST:-6}"; N_ROLL="${N_ROLL_TEST:-10}"; K="${K_TEST:-3}"
  STAGES="${STAGES_TEST:-sft_eval ctrlB_eval rollouts kept_ds ctrlA_ds merge train compare}"
  SFT="${SFT:-/work/outputs_large/train/molmoact2_sim_v1/checkpoints/001400/pretrained_model}"
  STEPS_R2="${STEPS_R2:-3000}"
fi
SFT="${SFT:?set SFT to the final SFT pretrained_model dir}"
mkdir -p "$B" /work/logs
export HF_HOME=/work/hf_home

pol() {  # policy-mode eval, or the expert stand-in in TEST mode
  local ckpt="$1" seed="$2" n="$3" out="$4"; shift 4
  if [ "$TEST" = 1 ]; then
    nice -n 19 python eval/sim_eval.py --mode expert --seed "$seed" --n "$n" --workers "$K" --out "$out" "$@"
  else
    python eval/sim_eval.py --mode policy --checkpoint "$ckpt" --seed "$seed" --n "$n" --workers "$K" \
      --out "$out" "$@"
  fi
}

run() {  # run <stage> <command...>; logs, EXIT line, stops the script on failure
  local st="$1"; shift
  local log="/work/logs/eval_bplus_${st}$([ "$TEST" = 1 ] && echo _test).log"
  echo "[bplus] $(date -u +%H:%M:%S) stage $st -> $log"
  ( "$@" ) > "$log" 2>&1
  local rc=$?
  echo "EXIT=$rc" >> "$log"
  if [ $rc -ne 0 ]; then echo "[bplus] stage $st failed (EXIT=$rc), see $log"; exit $rc; fi
}

ckpt_last() {  # last pretrained_model of a lerobot output dir
  ls -d "$1"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort | tail -1
}

for st in $STAGES; do
  case "$st" in
    probe)
      run probe bash -c "
        python eval/sim_eval.py --mode policy --checkpoint '$SFT' --seed 9000 --n 8 --workers 1 --fresh --out '$B/probe_k1' &&
        python eval/sim_eval.py --mode policy --checkpoint '$SFT' --seed 9000 --n $((2*K)) --workers $K --fresh --out '$B/probe_k$K' &&
        grep -h 'episodes_per_hour_steady\|inference_chunk' '$B/probe_k1/summary.json' '$B/probe_k$K/summary.json'" ;;
    sft_eval)
      run sft_eval pol "$SFT" "$EVAL_SEED" "$N_EVAL" "$B/eval_sft" ;;
    ctrlB_eval)
      run ctrlB_eval pol "$SFT" "$EVAL_SEED" "$N_EVAL" "$B/eval_sft_$SMOOTH" --smooth "$SMOOTH" \
        --ema-alpha "$EMA_ALPHA" --replan-every "$REPLAN" ;;
    rollouts)
      run rollouts pol "$SFT" "$ROLL_SEED" "$N_ROLL" "$B/roll_sft" ;;
    kept_ds)
      if [ -f "$B/ds_kept/meta/kept_ids.json" ]; then echo "[bplus] ds_kept exists, skip"; else
      run kept_ds nice -n 19 python eval/rollouts_to_dataset.py --runs "$B/roll_sft" --out "$B/ds_kept" \
        --workers "$CPU_W"; fi ;;
    ctrlA_ds)
      if [ -f "$B/ds_ctrlA/meta/control_a.json" ]; then echo "[bplus] ds_ctrlA exists, skip"; else
      run ctrlA_ds nice -n 19 python eval/control_a.py --kept "$B/ds_kept/meta/kept_ids.json" \
        --rollout-range "$N_ROLL" --out "$B/ctrlA_run" --dataset-dir "$B/ds_ctrlA" --workers "$CPU_W"; fi ;;
    merge)
      for arm in kept ctrlA; do
        if [ -f "$B/ds_r2_$arm/meta/info.json" ]; then echo "[bplus] ds_r2_$arm exists, skip"; continue; fi
        run "merge_$arm" nice -n 19 python eval/merge_datasets.py --src "$SFT_DS" "$B/ds_$arm" --out "$B/ds_r2_$arm"
      done ;;
    train)
      : "${STEPS_R2:?set STEPS_R2 (same for both arms)}"
      for arm in kept ctrlA; do
        if [ "$TEST" = 1 ]; then
          run "train_$arm" env INIT="$SFT" DATA="$B/ds_r2_$arm" STEPS="$STEPS_R2" SAVE="$SAVE_R2" \
            NAME="r2_${arm}_test" DRY=1 bash env/train_round2.sh
        else
          if [ -n "$(ckpt_last /work/outputs_large/train/r2_$arm)" ] && \
             [ -d "/work/outputs_large/train/r2_$arm/checkpoints/$(printf %06d "$STEPS_R2")" ]; then
            echo "[bplus] r2_$arm already trained, skip"; continue; fi
          echo "[bplus] $(date -u +%H:%M:%S) training r2_$arm (log /work/logs/train_r2_$arm.log)"
          INIT="$SFT" DATA="$B/ds_r2_$arm" STEPS="$STEPS_R2" SAVE="$SAVE_R2" NAME="r2_$arm" \
            bash env/train_round2.sh
          tail -1 "/work/logs/train_r2_$arm.log" | grep -q "EXIT=0" || { echo "train r2_$arm failed"; exit 1; }
        fi
      done ;;
    r2_eval)
      for arm in kept ctrlA; do
        ck="$(ckpt_last /work/outputs_large/train/r2_$arm)"
        [ -n "$ck" ] || { echo "no checkpoint for r2_$arm"; exit 1; }
        run "r2_eval_$arm" pol "$ck" "$EVAL_SEED" "$N_EVAL" "$B/eval_r2_$arm"
      done ;;
    compare)
      run compare bash -c "
        for pair in 'eval_sft eval_r2_kept' 'eval_r2_ctrlA eval_r2_kept' 'eval_sft_$SMOOTH eval_r2_kept' \
                    'eval_sft eval_sft_$SMOOTH' 'eval_sft eval_r2_ctrlA'; do
          set -- \$pair
          if [ -f '$B'/\$1/episodes.jsonl ] && [ -f '$B'/\$2/episodes.jsonl ]; then
            python eval/compare.py '$B'/\$1 '$B'/\$2 --json '$B'/compare_\$1__\$2.json
          else echo \"skip \$1 vs \$2 (missing)\"; fi
        done" ;;
    *) echo "unknown stage $st"; exit 2 ;;
  esac
done
echo "[bplus] $(date -u +%H:%M:%S) done: $STAGES"
