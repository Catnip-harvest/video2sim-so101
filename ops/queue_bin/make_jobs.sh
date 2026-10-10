#!/usr/bin/env bash
# Write queue job scripts for one model's P2.4 evals (run from the host).
#   bash /root/work/queue/bin/make_jobs.sh <prefix 2|3|4> <tag act|smolvla|actwide> <checkpoint pretrained_model dir> [s3000_n]
# Jobs: <prefix>00_<tag>_s3000 (seed 3000 ids 0..N-1), <prefix>01_<tag>_s4000_none, then one per ladder rung
# that MolmoAct2 ran (<prefix>02.. camera exposure colour combo distractor camera_height camera_pitch
# camera_yaw camera_distance), all seed 4000 ids 0-49. Code: /work/wt-p24 (eval/ood-ladder +
# small-model camera rename). Harness flags = defaults (same as P1.1 / MolmoAct2 rungs).
set -euo pipefail
P="$1"; TAG="$2"; CK="$3"; N3="${4:-200}"
Q=/root/work/queue
job() {  # name seed n ood out
  local name="$1" seed="$2" n="$3" ood="$4" out="$5" f="$Q/$1.sh"
  local oodarg=""; [ "$ood" != "-" ] && oodarg="--ood $ood"
  cat > "$f" <<EOF
#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# P2.4 ${TAG}: seed ${seed} ids 0-$((n-1)) ${ood}
docker exec sprint bash -lc 'cd /work/wt-p24 && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint ${CK} --seed ${seed} --n ${n} --videos 4 --workers 1 --device cuda ${oodarg} --out /work/outputs_large/eval/${out} > /work/logs/eval_${out}.log 2>&1; echo EXIT=\$? >> /work/logs/eval_${out}.log'
EOF
  echo "$f"
}
job "${P}00_${TAG}_s3000" 3000 "$N3" - "${TAG}_s3000"
job "${P}01_${TAG}_s4000_none" 4000 50 none "${TAG}_s4000_none"
i=2
for r in camera exposure colour combo distractor camera_height camera_pitch camera_yaw camera_distance; do
  job "$(printf "%s%02d" "$P" "$i")_${TAG}_s4000_${r}" 4000 50 "$r" "${TAG}_s4000_${r}"
  i=$((i+1))
done
