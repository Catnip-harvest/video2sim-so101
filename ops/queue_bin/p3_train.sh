#!/usr/bin/env bash
# P3-A-lite training (brief "Training A"): SmolVLA-multi + ACT-multi-no-language TOGETHER, only once the policy
# queue has drained (P2.4 + P2.5 evals done) so the GPU holds exactly these 2 processes. Last start 07:00 UTC
# (brief); if the queue is not empty by then, P3 training is dropped and that is logged.
# HOST: nohup bash /root/work/queue/bin/p3_train.sh > /root/work/logs/p3_train_launcher.log 2>&1 &
set -uo pipefail
Q=/root/work/queue; LEDGER=/root/work/LEDGER.md
DEADLINE=$(date -u -d '2026-10-10 07:00' +%s)
busy() {
  ls $Q/*.sh >/dev/null 2>&1 && return 0
  ps -eo args | grep -E '^(/opt/venv/bin/)?python[0-9.]* .*(lerobot-train|eval/sim_eval.py --mode policy)' | grep -vq grep && return 0
  return 1
}
while busy; do
  if [ "$(date -u +%s)" -ge "$DEADLINE" ]; then
    echo "$(date -u +%T) DEADLINE 07:00 reached with queue/GPU busy -> P3 training DROPPED"
    echo "- $(date -u +%H:%M) UTC: P3-A training DROPPED: P2.4/P2.5 evals or P2.3 runs still occupying the GPU at the 07:00 last-start (brief rule)." >> $LEDGER
    echo EXIT=0; exit 0
  fi
  sleep 60
done
echo "$(date -u +%T) queue empty and GPU free -> starting P3-A trainings"
echo "- $(date -u +%H:%M) UTC: P3-A trainings started (SmolVLA-multi 15k bs64 + ACT-multi-no-language 30k bs32, together)." >> $LEDGER
ENV='export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1'
D=/work/outputs_large/datagen/multitask_A_v1_small
docker exec -d sprint bash -lc "cd /work/wt-multi && $ENV && lerobot-train --policy.path=lerobot/smolvla_base --dataset.root=$D --dataset.repo_id=local/multitask_A_v1_small --dataset.video_backend=pyav --rename_map='{\"observation.images.cam_head\":\"observation.images.camera1\",\"observation.images.cam_right\":\"observation.images.camera2\"}' --policy.empty_cameras=1 --policy.push_to_hub=false --policy.device=cuda --batch_size=64 --steps=15000 --policy.scheduler_decay_steps=15000 --save_freq=5000 --log_freq=200 --num_workers=6 --wandb.enable=false --output_dir=/work/outputs_large/train/smolvla_multi_bs64 > /work/logs/train_smolvla_multi_bs64.log 2>&1; echo EXIT=\$? >> /work/logs/train_smolvla_multi_bs64.log"
docker exec -d sprint bash -lc "cd /work/wt-multi && $ENV && lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.n_action_steps=50 --dataset.root=$D --dataset.repo_id=local/multitask_A_v1_small --dataset.video_backend=pyav --batch_size=32 --steps=30000 --save_freq=10000 --log_freq=200 --num_workers=6 --wandb.enable=false --output_dir=/work/outputs_large/train/act_multi_nolang_bs32 > /work/logs/train_act_multi_nolang_bs32.log 2>&1; echo EXIT=\$? >> /work/logs/train_act_multi_nolang_bs32.log"
echo "$(date -u +%T) launched both"
echo EXIT=0
