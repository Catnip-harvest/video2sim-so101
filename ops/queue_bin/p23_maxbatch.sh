#!/usr/bin/env bash
# P2.3 extension: largest batch that fits each emulated cap (300 steps each, one training at a time).
# HOST: nohup bash /root/work/queue/bin/p23_maxbatch.sh > /root/work/logs/p23_maxbatch.log 2>&1 &
set -uo pipefail
ENV='export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1'
SMALL=/work/outputs_large/datagen/sugar_v10_n1000_small
SMOL="--policy.path=lerobot/smolvla_base --dataset.root=$SMALL --dataset.repo_id=local/sugar_v10_n1000_small --dataset.video_backend=pyav --rename_map='{\"observation.images.cam_head\":\"observation.images.camera1\",\"observation.images.cam_right\":\"observation.images.camera2\"}' --policy.empty_cameras=1 --policy.push_to_hub=false --policy.device=cuda --steps=300 --log_freq=50 --num_workers=6 --wandb.enable=false --save_checkpoint=false"
ACT="--policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.n_action_steps=50 --dataset.root=$SMALL --dataset.repo_id=local/sugar_v10_n1000_small --dataset.video_backend=pyav --steps=300 --log_freq=50 --num_workers=6 --wandb.enable=false --save_checkpoint=false"
for spec in "smolvla 7.0 16" "smolvla 7.0 32" "smolvla 5.0 8" "smolvla 5.0 16" "act 5.0 32" "act 5.0 64"; do
  set -- $spec; m=$1; cap=$2; bs=$3; name="p23max_${m}_cap${cap}_bs${bs}"
  args=$([ $m = smolvla ] && echo "$SMOL" || echo "$ACT")
  bash /root/work/queue/bin/gpu_slot.sh 1
  echo "$(date -u +%T) start $name"
  docker exec sprint bash -lc "cd /work/wt-small && $ENV && python env/capped.py --cap-gb $cap --result-json /work/outputs_large/train/$name.json -- lerobot-train $args --batch_size=$bs --output_dir=/work/outputs_large/train/$name > /work/logs/train_$name.log 2>&1; echo EXIT=\$? >> /work/logs/train_$name.log"
  echo "$(date -u +%T) end $name: $(tail -1 /root/work/logs/train_$name.log)"
done
echo EXIT=0
