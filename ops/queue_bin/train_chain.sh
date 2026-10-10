#!/usr/bin/env bash
# Training lane after SmolVLA (one training at a time; the eval queue is the only other GPU process).
# 1. P2.5 ACT-wide: P2.2 recipe on the wide dataset.
# 2. P2.3 capped training: 1000 steps each, SmolVLA / ACT at 7.0 GiB bs8 and 5.0 GiB bs4, each with an
#    "uncapped" reference at the same batch (lerobot-train has no grad-accumulation flag). The reference runs
#    use --cap-gb 180 (effectively no cap) so the same wrapper reports max_memory_allocated/reserved.
# Run on the HOST: nohup bash /root/work/queue/bin/train_chain.sh > /root/work/logs/train_chain.log 2>&1 &
set -uo pipefail
L=/root/work/logs
echo "$(date -u +%T) waiting for SmolVLA training to exit"
until grep -q "^EXIT=" $L/train_smolvla_single_bs64.log 2>/dev/null; do sleep 60; done
echo "$(date -u +%T) SmolVLA exited: $(tail -1 $L/train_smolvla_single_bs64.log)"
ENV='export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1'
SMALL=/work/outputs_large/datagen/sugar_v10_n1000_small
WIDE=/work/outputs_large/datagen/sugar_v10_n1000_wide_small

run() {  # name, command (inside container, cwd /work/wt-small)
  local name="$1"; shift
  echo "$(date -u +%T) start $name"
  bash /root/work/queue/bin/gpu_slot.sh 1
  docker exec sprint bash -lc "cd /work/wt-small && $ENV && $* > /work/logs/$name.log 2>&1; echo EXIT=\$? >> /work/logs/$name.log"
  echo "$(date -u +%T) end $name: $(tail -1 $L/$name.log)"
}

# 1. P2.5 ACT-wide (same recipe as P2.2 act_single_bs32)
run train_act_wide_bs32 "lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.n_action_steps=50 --dataset.root=$WIDE --dataset.repo_id=local/sugar_v10_n1000_wide_small --dataset.video_backend=pyav --batch_size=32 --steps=30000 --save_freq=10000 --log_freq=200 --num_workers=6 --wandb.enable=false --output_dir=/work/outputs_large/train/act_wide_bs32"

# 2. P2.3 capped training (1000 steps, no checkpoints)
SMOL="--policy.path=lerobot/smolvla_base --dataset.root=$SMALL --dataset.repo_id=local/sugar_v10_n1000_small --dataset.video_backend=pyav --rename_map='{\"observation.images.cam_head\":\"observation.images.camera1\",\"observation.images.cam_right\":\"observation.images.camera2\"}' --policy.empty_cameras=1 --policy.push_to_hub=false --policy.device=cuda --steps=1000 --log_freq=50 --num_workers=6 --wandb.enable=false --save_checkpoint=false"
ACT="--policy.type=act --policy.device=cuda --policy.push_to_hub=false --policy.n_action_steps=50 --dataset.root=$SMALL --dataset.repo_id=local/sugar_v10_n1000_small --dataset.video_backend=pyav --steps=1000 --log_freq=50 --num_workers=6 --wandb.enable=false --save_checkpoint=false"
for spec in "smolvla 7.0 8" "smolvla 5.0 4" "act 7.0 8" "act 5.0 4"; do
  set -- $spec; m=$1; cap=$2; bs=$3
  args=$([ $m = smolvla ] && echo "$SMOL" || echo "$ACT")
  for c in $cap 180; do
    tag=$([ $c = 180 ] && echo "ref" || echo "cap${cap}")
    name="train_p23_${m}_${tag}_bs${bs}"
    rm_out="/work/outputs_large/train/p23_${m}_${tag}_bs${bs}"
    run $name "python env/capped.py --cap-gb $c --result-json /work/outputs_large/train/p23_${m}_${tag}_bs${bs}.json -- lerobot-train $args --batch_size=$bs --output_dir=$rm_out"
  done
done
echo "$(date -u +%T) chain done"
echo EXIT=0
