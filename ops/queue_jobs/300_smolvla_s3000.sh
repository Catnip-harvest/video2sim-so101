#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# P2.4 smolvla: seed 3000 ids 0-99 -
docker exec sprint bash -lc 'cd /work/wt-p24 && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/final_models/smolvla_single_bs64_020000/pretrained_model --seed 3000 --n 100 --videos 4 --workers 1 --device cuda  --out /work/outputs_large/eval/smolvla_s3000 > /work/logs/eval_smolvla_s3000.log 2>&1; echo EXIT=$? >> /work/logs/eval_smolvla_s3000.log'
