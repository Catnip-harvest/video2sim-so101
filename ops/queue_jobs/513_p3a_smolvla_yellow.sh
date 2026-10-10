#!/usr/bin/env bash
until grep -q '^EXIT=0' /root/work/logs/train_smolvla_multi_bs64.log 2>/dev/null; do sleep 60; done  # wait for this model's training
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# P3-A eval: yellow seed 5000 ids 100..119 (scripted-expert demos; named-cube scoring)
docker exec sprint bash -lc 'cd /work/wt-multi && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/train/smolvla_multi_bs64/checkpoints/015000/pretrained_model --multitask yellow --seed 5000 --start 100 --n 20 --videos 4 --workers 1 --device cuda --out /work/outputs_large/eval/513_p3a_smolvla_yellow > /work/logs/eval_513_p3a_smolvla_yellow.log 2>&1; echo EXIT=$? >> /work/logs/eval_513_p3a_smolvla_yellow.log'
