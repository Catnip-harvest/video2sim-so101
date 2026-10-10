#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# P2.0 step 6: 3-episode smoke of the ACT 50-step checkpoint on dev seed 9000 (plumbing only; success not reported).
docker exec sprint bash -lc 'cd /work/wt-small && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/train/act_smoke/checkpoints/000050/pretrained_model --seed 9000 --n 3 --videos 1 --workers 1 --device cuda --out /work/outputs_large/eval/act_smoke_s9000 > /work/logs/eval_act_smoke.log 2>&1; echo EXIT=$? >> /work/logs/eval_act_smoke.log'
