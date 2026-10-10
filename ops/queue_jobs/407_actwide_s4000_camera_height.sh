#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# P2.4 actwide: seed 4000 ids 0-49 camera_height
docker exec sprint bash -lc 'cd /work/wt-p24 && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/final_models/act_wide_bs32_030000/pretrained_model --seed 4000 --n 50 --videos 4 --workers 1 --device cuda --ood camera_height --out /work/outputs_large/eval/actwide_s4000_camera_height > /work/logs/eval_actwide_s4000_camera_height.log 2>&1; echo EXIT=$? >> /work/logs/eval_actwide_s4000_camera_height.log'
