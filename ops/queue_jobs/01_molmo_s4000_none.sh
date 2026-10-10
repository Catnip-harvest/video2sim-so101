#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# P1.2 baseline: MolmoAct2 final, seed 4000 ids 0-49, no OOD change (eval/harness code = --ood none),
# same harness flags as P1.1 (defaults). Queue job: blocking docker exec, one policy process.
docker exec sprint bash -lc 'cd /work/wt-eval && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model --seed 4000 --n 50 --videos 4 --workers 1 --device cuda --out /work/outputs_large/eval/molmo_s4000_none > /work/logs/eval_molmo_s4000_none.log 2>&1; echo EXIT=$? >> /work/logs/eval_molmo_s4000_none.log'
