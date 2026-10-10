#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# P1.2 OOD rung 'exposure' (docs/OOD_LADDER.md on branch eval/ood-ladder): MolmoAct2 final, seed 4000 ids 0-49,
# same scenes as 01_molmo_s4000_none.sh (--ood none is byte-identical to eval/harness), harness flags at P1.1 defaults.
# Queue job: blocking docker exec, one policy process.
docker exec sprint bash -lc 'cd /work/wt-ood && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model --seed 4000 --n 50 --workers 1 --videos 4 --ood exposure --device cuda --out /work/outputs_large/eval/molmo_s4000_exposure > /work/logs/eval_molmo_s4000_exposure.log 2>&1; echo EXIT=$? >> /work/logs/eval_molmo_s4000_exposure.log'
