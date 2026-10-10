#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# PROPOSED (not queued): P1.2 single-axis camera sub-rung 'camera_pitch' (docs/OOD_LADDER.md, branch eval/ood-ladder).
# Move into /root/work/queue/ (renamed to an NN_ that sorts where you want it) to run. ~12-15 min each.
docker exec sprint bash -lc 'cd /work/wt-ood && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/final_models/molmoact2_sim_v1_023400/pretrained_model --seed 4000 --n 50 --workers 1 --videos 4 --ood camera_pitch --device cuda --out /work/outputs_large/eval/molmo_s4000_camera_pitch > /work/logs/eval_molmo_s4000_camera_pitch.log 2>&1; echo EXIT=$? >> /work/logs/eval_molmo_s4000_camera_pitch.log'
