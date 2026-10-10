#!/usr/bin/env bash
bash /root/work/queue/bin/gpu_slot.sh 1 || exit 1
# Re-run of the SmolVLA 3-episode smoke with the needs_obs fix (commit 7cdd01c, wt-p24): same seeds as
# 03_smolvla_smoke_eval (seed 9000 ids 0-2). Checks speed-up and that outcomes match the first smoke.
docker exec sprint bash -lc 'cd /work/wt-p24 && export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa HF_HOME=/work/hf_home HF_HUB_OFFLINE=1 && nice -n 5 env OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python eval/sim_eval.py --mode policy --checkpoint /work/outputs_large/train/smolvla_smoke/checkpoints/000050/pretrained_model --seed 9000 --n 3 --videos 1 --workers 1 --device cuda --out /work/outputs_large/eval/smolvla_smoke_s9000_v2 > /work/logs/eval_smolvla_smoke_v2.log 2>&1; echo EXIT=$? >> /work/logs/eval_smolvla_smoke_v2.log'
