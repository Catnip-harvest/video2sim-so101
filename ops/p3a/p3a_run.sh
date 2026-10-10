#!/bin/bash
# usage: p3a_run.sh <python args...>  (runs in container sprint, CPU only, nice 19, cwd /work/wt-multi/datagen)
exec docker exec -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e LP_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -w /work/wt-multi/datagen sprint nice -n 19 /work/manip/datagen/.venv/bin/python "$@"
