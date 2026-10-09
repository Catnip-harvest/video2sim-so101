#!/bin/bash
# Generate the JEPA-judge negative pools (each run also yields verifier-passed positives).
# All runs use base seed 1, like the positives in sugar_v10_n1000, so attempt i is the SAME
# scene (layout, look, camera) in every condition; only the physics/expert differs.
# Run inside the `sprint` container. 6 workers, niced, front+wrist, 10 Hz, fast render.
set -u
cd /work/wt-jepa/datagen
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1 OMP_NUM_THREADS=1
SPEC=/work/wt-jepa/demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json
OUT=/work/outputs_large/jepa/gen
COMMON="--spec $SPEC --seed 1 --render --render-quality fast --variety --record-hz 10 --keep-rejects --no-dataset --n 100000 --workers 6"
run() { name=$1; shift; echo "=== $name $(date)"; nice -n 19 python -I generate.py $COMMON --out $OUT/$name "$@" || return $?; }
run no_close --inject no_close --max-attempts 60 &&
run early_release --inject early_release --max-attempts 60 &&
run zoh --servo-hold zoh --max-attempts 160 &&
run widen --widen --max-attempts 200
code=$?
echo "EXIT=$code"
