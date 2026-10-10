#!/bin/bash
# P3-A post-processing (HOST): validate the exported dataset, make the _small copy
# (own stats; observation.environment_state dropped = the TARGET cube's pose), validate it,
# tar it. CPU only, nice 19. Log: /root/work/logs/p3a_post.log
set -uo pipefail
SRC=/work/outputs_large/datagen/multitask_A_v1_molmoact2
SMALL=/work/outputs_large/datagen/multitask_A_v1_small
TAR=/root/work/outputs_large/datasets/multitask_A_v1_small.tar.zst
DX="docker exec -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -e OMP_NUM_THREADS=1 -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -w /work/wt-multi/datagen sprint nice -n 19"
echo "[post] $(date -u +%H:%M:%S) validate $SRC"
$DX /opt/venv/bin/python validate_dataset.py $SRC --lerobot --quick-video; rc1=$?
echo "[post] validate src rc=$rc1"
rm -rf /root/work${SMALL#/work}
cp -a /root/work${SRC#/work} /root/work${SMALL#/work}
# prep_small_dataset.py expects meta/stats_dataset.json (the dataset's own stats). This export
# never had the MolmoAct2 pretrained-norm swap, so stats.json IS the dataset's own stats.
[ -f /root/work${SMALL#/work}/meta/stats_dataset.json ] || cp /root/work${SMALL#/work}/meta/stats.json /root/work${SMALL#/work}/meta/stats_dataset.json
$DX /opt/venv/bin/python /work/wt-small/small/prep_small_dataset.py $SMALL; rc2=$?
echo "[post] prep_small rc=$rc2"
$DX /opt/venv/bin/python validate_dataset.py $SMALL --lerobot --quick-video; rc3=$?
echo "[post] validate small rc=$rc3"
$DX /opt/venv/bin/python -c "
import json; i=json.load(open('$SMALL/meta/info.json'))
f=sorted(i['features']); print('features', f)
assert 'observation.environment_state' not in f and not [k for k in json.load(open('$SMALL/meta/stats.json')) if 'environment' in k]
print('environment_state absent: OK; total_episodes', i['total_episodes'], 'total_frames', i['total_frames'], 'tasks', i['total_tasks'])"; rc4=$?
mkdir -p /root/work/outputs_large/datasets
nice -n 19 tar -C /root/work/outputs_large/datagen -cf - multitask_A_v1_small | nice -n 19 zstd -T2 -q -o $TAR -f; rc5=$?
ls -la $TAR
echo "[post] tar rc=$rc5"
rc=$(( rc1 | rc2 | rc3 | rc4 | rc5 ))
echo "EXIT=$rc"
