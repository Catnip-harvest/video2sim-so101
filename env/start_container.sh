#!/usr/bin/env bash
# Start the GPU work container on an AMD Developer Cloud MI300X droplet (PyTorch 2.10 / ROCm 7.2 image).
# The repo is mounted at /work/<repo>; Hugging Face cache at /work/hf_home.
set -euo pipefail
WORK="${WORK:-/root/work}"
IMAGE="${IMAGE:-rocm/pytorch:rocm7.2_ubuntu24.04_py3.13_pytorch_release_2.10.0}"
mkdir -p "$WORK/logs"
docker run -d --name sprint --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -v "$WORK":/work -w /work \
  -e HF_HOME=/work/hf_home -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
  "$IMAGE" sleep infinity
docker exec sprint python -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available())"
