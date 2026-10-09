#!/usr/bin/env bash
# Run INSIDE the container:  docker exec sprint bash /work/video2sim-so101/env/setup_container.sh
# Installs GL libs + Python deps without replacing the ROCm PyTorch, then prefetches models/data.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libosmesa6 libgles2 libglvnd0 ffmpeg git >/dev/null
export PIP_CONSTRAINT="$REPO/env/constraints.txt"
pip install -q -r "$REPO/env/requirements-droplet.txt"
pip install -q -e "$REPO/task"
python -m venv --system-site-packages "$REPO/datagen/.venv"   # run_droplet.sh reuses this; inherits ROCm torch
python - <<'PY'
import torch, mujoco
from importlib.metadata import version
print("torch", torch.__version__, "gpu", torch.cuda.is_available(), "| lerobot", version("lerobot"), "| mujoco", mujoco.__version__)
from huggingface_hub import snapshot_download
for rid, kind in [("MinhDuk/final_dataset_annotated", "dataset"), ("lerobot/pi05_base", "model")]:
    print("prefetch", rid, snapshot_download(rid, repo_type=kind))
PY
echo SETUP_DONE
