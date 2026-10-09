#!/usr/bin/env bash
# Full rendered run of the real-to-sim data engine on a Linux box (written for a
# 20 vCPU droplet; CPU MuJoCo + EGL, falling back to OSMesa).
#
# Needs the sprint/manip tree (task/, third_party/menagerie_so101/, demo/outputs/,
# demo/test_assets/, datagen/). Run from anywhere:
#   bash manip/datagen/run_droplet.sh                  # both specs, N=1000 each
#   N=200 WORKERS=19 SPECS=sugar bash manip/datagen/run_droplet.sh
# System GL libraries (root): apt-get install -y libegl1 libgl1 libosmesa6
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # manip/datagen
MANIP="$(dirname "$HERE")"
N="${N:-1000}"
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
SIZE="${SIZE:-256x256}"
SEED="${SEED:-0}"
OUT="${OUT:-$HERE/runs}"
SPECS="${SPECS:-sugar synthetic}"
PY="${PY:-python3}"

# --- environment ---------------------------------------------------------------------
if [ ! -d "$HERE/.venv" ]; then "$PY" -m venv "$HERE/.venv"; fi
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q "mujoco==3.11.0" "numpy>=2,<3" av pyarrow pandas duckdb matplotlib pillow pytest
if [ "${INSTALL_LEROBOT:-1}" = "1" ]; then
  pip install -q lerobot || echo "[run_droplet] lerobot install failed; --lerobot validation will be skipped"
fi

gl_ok() { python -c "import sys; sys.path.insert(0, '$HERE'); from r2s.render import gl_available; ok, m = gl_available(); print('[gl]', ok, m); sys.exit(0 if ok else 1)"; }
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
if ! gl_ok; then
  echo "[run_droplet] EGL failed; trying OSMesa (CPU, slower)"
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  gl_ok || { echo "[run_droplet] no OpenGL backend; install libegl1 / libosmesa6"; exit 1; }
fi
CODEC=h264
python -c "import av; av.codec.Codec('libx264', 'w')" 2>/dev/null || CODEC=av1
echo "[run_droplet] MUJOCO_GL=$MUJOCO_GL codec=$CODEC N=$N workers=$WORKERS size=$SIZE"

# --- gates -----------------------------------------------------------------------------
python -m pytest -q -p no:cacheprovider "$MANIP/task/tests/test_grasp_physics.py"
python -m pytest -q -p no:cacheprovider "$HERE/tests/test_datagen.py"

# --- generate, validate, make the portfolio outputs -----------------------------------
for name in $SPECS; do
  case "$name" in
    sugar)     spec="$MANIP/demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json"
               clip="$MANIP/demo/test_assets/pexels_7488188_sugar_cube_0-3s.mp4" ;;
    synthetic) spec="$MANIP/demo/outputs/synthetic/so101_task_spec.json"
               clip="$MANIP/demo/test_assets/synthetic_pick_place.mp4" ;;
    *) echo "unknown spec $name"; exit 1 ;;
  esac
  run="$OUT/${name}_n${N}"
  python "$HERE/generate.py" --spec "$spec" --n "$N" --workers "$WORKERS" --render \
    --size "$SIZE" --codec "$CODEC" --seed "$SEED" --out "$run" 2>&1 | tee -a "$run.log"
  if python -c "import lerobot" 2>/dev/null; then
    python "$HERE/validate_dataset.py" "$run/lerobot" --lerobot
  else
    python "$HERE/validate_dataset.py" "$run/lerobot"
  fi
  python "$HERE/plot_trajectories.py" --dataset "$run/lerobot" --spec "$spec" --out "$run/trajectories.png"
  python "$HERE/contact_sheet.py" --dataset "$run/lerobot" --episodes 6 \
    --out "$run/contact_sheet.png" --video "$run/montage.mp4"
  python "$HERE/contact_sheet.py" --dataset "$run/lerobot" --episodes 6 --wrist \
    --out "$run/contact_sheet_wrist.png"
  for ep in 0 1 2; do
    python "$HERE/side_by_side.py" --dataset "$run/lerobot" --episode "$ep" --spec "$spec" \
      --human-video "$clip" --out "$run/side_by_side_ep${ep}.mp4"
  done
  cat "$run/acceptance.md"
done
echo "[run_droplet] done: $OUT"
