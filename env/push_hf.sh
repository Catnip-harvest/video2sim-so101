#!/usr/bin/env bash
# Upload a folder to one of the owner's PRIVATE Hugging Face repos. Run inside `sprint`.
#   bash env/push_hf.sh dataset VietHwang/so101-sim-sugar-10hz /work/outputs_large/datagen/x/lerobot [path_in_repo]
# Refuses to upload unless the repo exists and is private. On success appends
# "PUSHED <local path> -> <repo>" to /work/logs/hf_push.log (read by /work/status.sh).
set -euo pipefail
TYPE="$1"; REPO="$2"; LOCAL="$3"; IN_REPO="${4:-.}"
export HF_HOME="${HF_HOME:-/work/hf_home}"
hf repos create "$REPO" --repo-type "$TYPE" --private --exist-ok >/dev/null
python - "$TYPE" "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
t, r = sys.argv[1], sys.argv[2]
info = HfApi().repo_info(r, repo_type=t)
if not info.private:
  sys.exit(f"REFUSING: {t} repo {r} is PUBLIC")
print(f"[push_hf] {t} {r} is private")
PY
hf upload "$REPO" "$LOCAL" "$IN_REPO" --repo-type "$TYPE" \
  --commit-message "upload $(basename "$LOCAL") $(date -u +%FT%TZ)" --format quiet
echo "$(date -u +%FT%TZ) PUSHED ${LOCAL#/work/} -> $TYPE $REPO" | tee -a /work/logs/hf_push.log
