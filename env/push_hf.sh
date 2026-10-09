#!/usr/bin/env bash
# Upload a folder to one of the owner's PRIVATE Hugging Face repos. Run inside `sprint`.
#   bash env/push_hf.sh dataset VietHwang/so101-sim-sugar-10hz /work/outputs_large/datagen/x/lerobot [path_in_repo]
# REPLACE="checkpoints/*": delete remote files matching the glob (except the ones uploaded) in
# the same commit, then squash the repo history so the old blobs stop counting against the
# account's private storage limit (hit on 9 Oct with 11 GB checkpoints).
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
if [ -n "${REPLACE:-}" ]; then
  # (hf upload --delete globs are relative to IN_REPO, so delete siblings explicitly)
  python - "$TYPE" "$REPO" "$REPLACE" "$IN_REPO" <<'PY'
import fnmatch, sys
from huggingface_hub import HfApi, CommitOperationDelete
t, r, pat, keep = sys.argv[1:5]
api = HfApi()
files = api.list_repo_files(r, repo_type=t)
dels = [f for f in files if fnmatch.fnmatch(f, pat) and not f.startswith(keep.rstrip("/") + "/")]
if dels:
  api.create_commit(r, repo_type=t, operations=[CommitOperationDelete(f) for f in dels],
                    commit_message=f"remove {len(dels)} old files ({pat}), keep {keep}")
api.super_squash_history(r, repo_type=t)
print(f"[push_hf] removed {len(dels)} old files matching {pat}; history squashed")
PY
fi
echo "$(date -u +%FT%TZ) PUSHED ${LOCAL#/work/} -> $TYPE $REPO" | tee -a /work/logs/hf_push.log
