"""Make the small-model (SmolVLA / ACT) copy of a MolmoAct2-frame training dataset.

  python small/prep_small_dataset.py /work/outputs_large/datagen/sugar_v10_n1000_small

Run on a `cp -a` copy (never on the original). Two edits, both required by OVERNIGHT_BRIEF P2.0:
1. Normalisation: meta/stats.json holds MolmoAct2's PRETRAINED stats (use_pretrained_norm.py);
   small models are trained from their own data, so stats.json <- stats_dataset.json.
2. observation.environment_state (cube xyz + quaternion) is removed from info.json, stats.json,
   every data parquet and the per-episode stats columns. LeRobot ACT uses that feature as an input
   whenever it exists, i.e. it would see the answer no other model sees, and the eval harness
   gives no cube pose. The original dataset keeps it (the verifier's label).
"""
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

KEY = "observation.environment_state"


def main(root: Path) -> None:
  meta = root / "meta"
  stats = json.loads((meta / "stats_dataset.json").read_text())
  stats.pop(KEY, None)
  (meta / "stats.json").write_text(json.dumps(stats, indent=2))
  for f in ("stats_source.json",):  # described the MolmoAct2 pretrained-norm swap
    if (meta / f).exists():
      (meta / f).unlink()
  info = json.loads((meta / "info.json").read_text())
  info["features"].pop(KEY, None)
  (meta / "info.json").write_text(json.dumps(info, indent=2))
  n_cols = 0
  for p in sorted((root / "data").glob("*/*.parquet")) + sorted((meta / "episodes").glob("*/*.parquet")):
    t = pq.read_table(p)
    drop = [c for c in t.column_names if c == KEY or c.startswith(f"stats/{KEY}/")]
    if drop:
      tmp = p.with_suffix(".tmp.parquet")
      pq.write_table(t.drop(drop), tmp)
      shutil.move(tmp, p)
      n_cols += len(drop)
  left = [k for k in json.loads((meta / "stats.json").read_text()) if "environment" in k]
  assert not left and KEY not in json.loads((meta / "info.json").read_text())["features"]
  print(json.dumps({"dropped_columns": n_cols, "stats_keys": sorted(stats),
                    "features": sorted(info["features"])}, indent=1))


if __name__ == "__main__":
  main(Path(sys.argv[1]))
