"""Put a pretrained policy's own normalisation stats into a LeRobot dataset's meta/stats.json.

  python use_pretrained_norm.py DATASET_DIR [--checkpoint allenai/MolmoAct2-SO100_101]
                                [--tag so100_so101_molmoact2]

Why: lerobot-train normalises state/action with the dataset's stats. Fine-tuning
MolmoAct2-SO100_101 with the stats it was pretrained with keeps "normalised value v"
meaning the same joint angle as in pretraining, and gives every data source (sim, real)
the same normalisation. The dataset must already be in the checkpoint's joint frame
(generate.py --export-frame molmoact2). The original stats are kept in
meta/stats_dataset.json, and the script prints how much of the data falls outside q01..q99.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("dataset")
  ap.add_argument("--checkpoint", default="allenai/MolmoAct2-SO100_101")
  ap.add_argument("--tag", default="so100_so101_molmoact2")
  a = ap.parse_args()
  from huggingface_hub import hf_hub_download
  meta = json.loads(Path(hf_hub_download(a.checkpoint, "norm_stats.json")).read_text())
  tag = meta["metadata_by_tag"][a.tag]
  root = Path(a.dataset)
  info = json.loads((root / "meta" / "datagen.json").read_text()) \
    if (root / "meta" / "datagen.json").exists() else {}
  if info and info.get("joint_frame") != "molmoact2":
    raise SystemExit(f"dataset joint_frame is {info.get('joint_frame')!r}, not 'molmoact2'")
  p = root / "meta" / "stats.json"
  backup = root / "meta" / "stats_dataset.json"
  if not backup.exists():
    shutil.copy(p, backup)
  stats = json.loads(backup.read_text())
  report = {}
  for key, src in (("action", "action_stats"), ("observation.state", "state_stats")):
    own, pre = stats[key], tag[src]
    for q in ("min", "max", "mean", "std", "q01", "q99", "q10", "q50", "q90"):
      if q in pre:
        own[q] = [float(x) for x in pre[q]]
    report[key] = {"dataset_q01": [round(x, 1) for x in json.loads(backup.read_text())[key]["q01"]],
                   "dataset_q99": [round(x, 1) for x in json.loads(backup.read_text())[key]["q99"]],
                   "pretrained_q01": [round(x, 1) for x in pre["q01"]],
                   "pretrained_q99": [round(x, 1) for x in pre["q99"]]}
  p.write_text(json.dumps(stats, indent=2))  # numbers only: lerobot tensorises every entry
  (root / "meta" / "stats_source.json").write_text(json.dumps(
    {"normalization": f"{a.checkpoint} norm_stats.json tag {a.tag}",
     "original": "meta/stats_dataset.json", "comparison": report}, indent=2))
  print(json.dumps(report, indent=1))


if __name__ == "__main__":
  main()
