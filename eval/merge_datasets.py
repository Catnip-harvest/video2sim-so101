"""Merge LeRobot v3 datasets written by datagen/r2s/lerobot_io.write_dataset (one mp4 per
episode per camera) into one dataset of the same format, e.g. SFT set + kept rollouts.

  python eval/merge_datasets.py --src SFT_DATASET KEPT_DATASET --out MERGED [--link]

lerobot 0.6.1's DatasetConfig takes ONE repo_id, so round 2 trains on a merged dataset.
Frames, per-episode stats and the datagen sidecar rows are copied (sidecar rows get
source_dataset / source_episode_index); videos are hard-linked (--link, default) or copied,
never re-encoded. Global image stats are combined exactly from each source's
meta/stats.json (count-weighted mean / second moment, min, max), so no video is decoded.
All sources must share fps, features and joint frame. Then datagen/use_pretrained_norm.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))

import numpy as np  # noqa: E402


def _read_source(root: Path):
  import pyarrow.parquet as pq
  info = json.loads((root / "meta" / "info.json").read_text())
  stats = json.loads((root / "meta" / "stats.json").read_text())
  dg = json.loads((root / "meta" / "datagen.json").read_text())
  side = [json.loads(x) for x in (root / "meta" / "datagen_episodes.jsonl").read_text().splitlines() if x.strip()]
  side = {r["episode_index"]: r for r in side}
  rows = []
  for f in sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet"))):
    rows += pq.read_table(f).to_pylist()
  rows.sort(key=lambda r: r["episode_index"])
  cols = {}
  for f in sorted(glob.glob(str(root / "data" / "*" / "*.parquet"))):
    t = pq.read_table(f, columns=["episode_index", "frame_index", "action", "observation.state",
                                  "observation.environment_state", "task_index"]).to_pydict()
    for k, v in t.items():
      cols.setdefault(k, []).extend(v)
  data = {k: np.array(v) for k, v in cols.items()}
  import pandas as pd
  tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
  task_of = {int(i): t for t, i in zip(tasks.index, tasks["task_index"])}
  return info, stats, dg, side, rows, data, task_of


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--src", nargs="+", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--copy", action="store_true", help="copy videos instead of hard-linking")
  ap.add_argument("--no-pretrained-norm", action="store_true")
  a = ap.parse_args()
  from r2s.lerobot_io import VIDEO_PATH, EpisodeData, write_dataset

  t0 = time.perf_counter()
  out = Path(a.out)
  stage = out.parent / (out.name + "_staging")
  stage.mkdir(parents=True, exist_ok=True)
  eps, first_info, frames_by_src = [], None, []
  for si, src in enumerate(a.src):
    root = Path(src)
    info, stats, dg, side, rows, data, task_of = _read_source(root)
    if first_info is None:
      first_info = info
    else:
      for k in ("fps", "features", "robot_type"):
        if info[k] != first_info[k]:
          sys.exit(f"{src}: {k} differs from {a.src[0]}")
    if dg.get("joint_frame") != "molmoact2":
      sys.exit(f"{src}: joint_frame {dg.get('joint_frame')!r}")
    vkeys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    order = np.argsort(data["episode_index"] * 10_000_000 + data["frame_index"], kind="stable")
    for k in data:
      data[k] = data[k][order]
    bounds = np.searchsorted(data["episode_index"], [r["episode_index"] for r in rows] + [10**12])
    for j, r in enumerate(rows):
      e = r["episode_index"]
      sl = slice(bounds[j], bounds[j + 1])
      assert sl.stop - sl.start == r["length"], (src, e)
      vids = {}
      for k in vkeys:
        p = root / VIDEO_PATH.format(video_key=k, chunk_index=r[f"videos/{k}/chunk_index"],
                                     file_index=r[f"videos/{k}/file_index"])
        dst = stage / f"s{si}_e{e:06d}.{k.split('.')[-1]}.mp4"
        if dst.exists():
          dst.unlink()
        if a.copy:
          shutil.copy2(p, dst)
        else:
          os.link(p, dst)
        vids[k] = dst
      # Exact global image stats: the whole source's totals ride on its first episode.
      img = {}
      for k in vkeys:
        st = stats[k]
        n = float(st["count"][0])
        mean = np.asarray(st["mean"], float).reshape(3)
        std = np.asarray(st["std"], float).reshape(3)
        if j == 0:
          img[k] = {"sum": mean * n, "sumsq": (std ** 2 + mean ** 2) * n,
                    "min": np.asarray(st["min"], float).reshape(3),
                    "max": np.asarray(st["max"], float).reshape(3), "n": n}
        else:
          img[k] = {"sum": np.zeros(3), "sumsq": np.zeros(3), "min": np.ones(3), "max": np.zeros(3), "n": 0}
      meta = dict(side.get(e, {}))
      meta.pop("episode_index", None)
      meta.update({"source_dataset": str(root), "source_episode_index": int(e)})
      tasks = {task_of[int(t)] for t in data["task_index"][sl]}
      assert len(tasks) == 1
      eps.append(EpisodeData(
        state=data["observation.state"][sl].astype(np.float32),
        action=data["action"][sl].astype(np.float32),
        env_state=data["observation.environment_state"][sl].astype(np.float32),
        task=tasks.pop(), videos=vids, image_stats=img, meta=meta))
    frames_by_src.append({"src": str(root), "episodes": len(rows), "frames": int(len(data["action"]))})
  v0 = [k for k, f in first_info["features"].items() if f["dtype"] == "video"]
  f0 = first_info["features"][v0[0]]
  extra = {"generator": "eval/merge_datasets.py", "sources": frames_by_src, "joint_frame": "molmoact2",
           "units": "see source datasets (MolmoAct2 frame)", "conversion": "calib/so101_units.py"}
  info = write_dataset(out, eps, fps=first_info["fps"], video_keys=v0,
                       video_size=(f0["shape"][1], f0["shape"][0]), video_codec=f0["info"]["video.codec"],
                       robot_type=first_info["robot_type"], extra_meta=extra, move_videos=True)
  try:
    stage.rmdir()
  except OSError:
    pass
  if not a.no_pretrained_norm:
    subprocess.run([sys.executable, str(ROOT / "datagen" / "use_pretrained_norm.py"), str(out)],
                   check=True, stdout=subprocess.DEVNULL)
  print(json.dumps({"merged": str(out), "episodes": info["total_episodes"], "frames": info["total_frames"],
                    "sources": frames_by_src, "wall_s": round(time.perf_counter() - t0, 1)}, indent=1))


if __name__ == "__main__":
  main()
