"""Validate a LeRobot v3.0 dataset written by generate.py.

  python validate_dataset.py runs/sugar/lerobot [--lerobot]

1. Structural checks with whatever Parquet reader is available (pyarrow, else DuckDB):
   info.json fields, data/episode/task tables, index continuity, per-episode frame
   indices and timestamps, episode index ranges, task pointers, stats keys, joint
   ranges, the servo speed cap on the actions, and (if videos) that every mp4 exists
   and decodes to exactly `length` frames of the declared size.
2. With --lerobot (needs `pip install lerobot`, i.e. the droplet): loads the folder with
   lerobot's own LeRobotDataset and compares items against the Parquet rows (and that
   video frames come back as 3xHxW tensors).
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

ARM_VMAX, JAW_VMAX = 0.5, 2.0


def read_parquet(path: Path) -> dict:
  """Return {column: np.ndarray}; list columns become (N, k) arrays."""
  try:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    out = {}
    for name in t.column_names:
      col = t.column(name).to_pylist()
      out[name] = np.array(col, dtype=object if col and isinstance(col[0], str) else None)
    return out, t.schema.metadata or {}
  except ImportError:
    pass
  except Exception as e:  # noqa: BLE001  pyarrow present but blocked (Windows App Control)
    if "DLL" not in str(e) and "Application Control" not in str(e):
      raise
  import duckdb
  con = duckdb.connect()
  rel = con.execute(f"SELECT * FROM read_parquet('{path.as_posix()}')")
  names = [d[0] for d in rel.description]
  rows = rel.fetchall()
  out = {}
  for j, n in enumerate(names):
    vals = [r[j] for r in rows]
    first = vals[0] if vals else None
    if isinstance(first, str) or (isinstance(first, list) and first and isinstance(first[0], str)):
      arr = np.empty(len(vals), dtype=object)
      arr[:] = vals
      out[n] = arr
    else:
      out[n] = np.array(vals)
  meta = {k: v for k, v in con.execute(
    f"SELECT key, value FROM parquet_kv_metadata('{path.as_posix()}')").fetchall()}
  con.close()
  return out, meta


class Checker:
  def __init__(self):
    self.fail, self.ok = [], []

  def check(self, cond, msg):
    (self.ok if cond else self.fail).append(msg)
    return cond


def validate(root: Path, decode_videos: bool = True) -> Checker:
  c = Checker()
  info = json.loads((root / "meta" / "info.json").read_text())
  for k in ("codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks",
            "chunks_size", "fps", "splits", "data_path", "video_path", "features"):
    c.check(k in info, f"info.json has {k}")
  c.check(info["codebase_version"] == "v3.0", "codebase_version v3.0")
  feats = info["features"]
  fps = info["fps"]
  for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
    c.check(k in feats, f"default feature {k}")
  video_keys = [k for k, f in feats.items() if f["dtype"] == "video"]

  # episodes
  ep_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
  c.check(len(ep_files) > 0, "episode metadata parquet present")
  eps = {}
  for f in ep_files:
    cols, _ = read_parquet(f)
    for k, v in cols.items():
      eps.setdefault(k, []).extend(list(v))
  n_ep = len(eps["episode_index"])
  c.check(n_ep == info["total_episodes"], f"episodes table rows {n_ep} == total_episodes")
  for k in ("episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index", "meta/episodes/chunk_index",
            "meta/episodes/file_index"):
    c.check(k in eps, f"episodes column {k}")
  for vk in video_keys:
    for s in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
      c.check(f"videos/{vk}/{s}" in eps, f"episodes column videos/{vk}/{s}")

  # tasks
  tcols, tmeta = read_parquet(root / "meta" / "tasks.parquet")
  tasks = list(tcols.get("task", tcols.get("__index_level_0__", [])))
  c.check(len(tasks) == info["total_tasks"], f"{len(tasks)} tasks == total_tasks")
  pm = tmeta.get(b"pandas") or tmeta.get("pandas")
  if pm is not None:
    pm = json.loads(pm)
    c.check(pm.get("index_columns") and pm["index_columns"][0] in ("task", "__index_level_0__"),
            "tasks.parquet pandas metadata makes the task string the index")
  else:
    c.check(False, "tasks.parquet has pandas metadata")

  # data
  data_files = sorted((root / "data").glob("*/*.parquet"))
  data = {}
  for f in data_files:
    cols, _ = read_parquet(f)
    for k, v in cols.items():
      data.setdefault(k, []).append(np.asarray(v))
  data = {k: np.concatenate(v) for k, v in data.items()}
  N = len(data["index"])
  c.check(N == info["total_frames"], f"data rows {N} == total_frames")
  non_video = [k for k, f in feats.items() if f["dtype"] != "video"]
  c.check(sorted(data) == sorted(non_video), f"data columns match non-video features {sorted(data)}")
  for k in non_video:
    shp = tuple(feats[k]["shape"])
    arr = data[k]
    if shp == (1,):
      c.check(arr.ndim == 1, f"{k} is scalar per frame")
    else:
      c.check(arr.shape[1:] == shp, f"{k} shape {arr.shape[1:]} == {shp}")
  c.check(np.array_equal(data["index"], np.arange(N)), "index is 0..N-1")
  lengths = np.array(eps["length"], int)
  c.check(int(lengths.sum()) == N, "sum of episode lengths == total_frames")
  start = 0
  bad_ep = 0
  for e in range(n_ep):
    L = lengths[e]
    sl = slice(start, start + L)
    good = (eps["dataset_from_index"][e] == start and eps["dataset_to_index"][e] == start + L
            and np.all(data["episode_index"][sl] == eps["episode_index"][e])
            and np.array_equal(data["frame_index"][sl], np.arange(L))
            and np.allclose(data["timestamp"][sl], np.arange(L) / fps, atol=1e-4)
            and np.all((data["task_index"][sl] >= 0) & (data["task_index"][sl] < len(tasks))))
    bad_ep += not good
    start += L
  c.check(bad_ep == 0, f"per-episode index/frame/timestamp/task consistency ({bad_ep} bad)")

  # physical sanity of actions
  act = data["action"].astype(float)
  ep_id = data["episode_index"]
  same = ep_id[1:] == ep_id[:-1]
  dv = np.abs(np.diff(act, axis=0))[same] * fps
  c.check(dv[:, :5].max() <= ARM_VMAX + 1e-3, f"arm action speed max {dv[:, :5].max():.3f} <= 0.5 rad/s")
  c.check(dv[:, 5].max() <= JAW_VMAX + 1e-3, f"jaw action speed max {dv[:, 5].max():.3f} <= 2.0 rad/s")
  c.check(np.all(np.isfinite(data["observation.state"].astype(float))), "states finite")

  # Task success re-derived from the recorded object pose alone (independent of the
  # generator's verifier): starts on the table, rises >= 2 cm, ends resting on the
  # floor of that episode's bin, inside its walls.
  side = root / "meta" / "datagen_episodes.jsonl"
  meta = [json.loads(x) for x in side.read_text().splitlines() if x.strip()]     if side.exists() else []
  if meta and all("variation" in mm for mm in meta):
    from r2s.paths import mj_scene
    c.check(len(meta) == n_ep, "datagen_episodes.jsonl has one row per episode")
    env = data["observation.environment_state"].astype(float)
    bad, start = [], 0
    for e in range(min(n_ep, len(meta))):
      L = lengths[e]
      p = env[start:start + L, :3]
      start += L
      v = meta[e]["variation"]
      mesh = (v.get("extras") or {}).get("mesh")  # object3d hook: mesh objects log their size
      half = v["obj_size"] / 2 if not mesh else mesh["half_extents_m"][2]
      bx, by = v["bin_xy"]
      lim = mj_scene.BIN_INNER_HALF - 0.005
      ok = (abs(p[0, 2] - half) < 0.003 and p[:, 2].max() - p[0, 2] >= 0.02
            and abs(p[-1, 0] - bx) <= lim and abs(p[-1, 1] - by) <= lim
            and abs(p[-1, 2] - (mj_scene.bin_floor_top() + half)) < 0.004)
      if not ok:
        bad.append(e)
    c.check(not bad, f"every episode's object pose shows pick -> lift -> rest in its bin ({bad[:5]})")

  stats = json.loads((root / "meta" / "stats.json").read_text())
  for k in feats:
    c.check(k in stats and all(s in stats[k] for s in ("min", "max", "mean", "std", "count")),
            f"stats.json has {k}")

  if video_keys:
    import av
    tmpl = info["video_path"]
    bad = []
    for e in range(n_ep):
      for vk in video_keys:
        p = root / tmpl.format(video_key=vk, chunk_index=int(eps[f"videos/{vk}/chunk_index"][e]),
                               file_index=int(eps[f"videos/{vk}/file_index"][e]))
        if not p.exists():
          bad.append(f"missing {p}")
          continue
        if not decode_videos and e > 2:
          continue
        with av.open(str(p)) as cont:
          st = cont.streams.video[0]
          nfr = sum(1 for _ in cont.decode(st))
          h, w = feats[vk]["shape"][:2]
          if nfr != lengths[e] or st.codec_context.width != w or st.codec_context.height != h:
            bad.append(f"{p.name}: {nfr} frames (want {lengths[e]}), "
                       f"{st.codec_context.width}x{st.codec_context.height}")
    c.check(not bad, f"videos decode to `length` frames of the declared size ({bad[:3]})")
  return c


def validate_with_lerobot(root: Path) -> Checker:
  c = Checker()
  try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
  except ImportError:
    try:
      from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # older layout
    except ImportError as e:
      c.check(False, f"lerobot importable ({e})")
      return c
  info = json.loads((root / "meta" / "info.json").read_text())
  ds = LeRobotDataset(repo_id="local/so101_datagen", root=root)
  c.check(len(ds) == info["total_frames"], f"LeRobotDataset len {len(ds)} == total_frames")
  c.check(ds.num_episodes == info["total_episodes"], "LeRobotDataset num_episodes")
  data_files = sorted((root / "data").glob("*/*.parquet"))
  cols, _ = read_parquet(data_files[0])
  for i in sorted({0, 1, len(cols["index"]) // 2, len(cols["index"]) - 1}):
    item = ds[i]
    c.check(np.allclose(item["action"].numpy(), np.asarray(cols["action"][i], float), atol=1e-6),
            f"item {i} action matches parquet")
    c.check(isinstance(item.get("task"), str), f"item {i} has task string {item.get('task')!r}")
    for k, f in info["features"].items():
      if f["dtype"] == "video":
        h, w = f["shape"][:2]
        c.check(tuple(item[k].shape) == (3, h, w), f"item {i} {k} shape {tuple(item[k].shape)}")
  return c


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("root")
  ap.add_argument("--lerobot", action="store_true", help="also load with lerobot.LeRobotDataset")
  ap.add_argument("--quick-video", action="store_true", help="decode only the first 3 episodes")
  args = ap.parse_args()
  root = Path(args.root)
  c = validate(root, decode_videos=not args.quick_video)
  results = {"structural": {"passed": len(c.ok), "failed": c.fail}}
  if args.lerobot:
    c2 = validate_with_lerobot(root)
    results["lerobot"] = {"passed": len(c2.ok), "failed": c2.fail}
    c.fail += c2.fail
  print(json.dumps(results, indent=1))
  (root / "meta" / "datagen_validation.json").write_text(json.dumps(results, indent=1))
  sys.exit(1 if c.fail else 0)


if __name__ == "__main__":
  main()
