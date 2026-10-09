"""LeRobot dataset writer, codebase version v3.0.

Layout written (paths and field names follow lerobot/datasets/utils.py, v3.0):

  meta/info.json                                   codebase_version "v3.0", fps, features, paths
  meta/stats.json                                  per-feature min/max/mean/std/count/q01/q99
  meta/tasks.parquet                               index = task string, column task_index
  meta/episodes/chunk-000/file-000.parquet         one row per episode (lengths, index ranges,
                                                   data/video file pointers, per-episode stats)
  data/chunk-000/file-000.parquet                  one row per frame
  videos/<key>/chunk-CCC/file-FFF.mp4              one mp4 per episode (from_timestamp = 0)
  meta/datagen_episodes.jsonl                      ours, not LeRobot's: variation parameters and
                                                   verification stats per episode
  meta/datagen.json                                ours: generator config + acceptance table

Parquet backend: pyarrow when importable (the droplet; LeRobot itself needs it), else
DuckDB (this laptop: Windows Application Control blocks pyarrow's DLL). The DuckDB path
writes the same columns and adds pandas index metadata to tasks.parquet so that
``pd.read_parquet`` restores the task string as the index, as LeRobot expects.

Units: joint values are radians (MuJoCo joint angles), not LeRobot's degrees / 0-100
convention for the real SO-101; see README "Using the data".
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import rm

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_FILE_SIZE_MB = 100
VIDEO_FILE_SIZE_MB = 500
ROWS_PER_DATA_FILE = 400_000  # ~100 MB uncompressed for these narrow rows
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

JOINT_NAMES = [f"{j}.pos" for j in rm.ALL_JOINTS]
ENV_NAMES = ["obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz"]


def have_pyarrow() -> bool:
  try:
    import pyarrow  # noqa: F401
    import pyarrow.parquet  # noqa: F401
    return True
  except Exception:
    return False


def build_features(video_keys: list[str], size: tuple[int, int], codec: str, fps: int) -> dict:
  w, h = size
  f = {
    "action": {"dtype": "float32", "shape": [6], "names": JOINT_NAMES},
    "observation.state": {"dtype": "float32", "shape": [6], "names": JOINT_NAMES},
    "observation.environment_state": {"dtype": "float32", "shape": [7], "names": ENV_NAMES},
  }
  for k in video_keys:
    f[k] = {
      "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
      "info": {"video.height": h, "video.width": w, "video.codec": codec,
               "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps,
               "video.channels": 3, "has_audio": False},
    }
  f.update({
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
  })
  return f


def _num_stats(x: np.ndarray) -> dict:
  x = np.asarray(x, np.float64)
  if x.ndim == 1:
    x = x[:, None]
  return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
          "std": x.std(0).tolist(), "count": [int(len(x))],
          "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist()}


def _video_stats(path: Path, every: int = 5) -> dict:
  """Per-channel pixel stats in [0, 1] from every ``every``-th decoded frame."""
  import av
  acc = {"sum": np.zeros(3), "sumsq": np.zeros(3), "min": np.ones(3), "max": np.zeros(3), "n": 0}
  with av.open(str(path)) as c:
    for i, fr in enumerate(c.decode(c.streams.video[0])):
      if i % every:
        continue
      x = fr.to_ndarray(format="rgb24").reshape(-1, 3).astype(np.float64) / 255.0
      acc["sum"] += x.sum(0)
      acc["sumsq"] += (x * x).sum(0)
      acc["min"] = np.minimum(acc["min"], x.min(0))
      acc["max"] = np.maximum(acc["max"], x.max(0))
      acc["n"] += len(x)
  return acc


@dataclass
class EpisodeData:
  state: np.ndarray  # (T, 6)
  action: np.ndarray  # (T, 6)
  env_state: np.ndarray  # (T, 7)
  task: str
  videos: dict  # video_key -> staged mp4 path
  image_stats: dict  # video_key -> {"sum": (3,), "sumsq": (3,), "min", "max", "n"} in [0,1]
  meta: dict  # datagen sidecar row


class _ParquetBackend:
  def __init__(self):
    self.pa = have_pyarrow()
    if not self.pa:
      import duckdb  # noqa: F401

  def write_frames(self, path: Path, cols: dict) -> None:
    """cols: name -> np.ndarray ((N,) or (N, k))."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if self.pa:
      import pyarrow as pa
      import pyarrow.parquet as pq
      arrays, names = [], []
      for name, arr in cols.items():
        arr = np.asarray(arr)
        if arr.ndim == 2:
          flat = pa.array(arr.reshape(-1), type=pa.from_numpy_dtype(arr.dtype))
          arrays.append(pa.FixedSizeListArray.from_arrays(flat, arr.shape[1]))
        else:
          arrays.append(pa.array(arr))
        names.append(name)
      pq.write_table(pa.Table.from_arrays(arrays, names=names), path)
      return
    import duckdb
    import pandas as pd
    flat, select = {}, []
    for name, arr in cols.items():
      arr = np.asarray(arr)
      safe = name.replace(".", "_")
      if arr.ndim == 2:
        for j in range(arr.shape[1]):
          flat[f"{safe}__{j}"] = arr[:, j]
        sql_t = "FLOAT" if arr.dtype == np.float32 else "DOUBLE"
        items = ", ".join(f"{safe}__{j}" for j in range(arr.shape[1]))
        select.append(f'[{items}]::{sql_t}[{arr.shape[1]}] AS "{name}"')
      else:
        flat[safe] = arr
        select.append(f'"{safe}" AS "{name}"')
    con = duckdb.connect()
    con.register("t", pd.DataFrame(flat))
    con.execute(f"COPY (SELECT {', '.join(select)} FROM t) TO '{path.as_posix()}' "
                "(FORMAT PARQUET)")
    con.close()

  def write_table(self, path: Path, rows: list[dict]) -> None:
    """Episode metadata: scalars, strings, lists of strings or of floats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    if self.pa:
      import pyarrow as pa
      import pyarrow.parquet as pq
      pq.write_table(pa.Table.from_pylist(rows), path)
      return
    import duckdb
    con = duckdb.connect()
    defs = []
    for k in keys:
      v = rows[0][k]
      if isinstance(v, bool):
        t = "BOOLEAN"
      elif isinstance(v, int):
        t = "BIGINT"
      elif isinstance(v, float):
        t = "DOUBLE"
      elif isinstance(v, str):
        t = "VARCHAR"
      elif isinstance(v, list) and v and isinstance(v[0], str):
        t = "VARCHAR[]"
      elif isinstance(v, list):
        t = "DOUBLE[]"
      else:
        raise TypeError(f"{k}: {type(v)}")
      defs.append((k, t))
    con.execute("CREATE TABLE e (" + ", ".join(f'"{k}" {t}' for k, t in defs) + ")")
    con.executemany("INSERT INTO e VALUES (" + ", ".join("?" for _ in defs) + ")",
                    [[r[k] for k, _ in defs] for r in rows])
    con.execute(f"COPY e TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()

  def write_tasks(self, path: Path, tasks: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if self.pa:
      import pandas as pd
      pd.DataFrame({"task_index": np.arange(len(tasks), dtype=np.int64)},
                   index=pd.Index(tasks, name="task")).to_parquet(path)
      return
    import duckdb
    meta = {
      "index_columns": ["task"], "column_indexes": [],
      "columns": [
        {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
         "numpy_type": "int64", "metadata": None},
        {"name": "task", "field_name": "task", "pandas_type": "unicode",
         "numpy_type": "object", "metadata": None},
      ],
      "creator": {"library": "pyarrow", "version": "14.0.0"}, "pandas_version": "2.2.0",
    }
    con = duckdb.connect()
    con.execute("CREATE TABLE t (task_index BIGINT, task VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [[i, t] for i, t in enumerate(tasks)])
    meta_s = json.dumps(meta).replace("'", "''")
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET, KV_METADATA {{pandas: '{meta_s}'}})")
    con.close()


def write_dataset(root: Path, episodes: list[EpisodeData], fps: int, video_keys: list[str],
                  video_size: tuple[int, int], video_codec: str, robot_type: str,
                  extra_meta: dict, move_videos: bool = True) -> dict:
  """Write a complete LeRobot v3.0 dataset at ``root`` (must not exist or be empty)."""
  root = Path(root)
  if root.exists() and any(root.iterdir()):
    raise FileExistsError(f"{root} is not empty")
  root.mkdir(parents=True, exist_ok=True)
  be = _ParquetBackend()
  tasks = sorted({e.task for e in episodes})
  task_idx = {t: i for i, t in enumerate(tasks)}
  features = build_features(video_keys, video_size, video_codec, fps)

  ep_rows, all_state, all_action, all_env = [], [], [], []
  img_acc = {k: None for k in video_keys}
  global_i = 0
  file_rows: list[dict] = []
  data_chunk, data_file, rows_in_file = 0, 0, 0
  pending: dict[str, list] = {k: [] for k in
                              ("action", "observation.state", "observation.environment_state",
                               "timestamp", "frame_index", "episode_index", "index", "task_index")}

  def flush():
    nonlocal data_file, data_chunk, rows_in_file, pending
    if rows_in_file == 0:
      return
    cols = {k: np.concatenate(v) for k, v in pending.items()}
    be.write_frames(root / DATA_PATH.format(chunk_index=data_chunk, file_index=data_file), cols)
    pending = {k: [] for k in pending}
    rows_in_file = 0
    data_file += 1
    if data_file >= CHUNKS_SIZE:
      data_chunk, data_file = data_chunk + 1, 0

  for ei, ep in enumerate(episodes):
    T = len(ep.state)
    if rows_in_file and rows_in_file + T > ROWS_PER_DATA_FILE:
      flush()
    fi = np.arange(T, dtype=np.int64)
    pending["action"].append(ep.action.astype(np.float32))
    pending["observation.state"].append(ep.state.astype(np.float32))
    pending["observation.environment_state"].append(ep.env_state.astype(np.float32))
    pending["timestamp"].append((fi / fps).astype(np.float32))
    pending["frame_index"].append(fi)
    pending["episode_index"].append(np.full(T, ei, np.int64))
    pending["index"].append(global_i + fi)
    pending["task_index"].append(np.full(T, task_idx[ep.task], np.int64))
    row = {
      "episode_index": ei, "tasks": [ep.task], "length": T,
      "data/chunk_index": data_chunk, "data/file_index": data_file,
      "dataset_from_index": global_i, "dataset_to_index": global_i + T,
    }
    for k in video_keys:
      vc, vf = ei // CHUNKS_SIZE, ei % CHUNKS_SIZE
      dst = root / VIDEO_PATH.format(video_key=k, chunk_index=vc, file_index=vf)
      dst.parent.mkdir(parents=True, exist_ok=True)
      (shutil.move if move_videos else shutil.copy2)(str(ep.videos[k]), str(dst))
      row.update({f"videos/{k}/chunk_index": vc, f"videos/{k}/file_index": vf,
                  f"videos/{k}/from_timestamp": 0.0, f"videos/{k}/to_timestamp": T / fps})
    row.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
    for name, arr in (("action", ep.action), ("observation.state", ep.state),
                      ("observation.environment_state", ep.env_state)):
      st = _num_stats(arr)
      for s in ("min", "max", "mean", "std"):
        row[f"stats/{name}/{s}"] = [float(x) for x in st[s]]
      row[f"stats/{name}/count"] = [float(T)]
    ep_rows.append(row)
    all_state.append(ep.state)
    all_action.append(ep.action)
    all_env.append(ep.env_state)
    for k in video_keys:
      s = ep.image_stats.get(k) or _video_stats(root / VIDEO_PATH.format(
        video_key=k, chunk_index=ei // CHUNKS_SIZE, file_index=ei % CHUNKS_SIZE))
      acc = img_acc[k]
      if acc is None:
        img_acc[k] = {kk: np.array(vv, np.float64) for kk, vv in s.items()}
      else:
        acc["sum"] += s["sum"]
        acc["sumsq"] += s["sumsq"]
        acc["n"] += s["n"]
        acc["min"] = np.minimum(acc["min"], s["min"])
        acc["max"] = np.maximum(acc["max"], s["max"])
    global_i += T
    rows_in_file += T
  flush()

  be.write_table(root / EPISODES_PATH.format(chunk_index=0, file_index=0), ep_rows)
  be.write_tasks(root / "meta" / "tasks.parquet", tasks)

  stats = {
    "action": _num_stats(np.concatenate(all_action)),
    "observation.state": _num_stats(np.concatenate(all_state)),
    "observation.environment_state": _num_stats(np.concatenate(all_env)),
  }
  ts = np.concatenate([np.arange(len(e.state)) / fps for e in episodes])
  fidx = np.concatenate([np.arange(len(e.state)) for e in episodes])
  stats["timestamp"] = _num_stats(ts)
  stats["frame_index"] = _num_stats(fidx)
  stats["episode_index"] = _num_stats(np.concatenate(
    [np.full(len(e.state), i) for i, e in enumerate(episodes)]))
  stats["index"] = _num_stats(np.arange(global_i))
  stats["task_index"] = _num_stats(np.concatenate(
    [np.full(len(e.state), task_idx[e.task]) for e in episodes]))
  for k, acc in img_acc.items():
    if acc is None:
      continue
    n = acc["n"]
    mean = acc["sum"] / n
    std = np.sqrt(np.maximum(acc["sumsq"] / n - mean**2, 0))
    c = lambda x: np.asarray(x, float).reshape(3, 1, 1).tolist()  # noqa: E731
    stats[k] = {"min": c(acc["min"]), "max": c(acc["max"]), "mean": c(mean), "std": c(std),
                "count": [int(n)]}

  info = {
    "codebase_version": CODEBASE_VERSION, "robot_type": robot_type,
    "total_episodes": len(episodes), "total_frames": int(global_i), "total_tasks": len(tasks),
    "chunks_size": CHUNKS_SIZE, "data_files_size_in_mb": DATA_FILE_SIZE_MB,
    "video_files_size_in_mb": VIDEO_FILE_SIZE_MB, "fps": fps,
    "splits": {"train": f"0:{len(episodes)}"}, "data_path": DATA_PATH,
    "video_path": VIDEO_PATH if video_keys else None, "features": features,
  }
  (root / "meta").mkdir(exist_ok=True)
  (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
  (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
  with open(root / "meta" / "datagen_episodes.jsonl", "w", encoding="utf-8") as fh:
    for i, e in enumerate(episodes):
      fh.write(json.dumps({"episode_index": i, **e.meta}) + "\n")
  (root / "meta" / "datagen.json").write_text(json.dumps(extra_meta, indent=2))
  return info
