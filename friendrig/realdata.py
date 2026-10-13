"""Read the friend's LeRobot v3.0 dataset directly (parquet + AV1 mp4 via PyAV).

Faster and simpler than LeRobotDataset for our access pattern (a few frames from many episodes,
or a whole episode once). Frame k of episode e is the video frame at time
from_timestamp(e, cam) + k / fps in that camera's mp4 file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/work/outputs_large/datasets/friend_raw")
SPLIT = Path("/work/outputs_large/datasets/friend_split.json")
CAMS = ("cam_head", "cam_right")


class FriendData:
  def __init__(self, root: str | Path = ROOT):
    self.root = Path(root)
    self.info = json.loads((self.root / "meta/info.json").read_text())
    self.fps = int(self.info["fps"])
    self.eps = pd.read_parquet(self.root / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    cols = ["observation.state", "action", "episode_index", "frame_index"]
    frames = [pd.read_parquet(p, columns=cols) for p in sorted((self.root / "data").rglob("*.parquet"))]
    d = pd.concat(frames, ignore_index=True)
    self._state = np.stack(d["observation.state"].to_numpy()).astype(np.float64)
    self._action = np.stack(d["action"].to_numpy()).astype(np.float64)
    self._epi = d["episode_index"].to_numpy()
    self._fidx = d["frame_index"].to_numpy()
    self._rows = {}
    order = np.argsort(self._epi * 10**7 + self._fidx, kind="stable")
    e_sorted = self._epi[order]
    starts = np.r_[0, np.flatnonzero(np.diff(e_sorted)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for s, t in zip(starts, ends):
      self._rows[int(e_sorted[s])] = order[s:t]

  # ------------------------------------------------------------------ metadata
  @property
  def episodes(self) -> list[int]:
    return sorted(self._rows)

  def length(self, ep: int) -> int:
    return len(self._rows[ep])

  def task(self, ep: int) -> str:
    t = self.eps.loc[ep, "tasks"]
    return str(t[0]) if len(t) else ""

  def states(self, ep: int) -> np.ndarray:
    return self._state[self._rows[ep]]

  def actions(self, ep: int) -> np.ndarray:
    return self._action[self._rows[ep]]

  def video(self, ep: int, cam: str) -> tuple[Path, float]:
    pre = f"videos/observation.images.{cam}"
    r = self.eps.loc[ep]
    p = self.root / self.info["video_path"].format(video_key=f"observation.images.{cam}",
                                                  chunk_index=int(r[f"{pre}/chunk_index"]),
                                                  file_index=int(r[f"{pre}/file_index"]))
    return p, float(r[f"{pre}/from_timestamp"])

  # ------------------------------------------------------------------ frames
  def frames(self, ep: int, cam: str, idxs) -> list[np.ndarray]:
    """uint8 RGB [480,640,3] frames of episode `ep` at the given frame indices (any order)."""
    import av
    idxs = [int(i) for i in idxs]
    if not idxs:
      return []
    path, t0 = self.video(ep, cam)
    want = sorted(set(idxs))
    got = {}
    with av.open(str(path)) as c:
      st = c.streams.video[0]
      st.thread_type = "AUTO"
      # group targets so that far-apart targets get their own seek (decoding AV1 is the cost)
      groups, cur = [], [want[0]]
      for k in want[1:]:
        if k - cur[-1] > 2 * self.fps:
          groups.append(cur); cur = [k]
        else:
          cur.append(k)
      groups.append(cur)
      for g in groups:
        ts = t0 + g[0] / self.fps
        c.seek(int(max(0.0, ts - 0.5) / st.time_base), stream=st, backward=True, any_frame=False)
        pending = list(g)
        for fr in c.decode(st):
          t = float(fr.pts * st.time_base)
          while pending and t >= t0 + pending[0] / self.fps - 0.5 / self.fps:
            got[pending.pop(0)] = fr.to_ndarray(format="rgb24")
          if not pending:
            break
    return [got[i] for i in idxs]

  def all_frames(self, ep: int, cam: str, step: int = 1) -> np.ndarray:
    return np.stack(self.frames(ep, cam, range(0, self.length(ep), step)))


@lru_cache(1)
def split() -> dict[str, list[int]]:
  return json.loads(SPLIT.read_text())
