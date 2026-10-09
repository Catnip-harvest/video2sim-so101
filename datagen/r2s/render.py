"""Render an accepted episode's recorded states to mp4 (one file per camera).

Rendering is playback only: each frame sets the recorded full qpos on a separate MjData
and calls mj_forward to place bodies for the camera. It never feeds back into the
physics, which already ran (and was verified) without rendering. This is why only
accepted episodes are rendered.

Needs an OpenGL backend: on Linux set MUJOCO_GL=egl (GPU or headless EGL) or
MUJOCO_GL=osmesa (CPU). On the Windows laptop no backend works (gladLoadGL fails), so
generate.py runs with --no-render there.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .scene import FPS, build_episode_scene
from .variations import Variation

CAMERAS = {"observation.images.front": "front", "observation.images.wrist": "wrist_cam"}
CODECS = {"h264": "libx264", "av1": "libsvtav1"}
# OSMesa (CPU) cost per image, measured on the droplet with LP_NUM_THREADS=1 at 256x256:
# full 3.7/s, shadow 12/s, fast 39/s. Shadows + floor reflection + 4x MSAA dominate.
QUALITIES = ("full", "shadow", "fast")


def _apply_quality(m: mujoco.MjModel, quality: str) -> dict:
  """Set model-level render options; return scene flags to force on every frame."""
  if quality not in QUALITIES:
    raise ValueError(f"render quality {quality!r} not in {QUALITIES}")
  if quality == "full":
    return {}
  m.vis.quality.offsamples = 0
  flags = {mujoco.mjtRndFlag.mjRND_REFLECTION: 0}
  if quality == "shadow":
    m.vis.quality.shadowsize = 1024
  else:
    flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
  return flags


def gl_available() -> tuple[bool, str]:
  try:
    m = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><light pos="0 0 1"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
    r = mujoco.Renderer(m, 32, 32)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    r.render()
    r.close()
    return True, "ok"
  except Exception as e:  # noqa: BLE001
    return False, f"{type(e).__name__}: {e}"


class _Encoder:
  def __init__(self, path: Path, size: tuple[int, int], codec: str, fps: int = FPS):
    import av
    from fractions import Fraction
    self.av = av
    self.c = av.open(str(path), mode="w")
    self.s = self.c.add_stream(CODECS.get(codec, codec), rate=fps)
    self.s.width, self.s.height = size
    self.s.pix_fmt = "yuv420p"
    self.s.time_base = Fraction(1, fps)
    if codec in ("h264", "libx264"):
      self.s.options = {"crf": "23", "g": "2", "preset": "veryfast"}
    elif codec in ("av1", "libsvtav1"):
      self.s.options = {"crf": "30", "g": "2", "preset": "8"}
    self.k = 0

  def add(self, rgb: np.ndarray) -> None:
    f = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    f.pts = self.k
    self.k += 1
    for p in self.s.encode(f):
      self.c.mux(p)

  def close(self) -> None:
    for p in self.s.encode():
      self.c.mux(p)
    self.c.close()


def render_episode(v: Variation, qpos_full: np.ndarray, video_keys: list[str],
                   out_paths: dict, size: tuple[int, int] = (256, 256), codec: str = "h264",
                   object_kind: str = "box", stats_every: int = 5,
                   quality: str = "full", fps: int = FPS, exposure: dict | None = None) -> dict:
  """Write one mp4 per key in ``video_keys``; return per-key image stats in [0, 1]
  (sum, sumsq, min, max per channel, pixel count) from every ``stats_every``-th frame.
  ``exposure`` (--wide, r2s/wide.py) is applied to every rendered frame of every camera
  before encoding; the stats describe the stored (exposed) images."""
  sc = build_episode_scene(v, object_kind=object_kind, render_size=size)
  m = sc.model
  flags = _apply_quality(m, quality)
  d = mujoco.MjData(m)
  w, h = size
  r = mujoco.Renderer(m, h, w)
  encs = {k: _Encoder(Path(out_paths[k]), size, codec, fps=fps) for k in video_keys}
  stats = {k: {"sum": np.zeros(3), "sumsq": np.zeros(3), "min": np.ones(3),
               "max": np.zeros(3), "n": 0} for k in video_keys}
  if exposure:
    from .wide import apply_exposure, noise_rng
    nrng = {k: noise_rng(v.seed, j) for j, k in enumerate(video_keys)}
  try:
    for i, q in enumerate(qpos_full):
      d.qpos[:] = q
      mujoco.mj_forward(m, d)
      for k in video_keys:
        r.update_scene(d, camera=CAMERAS[k])
        for f, val in flags.items():
          r.scene.flags[f] = val
        img = r.render()
        if exposure:
          img = apply_exposure(img, rng=nrng[k], **exposure)
        encs[k].add(img)
        if i % stats_every == 0:
          x = img.reshape(-1, 3).astype(np.float64) / 255.0
          st = stats[k]
          st["sum"] += x.sum(0)
          st["sumsq"] += (x * x).sum(0)
          st["min"] = np.minimum(st["min"], x.min(0))
          st["max"] = np.maximum(st["max"], x.max(0))
          st["n"] += len(x)
  finally:
    for e in encs.values():
      e.close()
    r.close()
  return {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in s.items()}
          for k, s in stats.items()}
