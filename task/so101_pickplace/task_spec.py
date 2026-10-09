"""Task spec: the contract between the demo-video worker and the RL task.

A task spec is a JSON file (format ``so101_pickplace_task_spec/v2``) that says where
the object starts, where it must end up, how the *object* moved in the human demo,
and two things taken from the human hand: grasp / release timing and approach side.
Pure Python + numpy (no mjlab / torch).

Design (object-centric, X-Sim style): the dense reward makes the object follow the
demo's object trajectory. The hand path is deliberately NOT tracked: a human arm
hangs from a shoulder above the table while the SO-101 stands on it, so hand paths do
not transfer. ``hand_path`` from spec v1 is ignored with a warning.

Frame: the SO-101 **robot base frame** -- origin at the base's mounting point on the
table, x forward (direction the arm points at shoulder_pan = 0), y to the arm's left,
z up; the table surface is z = 0. Units: metres, seconds, radians. The shoulder-pan axis
is at (0.0388, 0). Measured reach for a near-vertical (<= 15 deg) grasp of a 30 mm
object on the table: 0.06-0.32 m from the pan axis over pan +-110 deg
(measurements/so101_measurements.json); keep ranges inside ~0.10-0.28 m.

Resolution order for the spec file: ``spec_path`` argument of the env-cfg factory,
the ``SO101_TASK_SPEC`` environment variable, then ``specs/default_task_spec.json``.

See README.md ("Task spec format") for the field-by-field description.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "so101_pickplace_task_spec/v2"
ACCEPTED_FORMATS = (FORMAT, "so101_pickplace_task_spec/v1")
SPECS_DIR = Path(__file__).resolve().parents[1] / "specs"
DEFAULT_SPEC_PATH = SPECS_DIR / "default_task_spec.json"
ENV_VAR = "SO101_TASK_SPEC"
PAN_AXIS_XY = (0.0388, 0.0)
MAX_REACH_FROM_PAN_AXIS = 0.32
MIN_REACH_FROM_PAN_AXIS = 0.06


@dataclass
class Range2:
  lo: float
  hi: float

  @staticmethod
  def parse(v) -> "Range2":
    if isinstance(v, (int, float)):
      return Range2(float(v), float(v))
    lo, hi = float(v[0]), float(v[1])
    if hi < lo:
      raise ValueError(f"range {v} has hi < lo")
    return Range2(lo, hi)

  def as_tuple(self) -> tuple[float, float]:
    return (self.lo, self.hi)


@dataclass
class ObjectSpec:
  kind: str = "box"  # "box" | "cylinder" (see README: cylinder caveat on MuJoCo Warp)
  size: float = 0.030  # box edge or cylinder diameter (m)
  height: float = 0.040  # cylinder height (box: ignored, = size)
  mass: float = 0.025
  x: Range2 = field(default_factory=lambda: Range2(0.17, 0.27))
  y: Range2 = field(default_factory=lambda: Range2(-0.10, 0.06))
  yaw: Range2 = field(default_factory=lambda: Range2(-math.pi, math.pi))

  @property
  def half_height(self) -> float:
    return self.size / 2 if self.kind == "box" else self.height / 2

  @property
  def half_width(self) -> float:
    return self.size / 2


@dataclass
class PlaceSpec:
  kind: str = "bin"  # only "bin" is implemented
  x: Range2 = field(default_factory=lambda: Range2(0.10, 0.20))
  y: Range2 = field(default_factory=lambda: Range2(0.15, 0.21))
  min_separation: float = 0.11  # object start <-> bin centre, xy
  success_xy_margin: float = 0.005  # object centre this far inside the bin walls


@dataclass
class ApproachSpec:
  """Approach side, from the human hand: the finger-axis direction (base frame) near
  the grasp. Small reward only; the robot may need a different wrist angle."""

  direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
  weight: float = 0.25
  gate_std: float = 0.06


@dataclass
class ObjectTrajectorySpec:
  """Demo object-centre trajectory, robot base frame, demo clock."""

  waypoints: np.ndarray  # (K, 4): x, y, z, t
  grasp_time: float  # demo clock: hand closes on the object
  release_time: float  # demo clock: hand lets go
  robot_time_scale: float = 2.0  # robot time = demo time * scale
  smoothing_window_s: float = 0.25
  resample_dt: float = 0.1
  weight: float = 4.0
  std: float = 0.05
  time_slack_s: float = 1.0

  def reference(self) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Smoothed, uniformly resampled reference in ROBOT time.

    Returns (t (N,), pos (N, 3), grasp_t, release_t), robot clock, t[0] = 0.
    """
    wp = np.asarray(self.waypoints, dtype=np.float64)
    order = np.argsort(wp[:, 3], kind="stable")
    wp = wp[order]
    t_demo = wp[:, 3] - wp[0, 3]
    scale = self.robot_time_scale
    t_end = float(t_demo[-1]) * scale
    dt = self.resample_dt
    t = np.arange(0.0, t_end + 1e-9, dt) if t_end > 0 else np.zeros(1)
    pos = np.stack([np.interp(t, t_demo * scale, wp[:, i]) for i in range(3)], axis=1)
    # Centred moving average (edge-padded) to remove video/pose-estimation jitter.
    k = int(round(self.smoothing_window_s / dt))
    if k >= 2 and len(t) > k:
      kern = np.ones(k) / k
      pad = (k // 2, k - 1 - k // 2)
      pos = np.stack(
        [np.convolve(np.pad(pos[:, i], pad, mode="edge"), kern, mode="valid") for i in range(3)],
        axis=1,
      )
    g = (self.grasp_time - wp[0, 3]) * scale
    r = (self.release_time - wp[0, 3]) * scale
    return t, pos, float(g), float(r)


@dataclass
class TaskSpec:
  object: ObjectSpec = field(default_factory=ObjectSpec)
  place: PlaceSpec = field(default_factory=PlaceSpec)
  approach: ApproachSpec = field(default_factory=ApproachSpec)
  trajectory: ObjectTrajectorySpec | None = None
  mount_height: float = 0.0
  source: dict = field(default_factory=dict)
  path: str | None = None


def _check_reach(name: str, x: Range2, y: Range2, problems: list[str]) -> None:
  for xx in (x.lo, x.hi):
    for yy in (y.lo, y.hi):
      r = math.hypot(xx - PAN_AXIS_XY[0], yy - PAN_AXIS_XY[1])
      if r > MAX_REACH_FROM_PAN_AXIS or r < MIN_REACH_FROM_PAN_AXIS:
        problems.append(
          f"{name} corner ({xx:.3f}, {yy:.3f}) is {r:.3f} m from the pan axis; "
          f"measured near-vertical reach is {MIN_REACH_FROM_PAN_AXIS}-"
          f"{MAX_REACH_FROM_PAN_AXIS} m"
        )


def _parse_trajectory(d: dict, obj: ObjectSpec) -> ObjectTrajectorySpec:
  wps = np.asarray(d.get("waypoints", []), dtype=np.float64)
  if wps.ndim != 2 or wps.shape[1] != 4 or len(wps) < 2:
    raise ValueError("object_trajectory.waypoints must be a list of >= 2 [x, y, z, t]")
  if np.any(np.diff(np.sort(wps[:, 3])) < 0) or not np.all(np.isfinite(wps)):
    raise ValueError("object_trajectory.waypoints must be finite")
  t0, t1 = float(wps[:, 3].min()), float(wps[:, 3].max())
  grasp = float(d.get("grasp_time", t0))
  release = float(d.get("release_time", t1))
  if not (t0 <= grasp <= release <= t1):
    raise ValueError(
      f"need waypoint t0 <= grasp_time <= release_time <= t_end, got "
      f"{t0}, {grasp}, {release}, {t1}"
    )
  if np.any(wps[:, 2] < -0.005):
    raise ValueError("object_trajectory z must be >= 0 (table is z = 0)")
  traj = ObjectTrajectorySpec(
    waypoints=wps,
    grasp_time=grasp,
    release_time=release,
    robot_time_scale=float(d.get("robot_time_scale", 2.0)),
    smoothing_window_s=float(d.get("smoothing_window_s", 0.25)),
    resample_dt=float(d.get("resample_dt", 0.1)),
    weight=float(d.get("weight", 4.0)),
    std=float(d.get("std", 0.05)),
    time_slack_s=float(d.get("time_slack_s", 1.0)),
  )
  if traj.robot_time_scale <= 0 or traj.resample_dt <= 0 or traj.std <= 0:
    raise ValueError("robot_time_scale, resample_dt and std must be > 0")
  return traj


def parse_task_spec(d: dict, path: str | None = None) -> TaskSpec:
  fmt = d.get("format")
  if fmt not in ACCEPTED_FORMATS:
    raise ValueError(f"task spec format must be {FORMAT!r}, got {fmt!r}")
  if d.get("frame", "robot_base") != "robot_base":
    raise ValueError("only frame='robot_base' is supported")
  o = d.get("object", {})
  rng = o.get("start_pose_range", {})
  obj = ObjectSpec(
    kind=o.get("kind", "box"),
    size=float(o.get("size", 0.030)),
    height=float(o.get("height", o.get("size", 0.030))),
    mass=float(o.get("mass", 0.025)),
    x=Range2.parse(rng.get("x", (0.17, 0.27))),
    y=Range2.parse(rng.get("y", (-0.10, 0.06))),
    yaw=Range2.parse(rng.get("yaw", (-math.pi, math.pi))),
  )
  if obj.kind not in ("box", "cylinder"):
    raise ValueError(f"object.kind must be box|cylinder, got {obj.kind!r}")
  if not (0.020 <= obj.size <= 0.050):
    raise ValueError(
      f"object.size {obj.size} outside the 20-50 mm band this gripper was tested for"
    )
  if obj.kind == "box":
    obj.height = obj.size
  p = d.get("place_target", {})
  prng = p.get("pos_range", {})
  place = PlaceSpec(
    kind=p.get("kind", "bin"),
    x=Range2.parse(prng.get("x", (0.10, 0.20))),
    y=Range2.parse(prng.get("y", (0.15, 0.21))),
    min_separation=float(p.get("min_separation", 0.11)),
    success_xy_margin=float(p.get("success_xy_margin", 0.005)),
  )
  if place.kind != "bin":
    raise ValueError("place_target.kind: only 'bin' is implemented")
  a = d.get("approach", {})
  direction = tuple(float(v) for v in a.get("direction", (0.0, 0.0, -1.0)))
  n = math.sqrt(sum(v * v for v in direction))
  if len(direction) != 3 or n < 1e-6:
    raise ValueError("approach.direction must be a non-zero 3-vector")
  approach = ApproachSpec(
    direction=tuple(v / n for v in direction),  # type: ignore[arg-type]
    weight=float(a.get("weight", 0.25)),
    gate_std=float(a.get("gate_std", 0.06)),
  )
  if d.get("hand_path"):
    warnings.warn(
      "task spec 'hand_path' is ignored: the reward is object-centric (the human arm "
      "hangs from a shoulder, the SO-101 stands on the table; hand paths do not "
      "transfer). Put the demo's object motion in 'object_trajectory'.",
      stacklevel=2,
    )
  traj = None
  if d.get("object_trajectory"):
    traj = _parse_trajectory(d["object_trajectory"], obj)
  spec = TaskSpec(object=obj, place=place, approach=approach, trajectory=traj,
                  mount_height=float(d.get("mount_height", 0.0)),
                  source=d.get("source", {}) or {}, path=path)
  problems: list[str] = []
  _check_reach("object start", obj.x, obj.y, problems)
  _check_reach("place target", place.x, place.y, problems)
  if traj is not None:
    for x, y, z, _ in traj.waypoints:
      r = math.hypot(x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1])
      if r > MAX_REACH_FROM_PAN_AXIS + 0.03:
        problems.append(f"trajectory point ({x:.3f}, {y:.3f}, {z:.3f}) is {r:.3f} m out")
  if problems:
    raise ValueError("task spec outside the measured workspace:\n  " + "\n  ".join(problems))
  return spec


def load_task_spec(path: str | os.PathLike | None = None) -> TaskSpec:
  if path is None:
    path = os.environ.get(ENV_VAR) or DEFAULT_SPEC_PATH
  path = Path(path)
  with open(path, encoding="utf-8") as fh:
    return parse_task_spec(json.load(fh), path=str(path))
