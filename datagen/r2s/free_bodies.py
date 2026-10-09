"""Extra free bodies on the table (distractors): one code path for every shape.

Shared by the OOD ladder's distractor rung (eval/ood.py, cylinders and spheres) and the
multi-task scene (P3-A, extra cubes). A body is a plain dict, stored in the Variation as
``v.extras["free_bodies"]`` so it round-trips through JSON and episodes.jsonl:

  {"name": "extra_0", "shape": "box" | "cylinder" | "sphere",
   "size": 0.03,          # edge (box) or diameter (cylinder, sphere), metres
   "height": 0.03,        # cylinder height (box/sphere: = size)
   "mass": 0.03, "friction": 0.6, "rgb": [r, g, b], "xy": [x, y], "yaw": 0.0}

scene.build_episode_scene adds each one as a body with a FREE JOINT and one primitive geom
with the world collision bits (same as the cube): it falls, slides, is pushed and can be
picked up like any object. No weld, no mocap, no equality constraint, no actuator. It starts
0.5 mm above the table and settles during the harness's settle phase like the cube.
"""

from __future__ import annotations

import colorsys
import math

import mujoco
import numpy as np

from .paths import mj_scene, rm

SHAPES = ("box", "cylinder", "sphere")
PREFIX = "extra_"


def half_height(b: dict) -> float:
  if b["shape"] == "cylinder":
    return float(b.get("height", b["size"])) / 2
  return float(b["size"]) / 2


def add_free_bodies(spec: mujoco.MjSpec, bodies: list[dict]) -> None:
  for b in bodies:
    shape, s = b["shape"], float(b["size"])
    if shape == "box":
      gtype, gsize, condim, fr = mujoco.mjtGeom.mjGEOM_BOX, (s / 2, s / 2, s / 2), 4, (b["friction"], 0.005, 0.0001)
    elif shape == "cylinder":
      gtype, gsize, condim, fr = (mujoco.mjtGeom.mjGEOM_CYLINDER, (s / 2, half_height(b), 0.0), 4,
                                  (b["friction"], 0.005, 0.0001))
    elif shape == "sphere":
      # condim 6 adds rolling friction so a nudged ball comes to rest instead of rolling away.
      gtype, gsize, condim, fr = mujoco.mjtGeom.mjGEOM_SPHERE, (s / 2, 0.0, 0.0), 6, (b["friction"], 0.005, 0.0005)
    else:
      raise ValueError(f"free body shape {shape!r} not in {SHAPES}")
    yaw = float(b.get("yaw", 0.0))
    body = spec.worldbody.add_body(
      name=b["name"], pos=(b["xy"][0], b["xy"][1], half_height(b) + 0.0005),
      quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    body.add_freejoint(name=b["name"] + "_joint")
    body.add_geom(name=b["name"] + "_geom", type=gtype, size=gsize, mass=float(b["mass"]),
                  friction=fr, rgba=(*b["rgb"], 1.0), contype=rm.WORLD_CONTYPE,
                  conaffinity=rm.WORLD_CONAFFINITY, condim=condim)


def _dist_to_square(p: np.ndarray, centre: np.ndarray, half: float) -> float:
  d = np.maximum(np.abs(p - centre) - half, 0.0)
  return float(np.linalg.norm(d))


def _unlike_hue(rng: np.random.Generator, target_rgb, min_hue_gap: float) -> list[float]:
  h0 = colorsys.rgb_to_hsv(*target_rgb)[0]
  while True:
    h = float(rng.uniform())
    gap = abs(h - h0)
    if min(gap, 1 - gap) >= min_hue_gap:
      break
  r, g, b = colorsys.hsv_to_rgb(h, float(rng.uniform(0.5, 0.95)), float(rng.uniform(0.5, 0.95)))
  return [float(r), float(g), float(b)]


def sample_free_bodies(v, rng: np.random.Generator, n: int, shapes=("cylinder", "sphere"),
                       size_range=(0.020, 0.035), rgbs: list | None = None,
                       min_from_object: float = 0.06, min_from_bin_wall: float = 0.06,
                       min_between: float = 0.05, reach=(0.10, 0.28),
                       bearing_deg=(-75.0, 75.0), min_hue_gap: float = 0.25,
                       mass_range=(0.015, 0.040), friction: float = 0.6,
                       prefix: str = PREFIX) -> list[dict]:
  """Draw ``n`` bodies on the table: centre >= ``min_from_object`` from the cube centre,
  >= ``min_from_bin_wall`` from the bin's outer walls (a square), >= ``min_between`` apart,
  inside the arm's reach band (``reach`` m from the pan axis, ``bearing_deg`` from +x), so
  the arm could touch them. Colours: ``rgbs`` if given, else hue >= ``min_hue_gap`` away
  from the cube's. Deterministic in ``rng``."""
  from .variations import PAN_AXIS
  obj = np.asarray(v.obj_xy, float)
  binc = np.asarray(v.bin_xy, float)
  bin_half = mj_scene.BIN_INNER_HALF + mj_scene.BIN_WALL_THICKNESS
  out: list[dict] = []
  for k in range(n):
    shape = shapes[int(rng.integers(len(shapes)))]
    size = float(rng.uniform(*size_range))
    for _ in range(2000):
      r = rng.uniform(*reach)
      a = math.radians(rng.uniform(*bearing_deg))
      p = PAN_AXIS + r * np.array([math.cos(a), math.sin(a)])
      if np.linalg.norm(p - obj) < min_from_object:
        continue
      if _dist_to_square(p, binc, bin_half) < min_from_bin_wall:
        continue
      if any(np.linalg.norm(p - np.asarray(b["xy"])) < min_between for b in out):
        continue
      break
    else:
      raise RuntimeError("could not place a free body with these constraints")
    rgb = list(rgbs[k]) if rgbs is not None else _unlike_hue(rng, v.object_rgb, min_hue_gap)
    out.append({"name": f"{prefix}{k}", "shape": shape, "size": size,
                "height": size, "mass": float(rng.uniform(*mass_range)), "friction": friction,
                "rgb": rgb, "xy": [float(p[0]), float(p[1])],
                "yaw": float(rng.uniform(-math.pi, math.pi))})
  return out
