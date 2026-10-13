"""The `friend_bowl` scene profile: the friend's rig as a datagen scene.

A profile is a JSON file (kept on the server, because it points at frames-derived files):
  {
    "name": "friend_bowl",
    "sessions": [ {"id": 0, "weight": n_train_episodes, "camera": {x, y, z, yaw, tilt_a, tilt_b, fovy},
                   "plate": "/abs/plate_s00.png", "ortho": "/abs/ortho_s00.png"}, ... ],
    "ortho_extent": {"x": [x0, x1], "y": [y0, y1]},       # base-frame metres covered by ortho PNGs
    "wrist_cam": null | {"dpos": [..3], "drot": [..3], "fovy": f},   # stage-2 correction
    "arm_offsets_deg": [5],                                  # stage-1 fitted joint offsets
    "cube": {"size": s, "rgb": [r, g, b]},
    "bowl": {"rim_radius": rt, "height": h, "base_radius": rb, "rgb": [r, g, b]},
    "placements_csv": "/abs/placements_train.csv",          # measured (cube_xy, bowl_xy) pairs
    "placement_jitter_m": 0.015,                             # modest widening, stated in the gate
    "arm_rgb": [r, g, b],
    "composite": true                                        # cam_head = sim over the real plate
  }

What it changes in the engine (nothing else): object/bowl size and start positions, the place
container (a bowl made of a floor disc + a ring of tilted wall boxes, named bin_floor/bin_wall_*
so the verifier's contact bookkeeping is unchanged), the look (black arm, real table texture at
its true position, real cube/bowl colours), the front camera (= fitted cam_head of a sampled
session) and the wrist camera correction. Physics rules and the verifier are untouched.
"""

from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

N_WALL = 20
WALL_T = 0.003
FLOOR_T = 0.003


def _engine():
  """so101_pickplace's mj_scene / robot_model, the same module objects the engine uses."""
  import sys
  root = Path(__file__).resolve().parents[1]
  for d in (root / "task", root / "datagen"):
    if str(d) not in sys.path:
      sys.path.insert(0, str(d))
  from so101_pickplace import mj_scene, robot_model
  return mj_scene, robot_model


@lru_cache(4)
def load_profile(path: str) -> dict:
  p = json.loads(Path(path).read_text())
  if p.get("placements_csv"):
    rows = list(csv.DictReader(open(p["placements_csv"])))
    p["_placements"] = np.array([[float(r["cube_x"]), float(r["cube_y"]), float(r["bowl_x"]), float(r["bowl_y"])]
                                 for r in rows])
  return p


def bowl_floor_radius(prof) -> float:
  b = prof["bowl"]
  return b["base_radius"] - WALL_T


def apply_bin_constants(prof) -> None:
  """Point the planner/verifier's container constants at the bowl (module attributes are read at
  call time). INNER_HALF = floor radius / sqrt(2): the success square fits inside the floor disc."""
  mj_scene, _ = _engine()
  b = prof["bowl"]
  mj_scene.BIN_FLOOR_THICKNESS = FLOOR_T
  mj_scene.BIN_WALL_HEIGHT = b["height"] - FLOOR_T - mj_scene.BIN_CLEARANCE
  mj_scene.BIN_INNER_HALF = bowl_floor_radius(prof) / math.sqrt(2)
  # the planner keeps the carried object high within INNER_HALF + WALL_THICKNESS (+ margins) of the
  # centre: make that the rim radius, so the object clears the bowl's flared rim
  mj_scene.BIN_WALL_THICKNESS = b["rim_radius"] - mj_scene.BIN_INNER_HALF


def bowl_spec(prof):
  """Bowl = floor disc (cylinder) + N_WALL tilted boxes on the frustum surface. Body origin at
  table level under the bowl centre, like mj_scene.bin_spec."""
  import mujoco
  mj_scene, rm = _engine()
  b = prof["bowl"]
  rt, h, rb = b["rim_radius"], b["height"], b["base_radius"]
  rgba = (*b.get("rgb", [0.92, 0.92, 0.92]), 1.0)
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="bin")
  common = dict(rgba=rgba, contype=rm.WORLD_CONTYPE, conaffinity=rm.WORLD_CONAFFINITY, condim=3,
                friction=(0.8, 0.005, 0.0001))
  body.add_geom(name="bin_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(rb, FLOOR_T / 2, 0),
                pos=(0, 0, mj_scene.BIN_CLEARANCE + FLOOR_T / 2), **common)
  z0 = mj_scene.BIN_CLEARANCE + FLOOR_T
  slant = math.hypot(rt - rb, h - z0)
  tilt = math.atan2(rt - rb, h - z0)          # wall leans outward by this angle from vertical
  rm_ = (rt + rb) / 2
  half_w = math.pi * (rt + rb) / N_WALL * 0.55  # slight overlap between panels
  for k in range(N_WALL):
    a = 2 * math.pi * k / N_WALL
    # panel frame: x radial (thickness), y tangential, z along the slant
    R = (np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
         @ np.array([[math.cos(tilt), 0, math.sin(tilt)], [0, 1, 0], [-math.sin(tilt), 0, math.cos(tilt)]]))
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    pos = (rm_ * math.cos(a), rm_ * math.sin(a), (z0 + h) / 2)
    body.add_geom(name=f"bin_wall_{k:02d}", type=mujoco.mjtGeom.mjGEOM_BOX, size=(WALL_T / 2, half_w, slant / 2),
                  pos=pos, quat=q.tolist(), **common)
  return spec


def sample_scene(prof, rng, xy_ok) -> dict:
  """Draw (session, cube_xy, bowl_xy, cube yaw) from the measured distribution + jitter.
  xy_ok(cube_xy, bowl_xy) -> bool is the engine's reach/separation test."""
  P = prof["_placements"]
  j = float(prof.get("placement_jitter_m", 0.015))
  w = np.array([s["weight"] for s in prof["sessions"]], float)
  sess = prof["sessions"][int(rng.choice(len(w), p=w / w.sum()))]
  for tries in range(1, 2001):
    r = P[int(rng.integers(len(P)))]
    c = r[:2] + rng.normal(0, j, 2)
    b = r[2:] + rng.normal(0, j, 2)
    if xy_ok(c, b):
      return {"session": sess["id"], "obj_xy": c.tolist(), "bin_xy": b.tolist(),
              "obj_yaw": float(rng.uniform(-math.pi / 4, math.pi / 4)), "tries": tries}
  raise RuntimeError("no feasible placement drawn from the measured distribution")


def session_of(prof, sid):
  return next(s for s in prof["sessions"] if s["id"] == sid)


def customise_spec(spec, v, prof) -> None:
  """Called by scene.build_episode_scene for the friend profile, BEFORE compile:
  table texture, black arm, front camera = cam_head, wrist camera correction."""
  import mujoco
  from friendrig.simcam import cam_pose
  f = v.extras["friend"]
  s = session_of(prof, f["session"])
  ex = prof["ortho_extent"]
  x0, x1 = ex["x"]; y0, y1 = ex["y"]
  # table texture: ortho PNG rows = x from x1 (top) to x0, cols = y from y1 (left) to y0
  tex = spec.texture("table_tex")
  spec.delete(tex)
  spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=s["ortho"])
  mat = spec.material("table_mat")
  mat.texrepeat = [1, 1]
  mat.texuniform = False
  mat.reflectance = 0.0
  table = spec.geom("table")
  table.pos = [(x0 + x1) / 2, (y0 + y1) / 2, 0.0]
  table.size = [(x1 - x0) / 2, (y1 - y0) / 2, 0.01]
  table.quat = f.get("table_quat", [1, 0, 0, 0])
  # front camera = fitted cam_head
  p = [s["camera"][k] for k in ("x", "y", "z", "yaw", "tilt_a", "tilt_b", "fovy")]
  pos, R = cam_pose(p)
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, R.flatten())
  cam = spec.camera("front")
  cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT  # the engine made it with xyaxes
  cam.pos, cam.quat, cam.fovy = pos.tolist(), q.tolist(), float(p[6])


def customise_robot(robot, prof) -> None:
  """Robot spec edits before attach: black material, wrist camera correction."""
  import mujoco
  from friendrig.simcam import _rx, _ry, _rz
  rgb = prof.get("arm_rgb", [0.05, 0.05, 0.06])
  for m in robot.materials:
    m.rgba = [*rgb, 1.0]
  wc = prof.get("wrist_cam")
  if wc:
    cam = robot.camera("wrist_cam")
    R0 = np.zeros(9)
    mujoco.mju_quat2Mat(R0, np.asarray(cam.quat, float))
    R0 = R0.reshape(3, 3)
    pos0 = np.asarray(cam.pos, float)
    if wc.get("flip"):
      # The friend's wrist camera sits on the OTHER face of the gripper (friendrig/wristcam.py):
      # turn the whole camera_mount body (camera, board, lens, mount mesh) 180 deg about the
      # wrist_roll axis. Its two collision boxes would then overlap the moving jaw's sweep, and the
      # real housing's shape is not measured, so they are made non-colliding (visual only).
      mount = robot.body("camera_mount")
      mount.quat = [0.0, 0.0, 0.0, 1.0]
      for g in ("camera_box1", "camera_box2"):
        robot.geom(g).contype = 0
        robot.geom(g).conaffinity = 0
    R = R0 @ _rx(wc["drot"][0]) @ _ry(wc["drot"][1]) @ _rz(wc["drot"][2])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    cam.pos = (pos0 + np.asarray(wc["dpos"])).tolist()
    cam.quat = q.tolist()
    cam.fovy = float(wc["fovy"])


@lru_cache(8)
def plate_image(path: str) -> np.ndarray:
  import cv2
  return cv2.imread(path)[..., ::-1].copy()


def apply_to_variation(v, profile_path: str):
  """Overwrite the generic draws of a Variation with friend_bowl draws (deterministic from v.seed,
  separate stream so the physics DR draws - masses, frictions, servos - are unchanged)."""
  from dataclasses import replace
  prof = load_profile(profile_path)
  rng = np.random.default_rng([v.seed, 11])
  mj_scene, _ = _engine()
  import sys
  from r2s.variations import _in_reach  # same reach band as the generic sampler
  cube = prof["cube"]
  bowl = prof["bowl"]
  sep = bowl["rim_radius"] + cube["size"] * 0.75 + 0.01

  def xy_ok(c, b):
    return _in_reach(np.asarray(c)) and _in_reach(np.asarray(b)) and np.linalg.norm(np.asarray(c) - b) >= sep

  d = sample_scene(prof, rng, xy_ok)
  dr = bool(prof.get("look_dr", True))
  jit = (lambda rgb: np.clip(np.asarray(rgb) * rng.uniform(0.9, 1.1, 3), 0, 1).tolist()) if dr else (lambda rgb: list(rgb))
  extras = dict(v.extras or {})
  extras["friend"] = {"profile": profile_path, "session": d["session"], "placement_tries": d["tries"],
                      "composite": bool(prof.get("composite", False))}
  kw = dict(obj_xy=d["obj_xy"], bin_xy=d["bin_xy"], obj_yaw=d["obj_yaw"], obj_size=float(cube["size"]),
            object_rgb=jit(cube["rgb"]), bin_rgb=jit(bowl.get("rgb", [0.92, 0.92, 0.92])), extras=extras)
  if not dr:
    kw.update(light_dir=[0.0, 0.0, -1.0], light_diffuse=0.6, ambient=0.3, headlight=0.35)
  return replace(v, **kw)
