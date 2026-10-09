"""Scene pieces shared by mjlab and the pure-MuJoCo tools (no mjlab / torch import).

* :func:`object_spec` - the graspable object (free body, primitive geom).
* :func:`bin_spec` - the shallow place bin (fixed body; mjlab wraps it in mocap).
* :func:`build_cpu_scene` - table plane + SO-101 (+ riser) + object (+ bin) compiled
  for plain CPU MuJoCo, with the same solver options as the mjlab task. Used by
  tests/test_grasp_physics.py and scripts/measure_so101.py.

Collision bitmasks: world things (table, object, bin, riser) are contype=1,
conaffinity=1; robot geoms are contype=2, conaffinity=3 (robot_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import robot_model as rm

# Solver options identical to env_cfg.py's SimulationCfg (and mjlab's lift-cube task).
TIMESTEP = 0.005
DECIMATION = 4  # 50 Hz policy
SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 20
IMPRATIO = 10.0

OBJECT_RGBA = (0.85, 0.25, 0.2, 1.0)
BIN_RGBA = (0.25, 0.45, 0.85, 1.0)

# Object defaults (see README "Object and mount choice").
OBJECT_KIND = "box"  # cylinder-box contacts are single-point in MuJoCo Warp
OBJECT_SIZE = 0.030  # edge length (box) or diameter (cylinder), metres
OBJECT_HEIGHT = 0.040  # cylinder height; box uses OBJECT_SIZE
OBJECT_MASS = 0.025  # kg (30 mm hardwood / PLA cube: ~15-25 g)
OBJECT_FRICTION = (0.8, 0.005, 0.0001)

# Bin: a shallow open tray.
BIN_INNER_HALF = 0.055  # inner half-width (11 cm x 11 cm inside)
BIN_WALL_HEIGHT = 0.025
BIN_WALL_THICKNESS = 0.005
BIN_FLOOR_THICKNESS = 0.004
BIN_CLEARANCE = 0.001  # floor bottom sits 1 mm above the table (no static-static contact)


def object_spec(
  kind: str = OBJECT_KIND,
  size: float = OBJECT_SIZE,
  height: float = OBJECT_HEIGHT,
  mass: float = OBJECT_MASS,
  friction: tuple[float, float, float] = OBJECT_FRICTION,
  rgba: tuple[float, float, float, float] = OBJECT_RGBA,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="object_joint")
  if kind == "box":
    gtype = mujoco.mjtGeom.mjGEOM_BOX
    gsize = (size / 2, size / 2, size / 2)
  elif kind == "cylinder":
    gtype = mujoco.mjtGeom.mjGEOM_CYLINDER
    gsize = (size / 2, height / 2, 0.0)
  else:
    raise ValueError(f"unknown object kind {kind!r}")
  body.add_geom(
    name="object_geom",
    type=gtype,
    size=gsize,
    mass=mass,
    friction=friction,
    rgba=rgba,
    contype=rm.WORLD_CONTYPE,
    conaffinity=rm.WORLD_CONAFFINITY,
    condim=4,
  )
  return spec


def object_half_height(kind: str = OBJECT_KIND, size: float = OBJECT_SIZE,
                       height: float = OBJECT_HEIGHT) -> float:
  return size / 2 if kind == "box" else height / 2


def bin_spec(
  inner_half: float = BIN_INNER_HALF,
  wall_height: float = BIN_WALL_HEIGHT,
  wall_thickness: float = BIN_WALL_THICKNESS,
  floor_thickness: float = BIN_FLOOR_THICKNESS,
) -> mujoco.MjSpec:
  """Open tray. Body origin = centre of the floor's top surface's footprint at table
  level (z=0 of the body is the table surface), so the place target is (x, y, 0)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="bin")
  common = dict(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    rgba=BIN_RGBA,
    contype=rm.WORLD_CONTYPE,
    conaffinity=rm.WORLD_CONAFFINITY,
    condim=3,
    friction=(0.8, 0.005, 0.0001),
  )
  outer = inner_half + wall_thickness
  body.add_geom(
    name="bin_floor",
    size=(outer, outer, floor_thickness / 2),
    pos=(0, 0, BIN_CLEARANCE + floor_thickness / 2),
    **common,
  )
  zc = BIN_CLEARANCE + floor_thickness + wall_height / 2
  for name, pos, size in (
    ("bin_wall_px", (inner_half + wall_thickness / 2, 0, zc), (wall_thickness / 2, outer, wall_height / 2)),
    ("bin_wall_nx", (-inner_half - wall_thickness / 2, 0, zc), (wall_thickness / 2, outer, wall_height / 2)),
    ("bin_wall_py", (0, inner_half + wall_thickness / 2, zc), (inner_half, wall_thickness / 2, wall_height / 2)),
    ("bin_wall_ny", (0, -inner_half - wall_thickness / 2, zc), (inner_half, wall_thickness / 2, wall_height / 2)),
  ):
    body.add_geom(name=name, pos=pos, size=size, **common)
  return spec


def bin_floor_top() -> float:
  return BIN_CLEARANCE + BIN_FLOOR_THICKNESS


def add_riser(spec: mujoco.MjSpec, mount_height: float) -> None:
  """Collision block under the robot base (the arm stands on it, z in [0, H])."""
  if mount_height <= 0.002:
    return
  spec.body("base").add_geom(
    name="riser",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.055, 0.05, (mount_height - 0.001) / 2),
    pos=(0.0, 0.0, -(mount_height - 0.001) / 2 - 0.001),
    rgba=(0.35, 0.35, 0.35, 1.0),
    contype=rm.WORLD_CONTYPE,
    conaffinity=rm.WORLD_CONAFFINITY | rm.ROBOT_CONTYPE,
    group=1,
  )


@dataclass
class CpuScene:
  model: mujoco.MjModel
  data: mujoco.MjData

  def jid(self, name: str) -> int:
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

  def qadr(self, name: str) -> int:
    return int(self.model.jnt_qposadr[self.jid(name)])

  def aid(self, name: str) -> int:
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

  def bid(self, name: str) -> int:
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

  def gid(self, name: str) -> int:
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

  def sid(self, name: str) -> int:
    return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)


def build_cpu_scene(
  mount_height: float,
  with_object: bool = True,
  with_bin: bool = False,
  object_kwargs: dict | None = None,
  object_pos: tuple[float, float, float] = (0.2, 0.0, 0.05),
  bin_pos: tuple[float, float] = (0.15, 0.15),
  arm_effort_limit: float = rm.ARM_EFFORT_LIMIT,
  gripper_effort_limit: float = rm.GRIPPER_EFFORT_LIMIT,
) -> CpuScene:
  """Plain-MuJoCo scene, robot base at (0, 0, mount_height), table = plane z=0."""
  spec = mujoco.MjSpec()
  spec.option.timestep = TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = IMPRATIO
  spec.option.iterations = SOLVER_ITERATIONS
  spec.option.ls_iterations = SOLVER_LS_ITERATIONS
  spec.worldbody.add_geom(
    name="table",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
    contype=rm.WORLD_CONTYPE,
    conaffinity=rm.WORLD_CONAFFINITY,
  )

  robot = rm.so101_spec()
  add_riser(robot, mount_height)
  rm.add_position_actuators(
    robot, arm_effort_limit=arm_effort_limit, gripper_effort_limit=gripper_effort_limit
  )
  frame = spec.worldbody.add_frame(pos=(0.0, 0.0, mount_height))
  spec.attach(robot, prefix="", frame=frame)

  if with_object:
    obj = object_spec(**(object_kwargs or {}))
    obj.body("object").pos = object_pos
    spec.attach(obj, prefix="", frame=spec.worldbody.add_frame())
  if with_bin:
    b = bin_spec()
    spec.attach(b, prefix="", frame=spec.worldbody.add_frame(pos=(bin_pos[0], bin_pos[1], 0.0)))

  model = spec.compile()
  return CpuScene(model=model, data=mujoco.MjData(model))


def gripper_axes(scene: CpuScene) -> tuple[np.ndarray, np.ndarray]:
  """(approach, closing) unit vectors in world: fingers point along -z of the
  gripper body; the moving jaw closes along -x of the gripper body."""
  R = scene.data.xmat[scene.bid("gripper")].reshape(3, 3)
  return -R[:, 2], -R[:, 0]
