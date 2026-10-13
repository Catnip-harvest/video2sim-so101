"""Build the MuJoCo model for one episode.

Same physical pieces as ``so101_pickplace.mj_scene.build_cpu_scene`` (which the
12-case grasp-physics gate uses): the Menagerie SO-101 with the task's load-time edits,
STS3215 position servos, the task's object and bin specs, a table plane, and the same
solver (implicitfast, elliptic cones, impratio 10, 10/20 iterations). Differences:

* timestep 1/210 s (4.76 ms) instead of 5 ms, so exactly 7 physics steps make one
  30 Hz control/recording frame;
* per-episode DR is applied to masses, frictions, gains and torque limits;
* render-only things are added: table texture, lights, a front camera. The robot's
  own ``wrist_cam`` (Menagerie camera mount) is kept.

Nothing here adds an equality constraint, weld, mocap body or actuator on the object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

from .free_bodies import add_free_bodies
from .paths import mj_scene, rm
from .variations import Variation

FPS = 30
SUBSTEPS = 7
TIMESTEP = 1.0 / (FPS * SUBSTEPS)

# Geom categories used by the contact bookkeeping.
CAT_OTHER, CAT_TABLE, CAT_OBJECT, CAT_BIN_FLOOR, CAT_BIN_WALL = 0, 1, 2, 3, 4
CAT_FIXED_JAW, CAT_MOVING_JAW, CAT_ROBOT = 5, 6, 7
CAT_EXTRA = 8  # extra free bodies (distractors), see free_bodies.py
FIXED_JAW = re.compile(r"fixed_jaw_.*")
MOVING_JAW = re.compile(r"moving_jaw_.*")


@dataclass
class EpisodeScene:
  model: mujoco.MjModel
  data: mujoco.MjData
  arm_qadr: np.ndarray
  arm_act: np.ndarray
  g_qadr: int
  g_act: int
  obj_body: int
  obj_qadr: int
  obj_dadr: int
  grip_body: int
  site: int
  geom_cat: np.ndarray
  joint_lo: np.ndarray  # (6,) arm + gripper
  joint_hi: np.ndarray
  effort: np.ndarray  # (6,)
  extra_bodies: tuple = ()  # body ids of v.extras["free_bodies"], in that order


def _cam_xyaxes(pos: np.ndarray, lookat: np.ndarray) -> list[float]:
  f = lookat - pos
  f = f / np.linalg.norm(f)
  x = np.cross(f, np.array([0.0, 0.0, 1.0]))
  x /= np.linalg.norm(x)
  y = np.cross(x, f)
  return [*x.tolist(), *y.tolist()]


def _add_spot_mark(spec: mujoco.MjSpec, k: int, mark: dict) -> None:
  """A flat marker on the table (Hanoi spot): a dark ring around a light disc. Visual only:
  contype = conaffinity = 0, so nothing ever touches it and the physics is unchanged."""
  outer, inner = mark["radius"]
  x, y = mark["xy"]
  for name, r, z, rgb in ((f"spot_{k}_ring", outer, 0.0003, (0.08, 0.08, 0.08)),
                          (f"spot_{k}_centre", inner, 0.0006, (0.93, 0.93, 0.90))):
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(r, z, 0.0),
                            pos=(x, y, 0.0), rgba=(*rgb, 1.0), contype=0, conaffinity=0,
                            group=2, mass=0.0)


def build_episode_scene(v: Variation, object_kind: str = "box",
                        render_size: tuple[int, int] = (256, 256)) -> EpisodeScene:
  spec = mujoco.MjSpec()
  spec.option.timestep = TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = mj_scene.IMPRATIO
  spec.option.iterations = mj_scene.SOLVER_ITERATIONS
  spec.option.ls_iterations = mj_scene.SOLVER_LS_ITERATIONS

  # --- render-only look ---------------------------------------------------------
  spec.visual.global_.offwidth = max(640, render_size[0])
  spec.visual.global_.offheight = max(480, render_size[1])
  spec.visual.headlight.diffuse = [v.headlight] * 3
  spec.visual.headlight.ambient = [v.ambient] * 3
  spec.visual.headlight.specular = [0.1] * 3
  spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=v.table_rgb1, rgb2=v.table_rgb2, width=256, height=256)
  mat = spec.add_material(name="table_mat", texrepeat=[v.table_texrepeat] * 2,
                          reflectance=0.05)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "table_tex"
  spec.worldbody.add_light(name="key", dir=v.light_dir, diffuse=[v.light_diffuse] * 3,
                           specular=[0.2] * 3, castshadow=True,
                           pos=(-np.array(v.light_dir) * 1.2).tolist(),
                           type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  spec.worldbody.add_camera(name="front", pos=v.cam_pos,
                            xyaxes=_cam_xyaxes(np.array(v.cam_pos), np.array(v.cam_lookat)),
                            fovy=v.cam_fovy)

  # --- physical scene (mirrors mj_scene.build_cpu_scene) -------------------------
  spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=(0.8, 0.8, 0.01), material="table_mat",
                          contype=rm.WORLD_CONTYPE, conaffinity=rm.WORLD_CONAFFINITY)
  robot = rm.so101_spec()
  robot.option.timestep = TIMESTEP  # avoid the attach-conflict warning; parent wins anyway
  rm.add_position_actuators(robot, arm_effort_limit=v.arm_effort,
                            gripper_effort_limit=v.gripper_effort)
  spec.attach(robot, prefix="", frame=spec.worldbody.add_frame(pos=(0.0, 0.0, 0.0)))

  extras = v.extras or {}
  half = v.obj_size / 2
  obj = mj_scene.object_spec(kind=object_kind, size=v.obj_size, height=v.obj_size,
                             mass=v.obj_mass, friction=(v.obj_friction, 0.005, 0.0001),
                             rgba=(*v.object_rgb, 1.0))
  # Start 0.5 mm above the table; the unrecorded settle phase lets it come to rest.
  # extras["object_z"] (Hanoi) starts it higher instead, e.g. on top of another cube.
  obj_z = float(extras.get("object_z", half + 0.0005))
  obj.body("object").pos = (v.obj_xy[0], v.obj_xy[1], obj_z)
  obj.body("object").quat = (np.cos(v.obj_yaw / 2), 0.0, 0.0, np.sin(v.obj_yaw / 2))
  spec.attach(obj, prefix="", frame=spec.worldbody.add_frame())
  if not extras.get("no_bin"):  # Hanoi has no tray
    b = mj_scene.bin_spec()
    for g in b.geoms:
      g.rgba = (*v.bin_rgb, 1.0)
    spec.attach(b, prefix="", frame=spec.worldbody.add_frame(pos=(v.bin_xy[0], v.bin_xy[1], 0.0)))
  for k, mark in enumerate(extras.get("spot_marks") or []):
    _add_spot_mark(spec, k, mark)
  extra = list(extras.get("free_bodies") or [])
  if extra:  # distractors: free joint + plain contacts, added after everything else
    add_free_bodies(spec, extra)

  model = spec.compile()
  data = mujoco.MjData(model)

  def jid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)

  def aid(n):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

  arm_qadr = np.array([model.jnt_qposadr[jid(n)] for n in rm.ARM_JOINTS])
  arm_act = np.array([aid(n) for n in rm.ARM_JOINTS])
  g_qadr = int(model.jnt_qposadr[jid(rm.GRIPPER_JOINT)])
  g_act = aid(rm.GRIPPER_JOINT)

  # Per-joint servo DR.
  for i, a in enumerate(arm_act):
    model.actuator_gainprm[a, 0] = v.arm_kp[i]
    model.actuator_biasprm[a, 1] = -v.arm_kp[i]
    model.actuator_biasprm[a, 2] = -v.arm_kv[i]
  model.actuator_gainprm[g_act, 0] = v.jaw_kp
  model.actuator_biasprm[g_act, 1] = -v.jaw_kp
  model.actuator_biasprm[g_act, 2] = -v.jaw_kv

  # Geom categories + finger friction DR.
  cat = np.zeros(model.ngeom, dtype=np.int8)
  extra_geoms = {b["name"] + "_geom" for b in extra}
  for g in range(model.ngeom):
    name = model.geom(g).name
    if name == "table":
      cat[g] = CAT_TABLE
    elif name == "object_geom":
      cat[g] = CAT_OBJECT
    elif name in extra_geoms:
      cat[g] = CAT_EXTRA
    elif name == "bin_floor":
      cat[g] = CAT_BIN_FLOOR
    elif name.startswith("bin_wall"):
      cat[g] = CAT_BIN_WALL
    elif FIXED_JAW.match(name):
      cat[g] = CAT_FIXED_JAW
      model.geom_friction[g, 0] = v.finger_friction
    elif MOVING_JAW.match(name):
      cat[g] = CAT_MOVING_JAW
      model.geom_friction[g, 0] = v.finger_friction
    elif model.geom_contype[g] == rm.ROBOT_CONTYPE:
      cat[g] = CAT_ROBOT

  all_j = [jid(n) for n in rm.ALL_JOINTS]
  obj_j = jid("object_joint")
  effort = np.array([v.arm_effort] * 5 + [v.gripper_effort])
  return EpisodeScene(
    model=model, data=data, arm_qadr=arm_qadr, arm_act=arm_act, g_qadr=g_qadr, g_act=g_act,
    obj_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object"),
    obj_qadr=int(model.jnt_qposadr[obj_j]), obj_dadr=int(model.jnt_dofadr[obj_j]),
    grip_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper"),
    site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"),
    geom_cat=cat,
    joint_lo=model.jnt_range[all_j, 0].copy(), joint_hi=model.jnt_range[all_j, 1].copy(),
    effort=effort,
    extra_bodies=tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b["name"]) for b in extra),
  )
