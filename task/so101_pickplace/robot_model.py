"""SO-101 model, pure MuJoCo (no mjlab, no torch, no warp).

Everything that defines the *physical* robot lives here so the CPU grasp-physics
test (tests/test_grasp_physics.py), the workspace measurement script and the mjlab
robot config (so101_constants.py) all build exactly the same arm.

Source MJCF: MuJoCo Menagerie ``robotstudio_so101`` (Apache-2.0), sparse-cloned into
``manip/third_party/menagerie_so101`` and never edited. All changes below are made on
an ``MjSpec`` at load time:

1. The XML's six ``<position>`` actuators are deleted. mjlab adds its own (see
   so101_constants.py); the CPU test adds identical ones with
   :func:`add_position_actuators`.
2. A ``grasp_site`` is added to the ``gripper`` body (see GRASP_SITE_POS).
3. Fingertip / jaw-pad sites are added for contact bookkeeping.
4. Every collision geom gets a name (Menagerie leaves the arm boxes unnamed) so
   mjlab regexes and contact sensors can address them.
5. Collision bitmasks: robot geoms use contype=2, conaffinity=3; the table and the
   object use contype=1, conaffinity=1. So robot-vs-table, robot-vs-object and
   robot-vs-robot (non-adjacent links, e.g. the wrist folding back onto the shoulder)
   all collide; parent/child pairs are filtered by MuJoCo as usual. Measured: the
   zero pose has no self-contact; 12 % of uniformly random joint configurations
   self-penetrate (arm folded back onto the shoulder), which is exactly what
   self-collision is there to forbid.

No weld, no equality constraint, no mocap attachment is ever added to the gripper.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

##
# Paths.
##

TASK_ROOT = Path(__file__).resolve().parents[1]  # manip/task
MANIP_ROOT = TASK_ROOT.parent  # manip
SO101_DIR = MANIP_ROOT / "third_party" / "menagerie_so101" / "robotstudio_so101"
SO101_XML = SO101_DIR / "so101.xml"

##
# Joints.
##

ARM_JOINTS: tuple[str, ...] = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
)
GRIPPER_JOINT = "gripper"
ALL_JOINTS: tuple[str, ...] = (*ARM_JOINTS, GRIPPER_JOINT)

##
# Feetech STS3215 servo model.
##
# Position servo, emulated as a MuJoCo <position> actuator (PD with saturated output).
#
# kp / kv: Menagerie robotstudio_so101 defaults (998.22 N*m/rad, 2.731 N*m*s/rad).
#   Menagerie derived them from LeRobot's servo P gain of 16 following
#   github.com/Gregory119/RBE501-RL-arm-project (gymnasium_env/README.md). LeRobot's
#   SO follower writes P=16, I=0, D=32 to every STS3215 at connect time
#   (lerobot/src/lerobot/robots/so_follower/config_so_follower.py, checked 2026-10-09).
# Torque limit: 2.94 N*m = 30 kg*cm, the 12 V STS3215 stall torque (Menagerie default
#   and TheRobotStudio README). The SO-101 BOM ships the 7.4 V C001 variant: the README
#   gives 16.5 kg*cm @ 6 V (1.62 N*m); ~19.5 kg*cm @ 7.4 V (1.91 N*m) is the Feetech
#   datasheet figure as remembered, NOT verified here. Training randomises the limit
#   down to 0.55x (1.62 N*m) so a policy cannot rely on the 12 V torque.
# Armature / damping / frictionloss: Menagerie ``sts3215`` default class (reflected
#   rotor inertia 0.028 kg*m^2, viscous 0.60 N*m*s/rad, dry friction 0.052 N*m).
# Gripper: LeRobot caps the gripper servo at Max_Torque_Limit=500 (50 %) and
#   Protection_Current=250 "to avoid burnout" (so_follower.py configure(), checked
#   2026-10-09), so the gripper limit is half the arm limit.
STS3215_KP = 998.22
STS3215_KV = 2.731
STS3215_TORQUE_12V = 2.94
STS3215_TORQUE_7V4_6V = 1.62  # 16.5 kg*cm @ 6 V (TheRobotStudio README)
STS3215_ARMATURE = 0.028
STS3215_DAMPING = 0.60
STS3215_FRICTIONLOSS = 0.052
STS3215_VELOCITY_LIMIT = 4.7  # rad/s, ~0.222 s/60 deg no-load; informational only.
GRIPPER_TORQUE_FRACTION = 0.5

ARM_EFFORT_LIMIT = STS3215_TORQUE_12V
GRIPPER_EFFORT_LIMIT = STS3215_TORQUE_12V * GRIPPER_TORQUE_FRACTION

##
# Gripper geometry (gripper-body frame, metres). Measured with
# scripts/measure_so101.py; see measurements/so101_measurements.json.
##
# Fingers point along -z of the ``gripper`` body; the moving jaw closes along -x onto
# the fixed jaw. The fixed jaw's innermost collision surface (tip spheres / pad 4,
# probed with mj_geomDistance) is at x ~= -0.0075 for z in [-0.104, -0.090]; its lowest
# point is the convex hull of the part0 collision mesh at z = -0.1072.
# The grasp site is where the centre of a nominal 30 mm object sits when it is
# squeezed against the fixed jaw, 15.2 mm above the lowest fixed-jaw point.
FIXED_JAW_INNER_X = -0.0075
FIXED_JAW_LOWEST_Z = -0.1072
NOMINAL_OBJECT_WIDTH = 0.030
GRASP_SITE_POS = (FIXED_JAW_INNER_X + NOMINAL_OBJECT_WIDTH / 2, 0.0, -0.092)
JAW_TIP_BELOW_SITE = GRASP_SITE_POS[2] - FIXED_JAW_LOWEST_Z  # 0.0152 m
# Gripper joint angles (rad). Joint range is [-0.1745, 1.7453]; the jaw pads touch at
# the lower limit. 1.0 rad gives an ~85 mm mouth at the moving-jaw tip.
GRIPPER_CLOSED = -0.17
GRIPPER_OPEN = 1.0

##
# Collision bitmasks.
##
ROBOT_CONTYPE = 2
ROBOT_CONAFFINITY = 3
WORLD_CONTYPE = 1
WORLD_CONAFFINITY = 1

FINGER_GEOM_REGEX = r"(fixed|moving)_jaw_.*"


def _name_collision_geoms(spec: mujoco.MjSpec) -> list[mujoco.MjsGeom]:
  """Give every colliding robot geom a stable name; return the collision geoms."""
  collision: list[mujoco.MjsGeom] = []
  counters: dict[str, int] = {}
  for geom in spec.geoms:
    # Visual geoms in Menagerie have contype=conaffinity=0 (class "visual").
    if geom.contype == 0 and geom.conaffinity == 0:
      continue
    collision.append(geom)
    if geom.name:
      continue
    body = geom.parent.name
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
      prefix = "fixed_jaw_mesh" if body == "gripper" else "moving_jaw_mesh"
    else:
      prefix = f"{body}_collision"
    counters[prefix] = counters.get(prefix, 0) + 1
    geom.name = f"{prefix}{counters[prefix]}"
  return collision


def so101_spec() -> mujoco.MjSpec:
  """Load the Menagerie SO-101 and apply the load-time edits listed in the docstring."""
  spec = mujoco.MjSpec.from_file(str(SO101_XML))

  # 1. Remove XML actuators (re-added by mjlab or add_position_actuators()).
  for act in list(spec.actuators):
    spec.delete(act)

  # 2./3. Sites.
  gripper = spec.body("gripper")
  gripper.add_site(
    name="grasp_site",
    pos=GRASP_SITE_POS,
    size=(0.004, 0.004, 0.004),
    rgba=(0.1, 0.9, 0.1, 0.6),
    group=4,
  )
  gripper.add_site(
    name="fixed_tip_site",
    pos=(-0.0085, 0.0, -0.104),
    size=(0.003, 0.003, 0.003),
    group=4,
  )
  spec.body("moving_jaw_so101_v1").add_site(
    name="moving_tip_site",
    pos=(-0.012, -0.078, 0.0192),
    size=(0.003, 0.003, 0.003),
    group=4,
  )

  # 4./5. Collision geoms.
  for geom in _name_collision_geoms(spec):
    geom.contype = ROBOT_CONTYPE
    geom.conaffinity = ROBOT_CONAFFINITY
  return spec


def add_position_actuators(
  spec: mujoco.MjSpec,
  arm_effort_limit: float = ARM_EFFORT_LIMIT,
  gripper_effort_limit: float = GRIPPER_EFFORT_LIMIT,
  kp_scale: float = 1.0,
  prefix: str = "",
) -> None:
  """Add STS3215 position actuators exactly as mjlab's create_position_actuator does.

  Only for pure-MuJoCo use (tests, measurement scripts). mjlab builds its own from
  so101_constants.ARTICULATION with the same numbers.
  """
  for name in ALL_JOINTS:
    effort = gripper_effort_limit if name == GRIPPER_JOINT else arm_effort_limit
    joint = spec.joint(prefix + name)
    joint.armature = STS3215_ARMATURE
    joint.damping[0] = STS3215_DAMPING
    joint.frictionloss = STS3215_FRICTIONLOSS
    act = spec.add_actuator(name=prefix + name, target=prefix + name)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.dyntype = mujoco.mjtDyn.mjDYN_NONE
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0] = STS3215_KP * kp_scale
    act.biasprm[1] = -STS3215_KP * kp_scale
    act.biasprm[2] = -STS3215_KV
    act.inheritrange = 0.0
    act.ctrllimited = False
    act.forcelimited = True
    act.forcerange[:] = np.array([-effort, effort])


def collision_geom_names(model: mujoco.MjModel, body_prefix: str = "") -> list[str]:
  out = []
  for g in range(model.ngeom):
    if model.geom_contype[g] == ROBOT_CONTYPE:
      out.append(model.geom(g).name)
  return out
