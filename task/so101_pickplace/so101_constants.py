"""mjlab robot config for the LeRobot / TheRobotStudio SO-101 (5-DOF arm + jaw).

Pattern follows mjlab's ``asset_zoo/robots/i2rt_yam/yam_constants.py``. The physical
model (MJCF edits, servo numbers, collision masks) lives in :mod:`robot_model` so the
pure-MuJoCo grasp test checks the same arm; this file only wraps it for mjlab.
"""

from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from . import robot_model as rm

# Command (action) delay: 0-3 physics steps of 5 ms, i.e. up to 15 ms of bus latency
# (STS3215 on a 1 Mbit/s half-duplex bus; LeRobot reads then writes 6 servos/step).
ACTION_DELAY_MAX_PHYSICS_STEPS = 3

_COMMON = dict(
  stiffness=rm.STS3215_KP,
  damping=rm.STS3215_KV,
  armature=rm.STS3215_ARMATURE,
  frictionloss=rm.STS3215_FRICTIONLOSS,
  viscous_damping=rm.STS3215_DAMPING,
  delay_min_lag=0,
  delay_max_lag=ACTION_DELAY_MAX_PHYSICS_STEPS,
  delay_hold_prob=0.5,
)

ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=rm.ARM_JOINTS,
  effort_limit=rm.ARM_EFFORT_LIMIT,
  **_COMMON,
)
GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(rm.GRIPPER_JOINT,),
  effort_limit=rm.GRIPPER_EFFORT_LIMIT,
  **_COMMON,
)

# Home: grasp site ~0.17 m in front of the pan axis, 0.12 m above the table, fingers
# tilted 20 deg forward of straight down (keeps wrist_flex 0.2 rad inside its limit),
# jaws ~54 mm open, closing along the base y axis.
HOME_JOINT_POS: dict[str, float] = {
  "shoulder_pan": 0.0,
  "shoulder_lift": -0.6,
  "elbow_flex": 0.4,
  "wrist_flex": 1.45,
  "wrist_roll": -1.55,
  "gripper": 0.6,
}


def home_keyframe(mount_height: float = 0.0) -> EntityCfg.InitialStateCfg:
  return EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, mount_height),
    joint_pos=dict(HOME_JOINT_POS),
    joint_vel={".*": 0.0},
  )


ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_ACTUATOR, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.95,
)

# Action scale (rad per unit action, offset = HOME). The YAM rule
# 0.25 * effort / stiffness gives 0.7 mrad for these stiff servos, so it is set by
# hand: +-1 covers ~+-0.6 rad around home on the big joints, and the gripper spans
# fully closed (-0.17) to open (1.0) for actions in about [-0.9, +0.45].
SO101_ACTION_SCALE: dict[str, float] = {
  "shoulder_pan": 0.6,
  "shoulder_lift": 0.6,
  "elbow_flex": 0.6,
  "wrist_flex": 0.6,
  "wrist_roll": 0.8,
  "gripper": 0.9,
}


def get_so101_robot_cfg(mount_height: float = 0.0) -> EntityCfg:
  from .mj_scene import add_riser

  def spec_fn():
    spec = rm.so101_spec()
    add_riser(spec, mount_height)
    return spec

  return EntityCfg(
    init_state=home_keyframe(mount_height),
    spec_fn=spec_fn,
    articulation=ARTICULATION,
  )
