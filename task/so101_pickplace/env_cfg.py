"""SO101-PickPlace-State: SO-101 picks an object off the table and drops it in a bin.

Ported from mjlab's lift-cube task (tasks/manipulation/lift_cube_env_cfg.py +
config/yam) and extended to pick-and-place with an object-centric reward.

Control: 10 Hz policy (decimation 20 x 5 ms physics), target-relative delta joint
actions capped at 0.5 rad/s (arm) / 2.0 rad/s (jaw), 0-1 step action latency.
Observations: actor sees only what the real loop has (noisy joints, own targets,
tracked object xy with dropouts, bin xy, clock); critic is privileged.
Reward: object follows the warped demo object trajectory + reach / grasp / lift /
carry / place / release shaping; no hand-path term.
DR: Squint-style (proprio noise 2->5 deg, PD +-50 %, gripper kp 500-2000, object size
+-3 mm, frictions, mass, torque limits, encoder bias, latency).
"""

from __future__ import annotations

import math

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manip_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp
from . import mj_scene
from . import robot_model as rm
from .so101_constants import get_so101_robot_cfg
from .task_spec import TaskSpec, load_task_spec

COMMAND = "pick_place"
CONTROL_HZ = 10.0
DECIMATION = int(round(1.0 / (CONTROL_HZ * mj_scene.TIMESTEP)))  # 20
ARM_MAX_SPEED = 0.5  # rad/s
GRIPPER_MAX_SPEED = 2.0  # rad/s
FINGER_GEOMS = r"(fixed|moving)_jaw_.*"


def _default_trajectory(spec: TaskSpec) -> tuple[list[float], list[float], float, float]:
  """Synthetic rest-lift-carry-lower-rest reference when the spec has none."""
  import numpy as np

  o, p = spec.object, spec.place
  start = np.array([(o.x.lo + o.x.hi) / 2, (o.y.lo + o.y.hi) / 2, o.half_height])
  end = np.array([(p.x.lo + p.x.hi) / 2, (p.y.lo + p.y.hi) / 2,
                  mj_scene.bin_floor_top() + o.half_height])
  up = 0.09
  keys = [(0.0, start), (3.5, start), (5.0, start + [0, 0, up - start[2]]),
          (8.0, np.array([end[0], end[1], up])), (9.0, end), (11.0, end)]
  t = np.arange(0.0, 11.0 + 1e-9, 0.1)
  kt = np.array([k[0] for k in keys])
  kp = np.stack([k[1] for k in keys])
  pos = np.stack([np.interp(t, kt, kp[:, i]) for i in range(3)], axis=1)
  return t.tolist(), pos.reshape(-1).tolist(), 3.5, 9.5


def so101_pickplace_env_cfg(
  spec: TaskSpec | str | None = None,
  play: bool = False,
  observe_object_yaw: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  """``observe_object_yaw``: None = on for a box (AprilTag on the cube gives yaw), off
  for a cylinder (yaw-symmetric; a colour blob gives xy only)."""
  if spec is None or isinstance(spec, str):
    spec = load_task_spec(spec)
  o, p = spec.object, spec.place
  if observe_object_yaw is None:
    observe_object_yaw = o.kind == "box"

  if spec.trajectory is not None:
    t, pos, g, r = spec.trajectory.reference()
    ref_t, ref_pos = t.tolist(), pos.reshape(-1).tolist()
    track_w, track_std = spec.trajectory.weight, spec.trajectory.std
    slack = spec.trajectory.time_slack_s
  else:
    ref_t, ref_pos, g, r = _default_trajectory(spec)
    track_w, track_std, slack = 4.0, 0.05, 1.0
  ref_end = ref_t[-1]

  robot = get_so101_robot_cfg(spec.mount_height)
  obj = EntityCfg(spec_fn=lambda: mj_scene.object_spec(
    kind=o.kind, size=o.size, height=o.height, mass=o.mass))
  bin_ = EntityCfg(spec_fn=mj_scene.bin_spec)

  # ---------------------------------------------------------------- observations
  robot_joints = SceneEntityCfg("robot", joint_names=rm.ALL_JOINTS)
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_noisy,
      params={"std_start_deg": 2.0, "std_end_deg": 5.0, "ramp_steps": 60_000,
              "asset_cfg": robot_joints},
    ),
    "joint_target": ObservationTermCfg(func=mdp.joint_target_rel,
                                       params={"action_name": "joint_pos"}),
    "object_xy": ObservationTermCfg(
      func=mdp.object_xy_measured,
      params={"command_name": COMMAND, "std": 0.01, "dropout_prob": 0.05},
    ),
    "target_xy": ObservationTermCfg(
      func=mdp.target_xy, params={"command_name": COMMAND},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "clock": ObservationTermCfg(func=mdp.episode_clock, params={"command_name": COMMAND}),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  if observe_object_yaw:
    actor_terms["object_yaw"] = ObservationTermCfg(
      func=mdp.object_yaw_measured,
      params={"command_name": COMMAND, "std_rad": 0.1,
              "symmetry": 4 if o.kind == "box" else 1},
    )
  critic_terms = {
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_true,
                                    params={"asset_cfg": robot_joints}),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel,
                                    params={"asset_cfg": robot_joints}),
    "joint_target": ObservationTermCfg(func=mdp.joint_target_rel,
                                       params={"action_name": "joint_pos"}),
    "object_pos": ObservationTermCfg(func=mdp.object_pos_base,
                                     params={"command_name": COMMAND}),
    "object_quat": ObservationTermCfg(func=mdp.object_quat, params={"command_name": COMMAND}),
    "object_linvel": ObservationTermCfg(func=mdp.object_linvel,
                                        params={"command_name": COMMAND}),
    "object_rel_ee": ObservationTermCfg(func=mdp.object_pos_rel_ee,
                                        params={"command_name": COMMAND}),
    "finger_axis": ObservationTermCfg(func=mdp.finger_axis, params={"command_name": COMMAND}),
    "jaw_contacts": ObservationTermCfg(func=mdp.jaw_contacts, params={"command_name": COMMAND}),
    "object_extent": ObservationTermCfg(func=mdp.object_extent,
                                        params={"command_name": COMMAND}),
    "ref_rel_object": ObservationTermCfg(func=mdp.ref_point_rel_object,
                                         params={"command_name": COMMAND}),
    "target_xy": ObservationTermCfg(func=mdp.target_xy, params={"command_name": COMMAND}),
    "clock": ObservationTermCfg(func=mdp.episode_clock, params={"command_name": COMMAND}),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  # ---------------------------------------------------------------- actions
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.TargetDeltaJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      max_speed=ARM_MAX_SPEED,
      gripper_max_speed=GRIPPER_MAX_SPEED,
      latency_prob=0.5,
    )
  }

  # ---------------------------------------------------------------- command
  commands: dict[str, CommandTermCfg] = {
    COMMAND: mdp.PickPlaceCommandCfg(
      self_name=COMMAND,
      resampling_time_range=(1.0e9, 1.0e9),  # once per episode (at reset)
      debug_vis=True,
      object_kind=o.kind,
      object_half_height=o.half_height,
      object_half_width=o.half_width,
      object_x=o.x.as_tuple(), object_y=o.y.as_tuple(), object_yaw=o.yaw.as_tuple(),
      target_x=p.x.as_tuple(), target_y=p.y.as_tuple(),
      min_separation=p.min_separation,
      success_xy_margin=p.success_xy_margin,
      mount_height=spec.mount_height,
      bin_inner_half=mj_scene.BIN_INNER_HALF,
      bin_floor_top=mj_scene.bin_floor_top(),
      lift_height=0.03,
      ref_t=tuple(ref_t), ref_pos=tuple(ref_pos),
      ref_grasp_t=g, ref_release_t=r, ref_time_slack=slack,
    )
  }

  # ---------------------------------------------------------------- events / DR
  finger_cfg = SceneEntityCfg("robot", geom_names=FINGER_GEOMS)
  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform, mode="reset",
      params={"pose_range": {}, "velocity_range": {},
              "asset_cfg": SceneEntityCfg("robot")},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_around_home, mode="reset",
      params={"position_range": (-0.08, 0.08),
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "object_size": EventTermCfg(
      func=mdp.object_size_abs, mode="reset",
      params={"delta_range": (-0.003, 0.003), "asset_cfg": SceneEntityCfg("object")},
    ),
    "object_mass": EventTermCfg(
      func=dr.pseudo_inertia, mode="startup",
      params={"alpha_range": (-0.25, 0.25),  # mass x [0.61, 1.65]
              "asset_cfg": SceneEntityCfg("object", body_names=("object",))},
    ),
    "object_friction": EventTermCfg(
      func=dr.geom_friction, mode="startup",
      params={"asset_cfg": SceneEntityCfg("object", geom_names=(".*",)),
              "operation": "abs", "distribution": "uniform", "axes": [0],
              "ranges": (0.4, 1.0)},
    ),
    "fingertip_friction_slide": EventTermCfg(
      func=dr.geom_friction, mode="startup",
      params={"asset_cfg": finger_cfg, "operation": "abs", "distribution": "uniform",
              "axes": [0], "ranges": (0.4, 1.4)},
    ),
    "fingertip_friction_spin": EventTermCfg(
      func=dr.geom_friction, mode="startup",
      params={"asset_cfg": finger_cfg, "operation": "abs",
              "distribution": "log_uniform", "axes": [1], "ranges": (1e-4, 2e-2)},
    ),
    "fingertip_friction_roll": EventTermCfg(
      func=dr.geom_friction, mode="startup",
      params={"asset_cfg": finger_cfg, "operation": "abs",
              "distribution": "log_uniform", "axes": [2], "ranges": (1e-5, 5e-3)},
    ),
    "servo_params": EventTermCfg(
      func=mdp.servo_params, mode="startup",
      params={
        "arm_kp_scale": (0.5, 1.5), "arm_kd_scale": (0.5, 1.5),
        # Squint: gripper stiffness 500-2000 (same N*m/rad units as MuJoCo kp; nominal
        # 998). Its damping 50-200 is a PhysX drive value ~36x our calibrated kv, so the
        # same x0.5-x2 spread is applied around the MuJoCo nominal instead.
        "gripper_kp_abs": (500.0, 2000.0), "gripper_kd_scale": (0.5, 2.0),
        # 0.55 x 2.94 = 1.62 N*m: the 7.4 V STS3215 the SO-101 BOM ships.
        "arm_effort_scale": (0.55, 1.0), "gripper_effort_scale": (0.55, 1.0),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias, mode="startup",
      params={"bias_range": (-0.02, 0.02), "asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  # ---------------------------------------------------------------- sensors
  fixed_jaw = ContactSensorCfg(
    name="fixed_jaw_object_contact",
    primary=ContactMatch(mode="body", pattern="gripper", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="object", entity="object"),
    fields=("found", "force"), reduce="maxforce", num_slots=1,
  )
  moving_jaw = ContactSensorCfg(
    name="moving_jaw_object_contact",
    primary=ContactMatch(mode="body", pattern="moving_jaw_so101_v1", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="object", entity="object"),
    fields=("found", "force"), reduce="maxforce", num_slots=1,
  )
  ee_table = ContactSensorCfg(
    name="ee_table_collision",
    primary=ContactMatch(mode="subtree", pattern="gripper", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=DECIMATION,
  )

  # ---------------------------------------------------------------- rewards
  rewards = {
    "object_track": RewardTermCfg(func=mdp.object_track, weight=track_w,
                                  params={"command_name": COMMAND, "std": track_std}),
    "reach": RewardTermCfg(func=mdp.reach, weight=1.0,
                           params={"command_name": COMMAND, "std": 0.08}),
    "grasp": RewardTermCfg(func=mdp.grasp, weight=1.0, params={"command_name": COMMAND}),
    "lift": RewardTermCfg(func=mdp.lift, weight=2.0,
                          params={"command_name": COMMAND, "height": 0.05}),
    "carry": RewardTermCfg(func=mdp.carry_to_target, weight=2.0,
                           params={"command_name": COMMAND, "std": 0.08}),
    "placed": RewardTermCfg(func=mdp.placed, weight=4.0, params={"command_name": COMMAND}),
    "released": RewardTermCfg(func=mdp.released, weight=6.0,
                              params={"command_name": COMMAND}),
    "approach_side": RewardTermCfg(
      func=mdp.approach_side, weight=spec.approach.weight,
      params={"command_name": COMMAND, "direction": spec.approach.direction,
              "gate_std": spec.approach.gate_std},
    ),
    "object_dropped": RewardTermCfg(func=mdp.object_dropped, weight=-2.0,
                                    params={"command_name": COMMAND}),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "ee_table_collision": TerminationTermCfg(
      func=manip_mdp.illegal_contact,
      params={"sensor_name": "ee_table_collision", "force_threshold": 15.0},
    ),
    "object_out_of_bounds": TerminationTermCfg(
      func=mdp.object_out_of_bounds, params={"command_name": COMMAND},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
      entities={"robot": robot, "object": obj, "bin": bin_},
      sensors=(fixed_jaw, moving_jaw, ee_table),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=0.9,
      elevation=-25.0,
      azimuth=150.0,
    ),
    sim=SimulationCfg(
      nconmax=100,
      njmax=800,
      mujoco=MujocoCfg(
        timestep=mj_scene.TIMESTEP,
        iterations=mj_scene.SOLVER_ITERATIONS,
        ls_iterations=mj_scene.SOLVER_LS_ITERATIONS,
        impratio=mj_scene.IMPRATIO,
        cone="elliptic",
      ),
    ),
    decimation=DECIMATION,
    # 15 s minimum: the scripted pick-and-place through the env at the 0.5 rad/s cap
    # needed 11-20 s (measurements/smoke_cpu.json); the reference is 11 s.
    episode_length_s=float(max(15.0, math.ceil(ref_end + 3.0))),
  )

  if play:
    # Same episode length (the object reference is time indexed, so episodes must
    # restart together with the command); noise and dropouts off.
    cfg.observations["actor"].enable_corruption = False
    a = cfg.observations["actor"].terms
    a["joint_pos"].params.update({"std_start_deg": 0.0, "std_end_deg": 0.0})
    a["object_xy"].params.update({"std": 0.0, "dropout_prob": 0.0})
    cfg.curriculum = {}
  return cfg
