"""Domain-randomisation events specific to this task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def object_size_abs(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  delta_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Change the object's primitive size by an absolute amount per env.

  ``delta_range`` is applied to the full extents (edge length / diameter / height),
  i.e. half sizes move by delta / 2. Box: one draw for all three edges (stays a cube).
  Cylinder: independent draws for diameter and height. Broadphase bounds are
  recomputed. The command term reads the new size at the same reset to place the
  object on the table.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  asset = env.scene[asset_cfg.name]
  gid = asset.indexing.geom_ids[0]
  default = env.sim.get_default_field("geom_size")[gid]  # (3,)
  n = len(env_ids)
  lo, hi = delta_range
  d1 = torch.empty(n, device=env.device).uniform_(lo, hi) / 2
  d2 = torch.empty(n, device=env.device).uniform_(lo, hi) / 2
  size = default[None, :].repeat(n, 1)
  import mujoco

  gtype = int(torch.as_tensor(env.sim.model.geom_type, device=env.device)[gid])
  if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
    size = size + d1[:, None]
  elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
    size[:, 0] = size[:, 0] + d1
    size[:, 1] = size[:, 1] + d2
  else:
    raise ValueError("object_size_abs supports box and cylinder objects")
  env.sim.model.geom_size[env_ids, gid] = size
  _recompute_geom_bounds(env, env_ids, asset_cfg)


@requires_model_fields("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
def servo_params(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  arm_kp_scale: tuple[float, float],
  arm_kd_scale: tuple[float, float],
  gripper_kp_abs: tuple[float, float],
  gripper_kd_scale: tuple[float, float],
  arm_effort_scale: tuple[float, float],
  gripper_effort_scale: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  gripper_joint: str = "gripper",
) -> None:
  """Randomise the position-servo PD gains and torque limits (Squint-style).

  Arm: kp and kd scaled independently per joint; gripper: kp drawn as an absolute
  value (N*m/rad), kd scaled. Effort limits scaled per joint (covers the 7.4 V vs 12 V
  STS3215). Written straight to the <position> actuator parameters: gainprm[0]=kp,
  biasprm[1]=-kp, biasprm[2]=-kd, forcerange=+-limit. mjlab's dr.pd_gains selects
  actuators by group index while SceneEntityCfg resolves per-joint actuator names, so
  this task does its own selection by target joint name.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  robot = env.scene[asset_cfg.name]
  m = env.sim.model
  g0 = env.sim.get_default_field("actuator_gainprm")
  b0 = env.sim.get_default_field("actuator_biasprm")
  f0 = env.sim.get_default_field("actuator_forcerange")
  n = len(env_ids)

  def u(rng, k):
    return torch.empty(n, k, device=env.device).uniform_(rng[0], rng[1])

  for act in robot.actuators:
    ids = act.global_ctrl_ids
    k = len(ids)
    is_gripper = gripper_joint in act.target_names
    e = env_ids[:, None]
    if is_gripper:
      kp = u(gripper_kp_abs, k)
      kd = b0[ids, 2] * u(gripper_kd_scale, k)  # b0[...,2] = -kd
      eff = u(gripper_effort_scale, k)
    else:
      kp = g0[ids, 0] * u(arm_kp_scale, k)
      kd = b0[ids, 2] * u(arm_kd_scale, k)
      eff = u(arm_effort_scale, k)
    m.actuator_gainprm[e, ids, 0] = kp
    m.actuator_biasprm[e, ids, 1] = -kp
    m.actuator_biasprm[e, ids, 2] = kd
    m.actuator_forcerange[e, ids, 0] = f0[ids, 0] * eff
    m.actuator_forcerange[e, ids, 1] = f0[ids, 1] * eff


def reset_joints_around_home(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Same as mjlab's reset_joints_by_offset (home + uniform offset, clamped to the soft
  limits, zero velocity), but robust to un-batched joint-limit tensors.

  On Warp's CPU device mjlab's TorchArray does not broadcast un-randomised model
  fields to nworld rows (numpy reports a non-zero stride on the size-1 world axis), so
  ``soft_joint_pos_limits`` is (1, J, 2) and ``limits[env_ids]`` raises for env_ids > 0.
  On CUDA / ROCm it is (nworld, J, 2) and both versions behave identically.
  """
  from mjlab.utils.lab_api.math import sample_uniform

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  q = asset.data.default_joint_pos[env_ids][:, ids].clone()
  q += sample_uniform(*position_range, q.shape, env.device)
  lim = asset.data.soft_joint_pos_limits
  lim = lim[env_ids] if lim.shape[0] > 1 else lim.expand(len(env_ids), -1, -1)
  lim = lim[:, ids]
  q = torch.maximum(torch.minimum(q, lim[..., 1]), lim[..., 0])
  joint_ids = ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids, joint_ids=joint_ids)
