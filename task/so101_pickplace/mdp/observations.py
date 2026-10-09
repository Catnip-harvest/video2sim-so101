"""Observation terms.

Actor terms are restricted to what the real deploy loop has:
  * joint positions from the servos (noisy; sigma ramps 2 -> 5 deg, Squint-style),
  * the joint target the loop itself commanded (exact),
  * object xy on the table from an AprilTag / colour blob + table homography
    (sigma ~1 cm, occasional dropouts that hold the last value),
  * optionally object yaw (AprilTag only),
  * the bin xy, the episode clock (the object reference is time indexed), last action.
Critic terms add privileged simulator state.
All positions are in the robot base frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import yaw_quat

from . import state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _ramp(env: ManagerBasedRlEnv, start: float, end: float, ramp_steps: int) -> float:
  if ramp_steps <= 0:
    return end
  a = min(1.0, max(0.0, env.common_step_counter / ramp_steps))
  return start + a * (end - start)


def joint_pos_noisy(
  env: ManagerBasedRlEnv,
  std_start_deg: float = 2.0,
  std_end_deg: float = 5.0,
  ramp_steps: int = 0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Encoder-frame joint positions minus home, + Gaussian noise whose sigma ramps
  linearly from std_start to std_end over ``ramp_steps`` env steps (curriculum)."""
  robot = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  q = robot.data.joint_pos_biased[:, ids] - robot.data.default_joint_pos[:, ids]
  std = math.radians(_ramp(env, std_start_deg, std_end_deg, ramp_steps))
  if std > 0:
    q = q + std * torch.randn_like(q)
  return q


def joint_pos_true(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  return robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]


def joint_target_rel(env: ManagerBasedRlEnv, action_name: str = "joint_pos",
                     asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  term = env.action_manager.get_term(action_name)
  robot = env.scene[asset_cfg.name]
  return term.target - robot.data.default_joint_pos[:, term.target_ids]


def _to_base_xy(cmd, p_w: torch.Tensor) -> torch.Tensor:
  return p_w[:, :2] - cmd.base_pos_w[:, :2]


def object_xy_measured(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.01,
  dropout_prob: float = 0.05,
) -> torch.Tensor:
  """Object centre xy in the base frame as a table-camera tracker would report it:
  Gaussian noise ``std`` (m) and, with probability ``dropout_prob`` per step, no new
  detection (the previous measurement is held). The buffer lives on the command and is
  re-initialised to the true value at reset."""
  cmd = state.get_command(env, command_name)
  true_xy = _to_base_xy(cmd, cmd.object.data.root_link_pos_w)
  fresh = true_xy + std * torch.randn_like(true_xy) if std > 0 else true_xy
  keep_old = torch.rand(true_xy.shape[0], device=env.device) < dropout_prob
  fresh = torch.where(keep_old[:, None], cmd.object_xy_meas, fresh)
  cmd.object_xy_meas[:] = fresh
  cmd.last_detection_dropout = keep_old  # reused by object_yaw_measured (same tag)
  return fresh


def object_yaw_measured(
  env: ManagerBasedRlEnv,
  command_name: str,
  std_rad: float = 0.1,
  symmetry: int = 4,
) -> torch.Tensor:
  """(cos, sin) of symmetry * yaw (a box looks the same every 90 deg), as an AprilTag
  on the object would report it: noisy, and held whenever the xy detection dropped out
  this step (same tag, same miss)."""
  cmd = state.get_command(env, command_name)
  q = yaw_quat(cmd.object.data.root_link_quat_w)
  yaw = 2.0 * torch.atan2(q[:, 3], q[:, 0])
  if std_rad > 0:
    yaw = yaw + std_rad * torch.randn_like(yaw)
  miss = getattr(cmd, "last_detection_dropout", None)
  if miss is not None:
    yaw = torch.where(miss, cmd.object_yaw_meas, yaw)
  cmd.object_yaw_meas[:] = yaw
  return torch.stack([torch.cos(symmetry * yaw), torch.sin(symmetry * yaw)], dim=-1)


def target_xy(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = state.get_command(env, command_name)
  return _to_base_xy(cmd, cmd.target_pos_w)


def episode_clock(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Episode time divided by the reference duration (1.0 = reference finished)."""
  cmd = state.get_command(env, command_name)
  return (cmd.episode_time() / max(cmd.ref_duration, 1e-6)).unsqueeze(-1)


# ---------------------------------------------------------------- privileged (critic)


def object_pos_rel_ee(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  s = state.compute(env, command_name, phase="obs")
  return s.obj_pos_w - s.ee_pos_w


def object_pos_base(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = state.get_command(env, command_name)
  p = cmd.object.data.root_link_pos_w - cmd.base_pos_w
  p[:, 2] = cmd.object.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return p


def object_quat(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  q = state.get_command(env, command_name).object.data.root_link_quat_w
  return torch.where(q[:, :1] < 0, -q, q)


def object_linvel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return state.get_command(env, command_name).object.data.root_link_lin_vel_w


def jaw_contacts(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  s = state.compute(env, command_name, phase="obs")
  return torch.stack([s.fixed_contact.float(), s.moving_contact.float()], dim=-1)


def object_extent(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = state.get_command(env, command_name)
  return torch.stack([cmd.object_half_height, cmd.object_half_width], dim=-1)


def ref_point_rel_object(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  s = state.compute(env, command_name, phase="obs")
  return s.ref_point_w - s.obj_pos_w


def finger_axis(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return state.compute(env, command_name, phase="obs").finger_axis_w


__all__ = [
  "joint_pos_noisy", "joint_pos_true", "joint_target_rel", "object_xy_measured",
  "object_yaw_measured", "target_xy", "episode_clock", "object_pos_rel_ee",
  "object_pos_base", "object_quat", "object_linvel", "jaw_contacts", "object_extent",
  "ref_point_rel_object", "finger_axis",
]
