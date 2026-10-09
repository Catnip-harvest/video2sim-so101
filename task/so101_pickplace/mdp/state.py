"""Per-step task state shared by rewards, terminations, observations and metrics.

Computed from simulator state only (contact sensors, body poses); nothing here writes
state. Cached per (phase, env step) on the command term so the dozen reward terms do
not recompute it; the cache is cleared whenever the command resamples (= env reset).

"phase" separates the two points in ``ManagerBasedRlEnv.step`` where the state is
read: "rt" (terminations + rewards, before resets) and "obs" (metrics + observations,
after resets and the forward call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .commands import PickPlaceCommand

GRASP_SITE = "grasp_site"
FIXED_SENSOR = "fixed_jaw_object_contact"
MOVING_SENSOR = "moving_jaw_object_contact"
REST_SPEED = 0.05  # m/s: "at rest" threshold for the placed test


@dataclass
class TaskState:
  t: torch.Tensor  # (B,) episode time, s
  ee_pos_w: torch.Tensor  # (B, 3) grasp site
  finger_axis_w: torch.Tensor  # (B, 3) unit, direction the fingers point
  obj_pos_w: torch.Tensor
  obj_quat_w: torch.Tensor
  obj_linvel_w: torch.Tensor
  object_bottom: torch.Tensor  # (B,) object centre height above table - half height
  ee_to_object: torch.Tensor  # (B,) |ee - obj|
  object_to_target_xy: torch.Tensor  # (B,)
  fixed_contact: torch.Tensor  # (B,) bool
  moving_contact: torch.Tensor
  grasped: torch.Tensor  # both jaws touch the object
  lifted: torch.Tensor  # grasped and object bottom >= lift_height
  in_bin_xy: torch.Tensor
  placed: torch.Tensor  # inside the bin footprint, resting on the floor
  released: torch.Tensor  # placed and no jaw contact
  ref_point_w: torch.Tensor  # (B, 3) reference object position at time t
  ref_error: torch.Tensor  # (B,) min distance to the reference within +-slack


def _found(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  data = env.scene[name].data
  assert data.found is not None
  return (data.found > 0).any(dim=-1)


def get_command(env: ManagerBasedRlEnv, command_name: str) -> PickPlaceCommand:
  return env.command_manager.get_term(command_name)  # type: ignore[return-value]


def compute(env: ManagerBasedRlEnv, command_name: str, phase: str = "rt") -> TaskState:
  cmd = get_command(env, command_name)
  key = (phase, int(env.common_step_counter))
  cached = cmd.state_cache.get(key)
  if cached is not None:
    return cached
  if len(cmd.state_cache) > 4:
    cmd.state_cache.clear()

  c = cmd.cfg
  robot = cmd.robot
  obj = cmd.object
  site_id = robot.find_sites((GRASP_SITE,))[0][0]
  ee_pos = robot.data.site_pos_w[:, site_id]
  ee_quat = robot.data.site_quat_w[:, site_id]
  down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(ee_pos)
  finger_axis = quat_apply(ee_quat, down)  # gripper -z

  obj_pos = obj.data.root_link_pos_w
  obj_quat = obj.data.root_link_quat_w
  obj_vel = obj.data.root_link_lin_vel_w
  table_z = env.scene.env_origins[:, 2]
  half_h = cmd.object_half_height
  half_w = cmd.object_half_width
  bottom = obj_pos[:, 2] - table_z - half_h

  fixed = _found(env, FIXED_SENSOR)
  moving = _found(env, MOVING_SENSOR)
  grasped = fixed & moving
  lifted = grasped & (bottom >= c.lift_height)

  d_xy = obj_pos[:, :2] - cmd.target_pos_w[:, :2]
  limit = c.bin_inner_half - half_w - c.success_xy_margin
  in_bin = (d_xy.abs() <= limit[:, None]).all(dim=-1)
  resting = obj_vel.norm(dim=-1) < REST_SPEED
  on_floor = bottom <= c.bin_floor_top + 0.01
  placed = in_bin & on_floor & resting
  released = placed & ~fixed & ~moving

  t = cmd.episode_time()
  k = cmd.ref_index(t)
  idx = torch.arange(env.num_envs, device=env.device)
  ref_point = cmd.ref_w[idx, k]
  w = int(round(c.ref_time_slack / cmd.ref_dt))
  offs = torch.arange(-w, w + 1, device=env.device)
  kk = (k[:, None] + offs[None, :]).clamp(0, cmd.ref_t.numel() - 1)
  window = cmd.ref_w[idx[:, None], kk]  # (B, W, 3)
  ref_err = (window - obj_pos[:, None, :]).norm(dim=-1).min(dim=-1).values

  s = TaskState(
    t=t, ee_pos_w=ee_pos, finger_axis_w=finger_axis, obj_pos_w=obj_pos,
    obj_quat_w=obj_quat, obj_linvel_w=obj_vel, object_bottom=bottom,
    ee_to_object=(ee_pos - obj_pos).norm(dim=-1),
    object_to_target_xy=d_xy.norm(dim=-1), fixed_contact=fixed, moving_contact=moving,
    grasped=grasped, lifted=lifted, in_bin_xy=in_bin, placed=placed, released=released,
    ref_point_w=ref_point, ref_error=ref_err,
  )
  cmd.state_cache[key] = s
  return s
