"""Rewards. Object-centric: the main dense term makes the OBJECT follow the (warped)
demo object trajectory; reach / grasp / lift / place / release are shaping on top.
No term asks the gripper to follow the human hand (it does not transfer between a
shoulder-hung human arm and a table-mounted SO-101).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_track(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """exp(-d^2 / std^2), d = distance from the object to the nearest reference sample
  within +-ref_time_slack of the current time (tolerates being early / late)."""
  s = state.compute(env, command_name)
  return torch.exp(-s.ref_error.square() / std**2)


def reach(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  s = state.compute(env, command_name)
  return torch.exp(-s.ee_to_object.square() / std**2) * (~s.placed).float()


def grasp(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  s = state.compute(env, command_name)
  return (s.grasped & ~s.placed).float()


def lift(env: ManagerBasedRlEnv, command_name: str, height: float) -> torch.Tensor:
  """Grasped and the object bottom raised, saturating at ``height``."""
  s = state.compute(env, command_name)
  frac = (s.object_bottom / height).clamp(0.0, 1.0)
  return frac * (s.grasped & ~s.placed).float()


def carry_to_target(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """While lifted: xy closeness of the object to the bin centre."""
  s = state.compute(env, command_name)
  return torch.exp(-s.object_to_target_xy.square() / std**2) * s.lifted.float()


def placed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return state.compute(env, command_name).placed.float()


def released(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return state.compute(env, command_name).released.float()


def approach_side(
  env: ManagerBasedRlEnv,
  command_name: str,
  direction: tuple[float, float, float],
  gate_std: float,
) -> torch.Tensor:
  """Before the grasp, near the object: finger axis aligned with the demo's approach
  side (taken from the human hand)."""
  s = state.compute(env, command_name)
  d = torch.tensor(direction, device=env.device)
  align = (s.finger_axis_w * d).sum(dim=-1).clamp(min=0.0)
  gate = torch.exp(-s.ee_to_object.square() / gate_std**2)
  return align * gate * (~s.grasped & ~s.placed).float()


def object_dropped(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Penalty indicator: object outside the bin, not grasped, falling fast."""
  s = state.compute(env, command_name)
  falling = s.obj_linvel_w[:, 2] < -0.3
  return (falling & ~s.grasped & ~s.in_bin_xy).float()
