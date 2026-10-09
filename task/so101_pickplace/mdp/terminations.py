from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_out_of_bounds(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_radius: float = 0.42,
  max_height: float = 0.35,
) -> torch.Tensor:
  """Object knocked beyond reach (from the pan axis), thrown up, or through the table."""
  cmd = state.get_command(env, command_name)
  p = cmd.object.data.root_link_pos_w
  rel = p - cmd.base_pos_w
  r = torch.hypot(rel[:, 0] - 0.0388, rel[:, 1])
  z = p[:, 2] - env.scene.env_origins[:, 2]
  bad = (r > max_radius) | (z > max_height) | (z < -0.02)
  return bad | ~torch.isfinite(p).all(dim=-1)
