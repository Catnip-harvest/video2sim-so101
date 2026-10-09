"""Target-relative delta joint-position action with a per-joint speed cap.

    target_t = clip(target_{t-1} + clip(a, -1, 1) * max_speed * step_dt, soft limits)

* The action is a delta on the *previous target* (not on the measured position), the
  same convention as the real deploy loop: the robot keeps its own commanded target and
  the policy nudges it every control period.
* ``max_speed`` (rad/s) caps how far the target can move per control period. Default
  0.5 rad/s for the arm and 2.0 rad/s for the jaw: the arm figure is the real-servo
  command speed reported for SO-101 by livekit-examples/so-frame (10 Hz loop, 0.05 rad
  per step); the jaw is quicker so it can open/close within a reach.
* 0-1 control-step action latency: each episode an env is, with probability
  ``latency_prob``, given one step of latency (it applies the previous action).
* Targets are kept in the encoder frame (the frame the policy observes); the
  per-env encoder bias is removed when the target is sent to the servo, exactly as
  mjlab's JointPositionAction does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.actuator.actuator import TransmissionType

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TargetDeltaJointPositionAction(BaseAction):
  cfg: TargetDeltaJointPositionActionCfg

  def __init__(self, cfg: TargetDeltaJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    n, dev = self.num_envs, self.device
    speeds = []
    for name in self._target_names:
      speeds.append(cfg.gripper_max_speed if name in cfg.gripper_names else cfg.max_speed)
    self.max_delta = torch.tensor(speeds, device=dev) * env.step_dt  # (A,)
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]  # (B, A, 2)
    self._lo = limits[..., 0].clone()
    self._hi = limits[..., 1].clone()
    self.target = self._entity.data.default_joint_pos[:, self._target_ids].clone()
    self._prev_raw = torch.zeros(n, self.action_dim, device=dev)
    self.latency = torch.zeros(n, dtype=torch.bool, device=dev)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    applied = torch.where(self.latency[:, None], self._prev_raw, actions)
    self._prev_raw[:] = actions
    delta = applied.clamp(-1.0, 1.0) * self.max_delta
    self.target = torch.maximum(torch.minimum(self.target + delta, self._hi), self._lo)
    self._processed_actions = self.target

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(self.target - encoder_bias,
                                           joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    idx = slice(None) if env_ids is None else env_ids
    # Start from the measured (encoder-frame) position, as the real loop does.
    measured = self._entity.data.joint_pos_biased[idx][:, self._target_ids]
    self.target[idx] = measured
    self._prev_raw[idx] = 0.0
    k = self.target[idx].shape[0]
    self.latency[idx] = torch.rand(k, device=self.device) < self.cfg.latency_prob


@dataclass(kw_only=True)
class TargetDeltaJointPositionActionCfg(BaseActionCfg):
  max_speed: float = 0.5  # rad/s, arm joints
  gripper_max_speed: float = 2.0  # rad/s
  gripper_names: tuple[str, ...] = field(default_factory=lambda: ("gripper",))
  latency_prob: float = 0.5  # P(one control step of action latency) per episode

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> TargetDeltaJointPositionAction:
    return TargetDeltaJointPositionAction(self, env)
