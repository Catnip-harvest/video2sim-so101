"""Pick-and-place command: object start, place target and the object reference path.

Once per episode (on reset) it

* samples the object's start (x, y, yaw) from the task spec and puts it on the table,
  resting on its own (possibly size-randomised) half height;
* samples the bin position from the task spec, rejecting draws closer than
  ``min_separation`` to the object, and moves the bin (a mocap body) there;
* warps the demo's object trajectory onto this episode: the reference starts exactly
  at the sampled object and ends resting on the sampled bin's floor. The start offset
  fades into the end offset over the carry phase (grasp_t -> release_t), so the
  demo's lift height and arc shape are kept.

This is the only place the object's state is ever written, and only at reset. During
the episode the object moves by physics alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class PickPlaceCommand(CommandTerm):
  cfg: PickPlaceCommandCfg

  def __init__(self, cfg: PickPlaceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.robot_name]
    self.object: Entity = env.scene[cfg.object_name]
    self.bin: Entity = env.scene[cfg.bin_name]
    n, dev = self.num_envs, self.device

    self.base_pos_w = torch.zeros(n, 3, device=dev)
    self.object_start_w = torch.zeros(n, 3, device=dev)
    self.target_pos_w = torch.zeros(n, 3, device=dev)  # bin centre on the table
    self.object_half_height = torch.full((n,), cfg.object_half_height, device=dev)
    self.object_half_width = torch.full((n,), cfg.object_half_width, device=dev)
    self.episode_success = torch.zeros(n, device=dev)

    # Reference trajectory (robot clock, base frame), uniformly sampled.
    if len(cfg.ref_t) < 2 or len(cfg.ref_pos) != 3 * len(cfg.ref_t):
      raise ValueError("PickPlaceCommandCfg needs ref_t (N>=2) and ref_pos (3N)")
    self.ref_t = torch.tensor(cfg.ref_t, device=dev, dtype=torch.float32)
    self.ref_pos = torch.tensor(cfg.ref_pos, device=dev, dtype=torch.float32).reshape(-1, 3)
    self.ref_dt = float(cfg.ref_t[1] - cfg.ref_t[0])
    self.ref_duration = float(cfg.ref_t[-1])
    span = max(cfg.ref_release_t - cfg.ref_grasp_t, 1e-6)
    self._blend = ((self.ref_t - cfg.ref_grasp_t) / span).clamp(0.0, 1.0)  # (N,)
    self.ref_w = torch.zeros(n, len(cfg.ref_t), 3, device=dev)

    # Measured-object buffer for the xy observation with dropouts (see observations).
    self.object_xy_meas = torch.zeros(n, 2, device=dev)
    self.object_yaw_meas = torch.zeros(n, device=dev)

    # Per-step cache for mdp.state.compute(); cleared on every resample (= reset).
    self.state_cache: dict = {}

    self._obj_geom_id = int(self.object.indexing.geom_ids[0].item())

    for k in ("object_height", "ee_to_object", "object_to_target_xy", "ref_error",
              "grasped", "lifted", "placed", "released", "episode_success"):
      self.metrics[k] = torch.zeros(n, device=dev)

  # ------------------------------------------------------------------ interface
  @property
  def command(self) -> torch.Tensor:
    return self.target_pos_w

  def compute_success(self) -> torch.Tensor:
    return self.metrics["released"] > 0.5

  def episode_time(self) -> torch.Tensor:
    return self._env.episode_length_buf.float() * self._env.step_dt

  def ref_index(self, t: torch.Tensor) -> torch.Tensor:
    return (t / self.ref_dt).round().long().clamp(0, self.ref_t.numel() - 1)

  # ------------------------------------------------------------------ helpers
  def _object_extents(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env half height / half width from the (possibly randomised) geom size."""
    gs = self._env.sim.model.geom_size  # (nworld | 1, ngeom, 3)
    if gs.shape[0] > 1:
      size = gs[env_ids, self._obj_geom_id]
    else:  # not randomised (and not broadcast on Warp's CPU device)
      size = gs[0, self._obj_geom_id].expand(len(env_ids), 3)
    if self.cfg.object_kind == "box":
      return size[:, 2], size[:, 0]
    return size[:, 1], size[:, 0]  # cylinder: (radius, half-length) -> (h, w)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    dev = self.device
    c = self.cfg
    self.state_cache.clear()
    origins = self._env.scene.env_origins[env_ids]
    # The base is a mocap body placed at its env origin (+ mount height), identity
    # rotation, by the reset_base event. Computed here rather than read back because
    # derived quantities are stale inside a state-writing function (mjlab FAQ).
    base_pos = origins + torch.tensor([0.0, 0.0, c.mount_height], device=dev)
    table_z = origins[:, 2]
    self.base_pos_w[env_ids] = base_pos

    half_h, half_w = self._object_extents(env_ids)
    self.object_half_height[env_ids] = half_h
    self.object_half_width[env_ids] = half_w

    ox = sample_uniform(c.object_x[0], c.object_x[1], (n,), device=dev)
    oy = sample_uniform(c.object_y[0], c.object_y[1], (n,), device=dev)
    yaw = sample_uniform(c.object_yaw[0], c.object_yaw[1], (n,), device=dev)
    tx = sample_uniform(c.target_x[0], c.target_x[1], (n,), device=dev)
    ty = sample_uniform(c.target_y[0], c.target_y[1], (n,), device=dev)
    for _ in range(c.max_separation_tries):  # vectorised rejection sampling
      bad = torch.hypot(tx - ox, ty - oy) < c.min_separation
      if not bool(bad.any()):
        break
      k = int(bad.sum())
      tx[bad] = sample_uniform(c.target_x[0], c.target_x[1], (k,), device=dev)
      ty[bad] = sample_uniform(c.target_y[0], c.target_y[1], (k,), device=dev)

    zeros = torch.zeros(n, device=dev)
    obj_w = torch.stack([base_pos[:, 0] + ox, base_pos[:, 1] + oy,
                         table_z + half_h + c.spawn_clearance], dim=-1)
    tgt_w = torch.stack([base_pos[:, 0] + tx, base_pos[:, 1] + ty, table_z], dim=-1)
    self.object_start_w[env_ids] = obj_w
    self.target_pos_w[env_ids] = tgt_w
    self.episode_success[env_ids] = 0.0

    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    self.object.write_root_link_pose_to_sim(torch.cat([obj_w, quat], dim=-1), env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
    one = torch.ones(n, device=dev)
    bin_quat = torch.stack([one, zeros, zeros, zeros], dim=-1)
    self.bin.write_mocap_pose_to_sim(torch.cat([tgt_w, bin_quat], dim=-1), env_ids=env_ids)

    # Warp the reference (base frame -> world, then blend start/end offsets).
    ref_local = self.ref_pos  # (N, 3), base frame, table at z = 0
    start_local = torch.stack([ox, oy, half_h + c.spawn_clearance], dim=-1)
    end_local = torch.stack([tx, ty, c.bin_floor_top + half_h], dim=-1)
    d0 = start_local - ref_local[0]  # (n, 3)
    d1 = end_local - ref_local[-1]
    b = self._blend[None, :, None]  # (1, N, 1)
    warped = ref_local[None] + (1 - b) * d0[:, None, :] + b * d1[:, None, :]
    base_xy0 = torch.stack([base_pos[:, 0], base_pos[:, 1], table_z], dim=-1)
    self.ref_w[env_ids] = warped + base_xy0[:, None, :]

    # Measurement buffers start at the true values.
    self.object_xy_meas[env_ids] = torch.stack([ox, oy], dim=-1)
    self.object_yaw_meas[env_ids] = yaw

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    from . import state  # local import: avoids an import cycle

    s = state.compute(self._env, self.cfg.self_name, phase="obs")
    self.episode_success = torch.maximum(self.episode_success, s.released.float())
    self.metrics["object_height"] = s.object_bottom
    self.metrics["ee_to_object"] = s.ee_to_object
    self.metrics["object_to_target_xy"] = s.object_to_target_xy
    self.metrics["ref_error"] = s.ref_error
    self.metrics["grasped"] = s.grasped.float()
    self.metrics["lifted"] = s.lifted.float()
    self.metrics["placed"] = s.placed.float()
    self.metrics["released"] = s.released.float()
    self.metrics["episode_success"] = self.episode_success

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    t = self.episode_time()
    k = self.ref_index(t)
    for batch in visualizer.get_env_indices(self.num_envs):
      tgt = self.target_pos_w[batch].cpu().numpy().copy()
      tgt[2] += 0.04
      visualizer.add_sphere(center=tgt, radius=0.010, color=(1.0, 0.5, 0.0, 0.5),
                            label=f"place_target_{batch}")
      ref = self.ref_w[batch, int(k[batch])].cpu().numpy()
      visualizer.add_sphere(center=ref, radius=0.008, color=(0.2, 0.8, 1.0, 0.6),
                            label=f"object_ref_{batch}")


@dataclass(kw_only=True)
class PickPlaceCommandCfg(CommandTermCfg):
  self_name: str = "pick_place"  # this term's key in cfg.commands
  robot_name: str = "robot"
  object_name: str = "object"
  bin_name: str = "bin"
  object_kind: str = "cylinder"
  object_half_height: float = 0.020
  object_half_width: float = 0.015
  object_x: tuple[float, float] = (0.17, 0.27)
  object_y: tuple[float, float] = (-0.10, 0.06)
  object_yaw: tuple[float, float] = (-3.14159, 3.14159)
  target_x: tuple[float, float] = (0.10, 0.20)
  target_y: tuple[float, float] = (0.15, 0.21)
  min_separation: float = 0.11
  max_separation_tries: int = 8
  spawn_clearance: float = 0.001
  mount_height: float = 0.0
  # Bin geometry needed for the "placed" test.
  bin_inner_half: float = 0.055
  bin_floor_top: float = 0.005
  success_xy_margin: float = 0.005
  lift_height: float = 0.03  # "lifted" = grasped and object bottom >= this (m)
  # Object reference trajectory, robot clock / base frame, flattened for tyro.
  ref_t: tuple[float, ...] = ()
  ref_pos: tuple[float, ...] = ()
  ref_grasp_t: float = 0.0
  ref_release_t: float = 0.0
  ref_time_slack: float = 1.0

  def build(self, env: ManagerBasedRlEnv) -> PickPlaceCommand:
    return PickPlaceCommand(self, env)
