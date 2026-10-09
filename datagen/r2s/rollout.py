"""Simulate one episode with full physics and decide whether to keep it.

Physics: CPU MuJoCo, position actuators only. After the initial placement (object put
on the table, arm at its start pose) nothing writes qpos/qvel/xpos of anything: the
expert's joint targets go into ``data.ctrl`` and ``mj_step`` does the rest. The model is
asserted to contain no equality constraints and no mocap bodies.

Recording: 30 Hz. Frame k stores the measured joint positions at the start of the
frame (``observation.state``), the joint targets held for the next 7 physics steps
(``action``), the object pose, and the full qpos (used only to re-render accepted
episodes offline; rendering never feeds back into the physics).

Verification (all must hold to KEEP; every failure reason is logged):
  ik_unreachable / joint_limit_plan   the expert could not plan inside 95 % joint range
  object_disturbed_before_grasp       object moved > 5 mm before the jaw closed
  grasp_failed                        after the lift, the object is not up and held
  dropped_or_slipped                  a carried step without BOTH jaws on the object
  foreign_contact_while_carried       while carried, the object touched anything but the
                                      jaws (table, bin wall, arm links) - the bin floor is
                                      allowed only in the final lowering phase
  arm_body_touched_object             a non-jaw robot geom touched the object, any time
  arm_table_collision / arm_bin_collision / self_collision   any time
  joint_limit                         a joint went > 0.01 rad past its range
  torque_saturation                   an arm servo could not hold: for > 0.25 s its target
                                      was still, its torque was >= 98 % of the limit AND it
                                      sagged > 0.02 rad. (A kp ~ 1000 N*m/rad servo
                                      saturates on 1-2 mrad of error, so saturation alone
                                      says nothing; the jaw saturating while squeezing is
                                      how a position-servo gripper holds.)
  tracking_error                      an arm joint lagged its target by > 0.1 rad
  speed_cap                           a target moved faster than 0.5 rad/s (arm) / 2 rad/s (jaw)
  not_lifted                          object never rose >= 2 cm
  not_in_place_region                 final object centre not inside the bin walls by the
                                      spec's success margin, or not resting on the floor
  not_resting                         object still moving at the end
  not_released                        a jaw still touches the object in the last 0.5 s
  unstable                            non-finite state or absurd speeds
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import scene as S
from .paths import TaskSpec, mj_scene
from .planner import (ARM_VMAX, JAW_VMAX, Expert, Plan, PlanConfig, PlanError, plan_config_for,
                      rate_limit, smooth_targets)
from .variations import HOME_Q, Variation

REJECTION_ORDER = [
  "ik_unreachable", "joint_limit_plan", "unstable", "object_disturbed_before_grasp",
  "grasp_failed", "dropped_or_slipped", "foreign_contact_while_carried",
  "arm_body_touched_object", "arm_table_collision", "arm_bin_collision", "self_collision",
  "joint_limit", "torque_saturation", "tracking_error", "speed_cap", "not_lifted", "not_in_place_region",
  "not_resting", "not_released",
]
ROBOT_CATS = (S.CAT_FIXED_JAW, S.CAT_MOVING_JAW, S.CAT_ROBOT)
CARRY_PHASES = ("lift", "carry", "lower", "pre_release")
PRE_GRASP_PHASES = ("start", "approach_transit", "approach_line", "descend", "pre_close")
SETTLE_S = 0.3


@dataclass
class EpisodeResult:
  attempt: int
  accepted: bool
  reasons: list[str]
  primary_reason: str | None
  stats: dict
  variation: dict
  plan_info: dict
  sim_seconds: float
  wall_seconds: float
  # arrays (only meaningful when frames were simulated)
  state: np.ndarray | None = None  # (T, 6)
  action: np.ndarray | None = None  # (T, 6)
  env_state: np.ndarray | None = None  # (T, 7) object pos + quat (wxyz)
  qpos_full: np.ndarray | None = None  # (T, nq)
  phases: list[str] = field(default_factory=list)

  def summary(self) -> dict:
    return {"attempt": self.attempt, "accepted": self.accepted, "reasons": self.reasons,
            "primary_reason": self.primary_reason, "frames": 0 if self.state is None
            else int(len(self.state)), "stats": self.stats, "plan": self.plan_info,
            "variation": self.variation, "sim_seconds": round(self.sim_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3)}


class Monitor:
  """Per-physics-step contact bookkeeping."""

  def __init__(self, sc: S.EpisodeScene):
    self.sc = sc
    self.cat = sc.geom_cat
    self.c = {k: 0 for k in (
      "carry_steps", "carry_both_jaws", "carry_lost_grip", "carry_foreign",
      "arm_body_object", "arm_table", "arm_bin", "self", "joint_limit", "unstable",
      "end_jaw")}
    self.lifted_off = False
    self.rest_z = float(sc.data.xpos[sc.obj_body, 2])
    self.sat_run = np.zeros(5, int)
    self.sat_max_run = np.zeros(5, int)
    self.max_torque_frac = np.zeros(5)
    self.first = {}
    self.max_obj_z = -1.0
    self.carry_foreign_kinds: dict[str, int] = {}
    self.torque_by_phase: dict[str, np.ndarray] = {}
    self.max_track_err = 0.0
    self.prev_still = np.zeros(5, bool)

  def _mark(self, key: str, t: float) -> None:
    self.c[key] += 1
    self.first.setdefault(key, round(t, 3))

  def step(self, phase: str, holding: np.ndarray | None = None) -> None:
    sc, d, cat = self.sc, self.sc.data, self.cat
    n = d.ncon
    obj_fixed = obj_moving = obj_table = obj_floor = obj_wall = obj_robot = False
    if n:
      g = d.contact.geom[:n]
      c1, c2 = cat[g[:, 0]], cat[g[:, 1]]
      is_obj1, is_obj2 = c1 == S.CAT_OBJECT, c2 == S.CAT_OBJECT
      other = np.where(is_obj1, c2, np.where(is_obj2, c1, -1))
      oo = other[other >= 0]
      if len(oo):
        obj_fixed = bool(np.any(oo == S.CAT_FIXED_JAW))
        obj_moving = bool(np.any(oo == S.CAT_MOVING_JAW))
        obj_table = bool(np.any(oo == S.CAT_TABLE))
        obj_floor = bool(np.any(oo == S.CAT_BIN_FLOOR))
        obj_wall = bool(np.any(oo == S.CAT_BIN_WALL))
        obj_robot = bool(np.any(oo == S.CAT_ROBOT))
      r1 = np.isin(c1, ROBOT_CATS)
      r2 = np.isin(c2, ROBOT_CATS)
      if np.any((r1 & (c2 == S.CAT_TABLE)) | (r2 & (c1 == S.CAT_TABLE))):
        self._mark("arm_table", d.time)
      binc = (S.CAT_BIN_FLOOR, S.CAT_BIN_WALL)
      if np.any((r1 & np.isin(c2, binc)) | (r2 & np.isin(c1, binc))):
        self._mark("arm_bin", d.time)
      if np.any(r1 & r2):
        self._mark("self", d.time)
    if obj_robot:
      self._mark("arm_body_object", d.time)
    oz = float(d.xpos[sc.obj_body, 2])
    self.max_obj_z = max(self.max_obj_z, oz)
    if phase in CARRY_PHASES:
      if (not self.lifted_off and oz > self.rest_z + 0.005 and not obj_table
          and (obj_fixed or obj_moving)):
        self.lifted_off = True
      if self.lifted_off:
        self.c["carry_steps"] += 1
        if obj_fixed and obj_moving:
          self.c["carry_both_jaws"] += 1
        else:
          self._mark("carry_lost_grip", d.time)
        floor_ok = phase in ("lower", "pre_release")
        foreign = obj_table or obj_wall or obj_robot or (obj_floor and not floor_ok)
        if foreign:
          self._mark("carry_foreign", d.time)
          for k, flag in (("table", obj_table), ("bin_wall", obj_wall), ("arm", obj_robot),
                          ("bin_floor", obj_floor and not floor_ok)):
            if flag:
              self.carry_foreign_kinds[k] = self.carry_foreign_kinds.get(k, 0) + 1
    if phase == "rest" and (obj_fixed or obj_moving):
      self._mark("end_jaw", d.time)
    # joint limits (arm + jaw)
    q = d.qpos[np.r_[sc.arm_qadr, sc.g_qadr]]
    if np.any(q < sc.joint_lo - 0.01) or np.any(q > sc.joint_hi + 0.01):
      self._mark("joint_limit", d.time)
    f = np.abs(d.actuator_force[sc.arm_act])
    frac = f / sc.effort[:5]
    self.max_torque_frac = np.maximum(self.max_torque_frac, frac)
    prev = self.torque_by_phase.get(phase)
    self.torque_by_phase[phase] = frac.copy() if prev is None else np.maximum(prev, frac)
    # "Cannot hold": saturated while the target is still AND sagging > 0.02 rad.
    sag = np.abs(d.qpos[sc.arm_qadr] - d.ctrl[sc.arm_act]) > 0.02
    sat = (frac >= 0.98) & sag
    if holding is not None:
      sat &= holding
    self.sat_run = np.where(sat, self.sat_run + 1, 0)
    self.sat_max_run = np.maximum(self.sat_max_run, self.sat_run)
    if not np.all(np.isfinite(d.qpos)) or float(np.abs(d.qvel).max()) > 50.0:
      self._mark("unstable", d.time)

  def obj_jaw_contacts(self) -> tuple[bool, bool]:
    d, cat = self.sc.data, self.cat
    n = d.ncon
    if not n:
      return False, False
    g = d.contact.geom[:n]
    c1, c2 = cat[g[:, 0]], cat[g[:, 1]]
    other = np.where(c1 == S.CAT_OBJECT, c2, np.where(c2 == S.CAT_OBJECT, c1, -1))
    return bool(np.any(other == S.CAT_FIXED_JAW)), bool(np.any(other == S.CAT_MOVING_JAW))


class Recorder:
  def __init__(self, sc: S.EpisodeScene):
    self.sc = sc
    self.state, self.action, self.env, self.qfull, self.phase = [], [], [], [], []

  def frame(self, target: np.ndarray, phase: str) -> None:
    d, sc = self.sc.data, self.sc
    self.state.append(d.qpos[np.r_[sc.arm_qadr, sc.g_qadr]].copy())
    self.action.append(target.copy())
    self.env.append(np.r_[d.xpos[sc.obj_body], d.xquat[sc.obj_body]])
    self.qfull.append(d.qpos.copy())
    self.phase.append(phase)


def _run_frames(sc, mon: Monitor, rec: Recorder, targets: np.ndarray, phases: list[str],
                servo_foh: bool = True) -> None:
  """Simulate frames. ``servo_foh``: the 7 physics steps of a frame ramp the servo goal
  linearly from the previous target to the new one (first-order hold), standing in for
  the STS3215's onboard motion profile. With servo_foh=False the goal jumps once per
  frame (zero-order hold): a kp ~ 1000 N*m/rad servo then snaps to every 15 mrad step,
  and those 30 Hz jolts shake small cubes out of the jaws (in-hand slip 10-60 mm instead
  of ~0.2 mm; measured, see README)."""
  d, m = sc.data, sc.model
  for k in range(len(targets)):
    rec.frame(targets[k], phases[k])
    err = np.abs(d.qpos[sc.arm_qadr] - d.ctrl[sc.arm_act])
    mon.max_track_err = max(mon.max_track_err, float(np.max(err)))
    prev = d.ctrl[sc.arm_act].copy()
    prev_g = float(d.ctrl[sc.g_act])
    # A joint is "holding" when its target did not move this frame or the last one.
    still = np.abs(targets[k, :5] - prev) < 1e-7
    holding = still & mon.prev_still
    mon.prev_still = still
    for j in range(S.SUBSTEPS):
      if servo_foh:
        # First-order hold: the servo's internal motion profile ramps from the last
        # goal to the new one over the 33 ms control period instead of jumping.
        a = (j + 1) / S.SUBSTEPS
        d.ctrl[sc.arm_act] = prev + a * (targets[k, :5] - prev)
        d.ctrl[sc.g_act] = prev_g + a * (targets[k, 5] - prev_g)
      else:
        d.ctrl[sc.arm_act] = targets[k, :5]
        d.ctrl[sc.g_act] = targets[k, 5]
      mujoco.mj_step(m, d)
      mon.step(phases[k], holding)
    if mon.c["unstable"]:
      return


def _speed_ok(targets: np.ndarray, prev: np.ndarray | None) -> tuple[bool, float, float]:
  t = targets if prev is None else np.concatenate([prev[None], targets], 0)
  if len(t) < 2:
    return True, 0.0, 0.0
  dv = np.abs(np.diff(t, axis=0)) * S.FPS
  arm, jaw = float(dv[:, :5].max()), float(dv[:, 5].max())
  return (arm <= ARM_VMAX + 1e-6 and jaw <= JAW_VMAX + 1e-6), arm, jaw


def run_episode(spec: TaskSpec, v: Variation, cfg: PlanConfig | None = None,
                pinch_width: float | None = None, render_size=(256, 256),
                keep_arrays: bool = True, servo_foh: bool = True) -> EpisodeResult:
  wall0 = time.perf_counter()
  cfg = cfg or plan_config_for(v)
  sc = S.build_episode_scene(v, object_kind=spec.object.kind, render_size=render_size)
  m, d = sc.model, sc.data
  assert m.neq == 0 and m.nmocap == 0, "model must have no equality constraints or mocap"
  # Initial placement (the only state write in the episode).
  mujoco.mj_resetData(m, d)
  q0 = np.array(v.q_start)
  d.qpos[sc.arm_qadr] = q0
  d.qpos[sc.g_qadr] = v.gripper_start
  d.ctrl[sc.arm_act] = q0
  d.ctrl[sc.g_act] = v.gripper_start
  mujoco.mj_forward(m, d)
  for _ in range(int(SETTLE_S / m.opt.timestep)):
    mujoco.mj_step(m, d)
  d.time = 0.0
  obj_rest = d.xpos[sc.obj_body].copy()
  quat = d.xquat[sc.obj_body]
  rest_yaw = 2 * math.atan2(quat[3], quat[0])

  mon, rec = Monitor(sc), Recorder(sc)
  reason_t: dict[str, float] = {}  # reason -> sim time it first applied

  def flag(reason: str, t: float | None = None) -> None:
    reason_t.setdefault(reason, round(float(d.time if t is None else t), 3))

  def contact_reasons() -> None:
    """Reasons derived from the per-step monitor; checked however the episode ended."""
    c, f = mon.c, mon.first
    for key, reason in (("carry_lost_grip", "dropped_or_slipped"),
                        ("carry_foreign", "foreign_contact_while_carried"),
                        ("arm_body_object", "arm_body_touched_object"),
                        ("arm_table", "arm_table_collision"), ("arm_bin", "arm_bin_collision"),
                        ("self", "self_collision"), ("joint_limit", "joint_limit"),
                        ("unstable", "unstable")):
      if c[key]:
        flag(reason, f.get(key))
    if np.any(mon.sat_max_run * m.opt.timestep > 0.25):
      flag("torque_saturation")
    if mon.max_track_err > 0.1:
      flag("tracking_error")
    if not speed_ok:
      flag("speed_cap")

  plan_info: dict = {}
  stats: dict = {}
  expert = Expert(sc, spec, v, cfg, pinch_width=pinch_width)
  speed_ok = True
  speed_max = [0.0, 0.0]

  def finish(accepted_possible: bool) -> EpisodeResult:
    sim_s = len(rec.state) / S.FPS
    stats.update({
      "contacts": dict(mon.c), "first_violation_t": mon.first,
      "carry_foreign_kinds": mon.carry_foreign_kinds,
      "max_obj_lift_mm": round((mon.max_obj_z - obj_rest[2]) * 1e3, 2),
      "max_arm_torque_frac": np.round(mon.max_torque_frac, 3).tolist(),
      "max_saturated_while_holding_s": np.round(mon.sat_max_run * m.opt.timestep, 3).tolist(),
      "max_tracking_error_rad": round(mon.max_track_err, 4),
      "max_torque_frac_by_phase": {k: np.round(val, 2).tolist()
                                   for k, val in mon.torque_by_phase.items()},
      "ik_worst": {k: round(val, 3) for k, val in expert.ik_worst.items()},
      "max_target_speed_arm_rad_s": round(speed_max[0], 4),
      "max_target_speed_jaw_rad_s": round(speed_max[1], 4),
    })
    contact_reasons()
    # Primary reason = the one that happened first in sim time (ties: REJECTION_ORDER),
    # so a drop that later makes the re-planner fail is reported as the drop.
    uniq = sorted(reason_t, key=lambda r: (reason_t[r], REJECTION_ORDER.index(r)))
    stats["reason_times_s"] = dict(reason_t)
    acc = accepted_possible and not uniq
    res = EpisodeResult(
      attempt=v.attempt, accepted=acc, reasons=uniq, primary_reason=uniq[0] if uniq else None,
      stats=stats, variation=v.to_dict(), plan_info=_jsonable(plan_info), sim_seconds=sim_s,
      wall_seconds=time.perf_counter() - wall0, phases=rec.phase)
    if keep_arrays and rec.state:
      res.state = np.array(rec.state, np.float32)
      res.action = np.array(rec.action, np.float32)
      res.env_state = np.array(rec.env, np.float32)
      res.qpos_full = np.array(rec.qfull, np.float64)
    return res

  # ---- stage 1: approach and close ----------------------------------------------
  try:
    plan, pick_info = expert.plan_pick(q0, v.gripper_start, obj_rest, rest_yaw)
  except PlanError as e:
    flag(e.reason)
    plan_info["error"] = str(e)
    return finish(False)
  plan_info["pick"] = pick_info
  targets, phases = plan.arrays()
  targets = rate_limit(smooth_targets(targets, cfg.smooth_window), targets[0])
  ok, a, j = _speed_ok(targets, None)
  speed_ok &= ok
  speed_max = [max(speed_max[0], a), max(speed_max[1], j)]
  _run_frames(sc, mon, rec, targets, phases, servo_foh)
  if mon.c["unstable"]:
    flag("unstable")
    return finish(False)
  # Object displacement before the jaw starts closing (state at the start of "close").
  k_close = phases.index("close")
  moved = float(np.linalg.norm(rec.env[k_close][:2] - obj_rest[:2]))
  stats["object_shift_before_close_mm"] = round(moved * 1e3, 2)
  stats["object_shift_after_squeeze_mm"] = round(
    float(np.linalg.norm(d.xpos[sc.obj_body][:2] - obj_rest[:2])) * 1e3, 2)
  if moved > 0.005:
    flag("object_disturbed_before_grasp", k_close / S.FPS)
  fixed, moving = mon.obj_jaw_contacts()
  stats["jaw_angle_after_squeeze"] = round(float(d.qpos[sc.g_qadr]), 4)
  if not (fixed and moving):
    flag("grasp_failed")
    plan_info["grasp_failed_at"] = "squeeze (jaws not both on object)"

  # ---- stages 2 and 3: re-planned from the simulated state ------------------------
  def run_stage(plan_fn) -> dict:
    """Plan from the last applied target, smooth across the seam, check the speed cap,
    then simulate. plan_fn(plan) appends frames and returns an info dict."""
    nonlocal speed_ok, speed_max
    last = np.asarray(rec.action[-1], float)
    p = Plan()
    p.add(last[None, :5], np.array([last[5]]), "anchor")
    info = plan_fn(p)
    t, ph = p.arrays()
    t, ph = t[1:], ph[1:]
    prev = np.asarray(rec.action[-3:], float)
    t = smooth_targets(np.concatenate([prev, t], 0), cfg.smooth_window)[len(prev):]
    t = rate_limit(t, last)
    ok, a, j = _speed_ok(t, last)
    speed_ok &= ok
    speed_max = [max(speed_max[0], a), max(speed_max[1], j)]
    _run_frames(sc, mon, rec, t, ph, servo_foh)
    return info

  def grip_state():
    return (d.xpos[sc.obj_body].copy(), d.site_xpos[sc.site].copy(),
            d.xmat[sc.grip_body].reshape(3, 3).copy())

  try:
    expert.attach_object(pick_info, *grip_state())
    plan_info["lift"] = run_stage(lambda p: expert.plan_lift(p, d.xpos[sc.obj_body].copy()))
    fixed, moving = mon.obj_jaw_contacts()
    lifted = d.xpos[sc.obj_body][2] - obj_rest[2]
    stats["lift_check_mm"] = round(float(lifted) * 1e3, 2)
    if not (fixed and moving) or lifted < 0.01:
      flag("grasp_failed")
      return finish(False)
    plan_info["carry"] = run_stage(lambda p: expert.plan_carry(p, np.array(v.bin_xy)))
    if mon.c["unstable"]:
      flag("unstable")
      return finish(False)
    o, st, R = grip_state()
    R_obj = d.xmat[sc.obj_body].reshape(3, 3).copy()
    plan_info["release"] = run_stage(
      lambda p: expert.plan_release(p, o, st, R, np.array(v.bin_xy), HOME_Q, R_obj=R_obj))
  except PlanError as e:
    flag(e.reason)
    plan_info["error"] = str(e)
    return finish(False)
  if mon.c["unstable"]:
    flag("unstable")
    return finish(False)
  plan_info["retilts"] = expert.retilts
  a0 = np.array(plan_info["lift"]["obj_in_gripper_mm"])
  a1 = np.array(plan_info["release"]["obj_in_gripper_before_lower_mm"])
  stats["in_hand_slip_mm"] = round(float(np.linalg.norm(a1 - a0)), 2)

  # ---- verdict (end-of-episode checks; contact checks run in finish) ------------
  c = mon.c
  if mon.max_obj_z - obj_rest[2] < 0.02:
    flag("not_lifted")
  op = d.xpos[sc.obj_body]
  bx, by = v.bin_xy
  lim = mj_scene.BIN_INNER_HALF - spec.place.success_xy_margin
  rest_z = mj_scene.bin_floor_top() + v.obj_size / 2
  in_xy = abs(op[0] - bx) <= lim and abs(op[1] - by) <= lim
  on_floor = abs(op[2] - rest_z) < 0.004
  if not (in_xy and on_floor):
    flag("not_in_place_region")
  speed = float(np.linalg.norm(d.qvel[sc.obj_dadr:sc.obj_dadr + 3]))
  if speed > 0.02:
    flag("not_resting")
  # released: no jaw contact during the final 0.5 s "rest" phase, nor at the very end
  fj, mj = mon.obj_jaw_contacts()
  if c["end_jaw"] or fj or mj:
    flag("not_released")
  stats.update({
    "final_obj_offset_from_bin_mm": [round((op[0] - bx) * 1e3, 2), round((op[1] - by) * 1e3, 2)],
    "final_obj_z_err_mm": round((op[2] - rest_z) * 1e3, 2),
    "final_obj_speed_m_s": round(speed, 4),
    "carry_steps": c["carry_steps"], "carry_both_jaw_steps": c["carry_both_jaws"],
  })
  return finish(True)


def _jsonable(x):
  return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
