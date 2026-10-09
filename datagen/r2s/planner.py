"""Scripted expert for one episode: IK waypoints -> speed-capped 30 Hz joint targets.

The expert never touches the simulation state. It only produces joint *targets*
(what a LeRobot SO-101 follower receives); the position servos then have to reach them
through the dynamics. It is planned in two stages, like any scripted data collector
with state feedback:

  stage 1 (open loop from the start state): transit to a pre-approach point on the
          human's approach side -> straight line to a hover point -> vertical descent ->
          close the jaw;
  stage 2 (re-planned from the simulated state after the squeeze, using where the
          object actually sits between the jaws): vertical lift -> carry along the
          human's object path (mapped onto this episode's pick and place) -> lower into
          the bin -> open -> retreat -> return home.

Speed caps: arm joints 0.5 rad/s (real STS3215 command speed used by the mjlab task and
livekit-examples/so-frame), jaw 2.0 rad/s (the task's env_cfg value). Cartesian
segments are additionally capped at a few cm/s. All arm targets stay inside 95 % of the
joint range.

The carry follows R2R2R's lesson (smooth AND densely interpolate the object path) and
X-Sim's object-centric idea (the object, not the hand, follows the demo).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .paths import TaskSpec, ik, mj_scene, rm
from .scene import FPS, EpisodeScene
from .variations import PAN_AXIS, Variation

ARM_VMAX = 0.5  # rad/s, hard cap (verified on the produced targets)
JAW_VMAX = 2.0  # rad/s
PLAN_MARGIN = 0.9  # plan at 90 % of the caps so smoothing can never exceed them
SOFT_LIMIT = 0.95
DT = 1.0 / FPS
# Highest grasp-site height with a near-vertical (<= 15 deg) approach, by radius from the
# pan axis (manip/task/README.md section 4, measured). Used as a per-point ceiling on
# the carried object's path, with a margin.
REACH_R = np.array([0.05, 0.10, 0.15, 0.175, 0.25, 0.30])
REACH_Z = np.array([0.051, 0.102, 0.126, 0.133, 0.114, 0.072])
RETILT_CANDIDATES = tuple(math.radians(t) for t in (0, 10, -10, 20, -20, 30, -30))


@dataclass
class PlanConfig:
  approach_tilt_max_deg: float = 10.0  # finger-axis tilt allowed towards the spec's approach side
  side_offset: float = 0.04  # m: pre-approach point offset along the human approach side
  preapproach_height: float = 0.07  # m above the grasp-site height
  hover_height: float = 0.04
  descend_speed: float = 0.03  # m/s
  line_speed: float = 0.05
  lift_speed: float = 0.04
  lower_speed: float = 0.03
  retreat_height: float = 0.06
  wall_clearance: float = 0.02  # object bottom above the bin rim while carried
  carry_z_max: float = 0.12  # object-centre ceiling (vertical-ish fingers lose reach above)
  release_gap: float = 0.005  # object bottom above the bin floor when the jaw opens
  min_carry_radius: float = 0.15  # keep the carried object this far from the pan axis
  lift_clearance: float = 0.02  # vertical lift until the object bottom is this high
  fixed_jaw_clearance: float = 0.004  # same 4 mm as the grasp gate
  tip_clearance: float = 0.002  # lowest jaw point above the table at the grasp
  arc_scales: tuple = (1.0, 0.7, 0.4, 0.0)  # human lift-arc scales tried if IK can't reach
  open_margin: float = 0.040  # jaw mouth = pinch width + this
  release_mouth_margin: float = 0.020  # jaw mouth at release = object size + this
  squeeze: float = 0.35  # rad: close target this far past the contact angle
  ik_pos_tol: float = 0.003
  ik_app_tol_deg: float = 5.0
  carry_app_tol_deg: float = 25.0
  smooth_window: int = 5
  close_jaw: bool = True  # False = negative control (tests): the jaw never closes
  # Pauses (s). Defaults are the original fixed values; --variety randomises them.
  hover_pause_s: float = 0.0  # extra stop at the hover point before descending
  pre_close_s: float = 0.25
  pre_release_s: float = 0.2
  post_release_s: float = 0.4


# Ranges drawn per episode by variations.sample_variation(..., variety=True). Speeds are
# scaled together (one "operator tempo") and stay under the servo caps (rate_limit).
VARIETY_RANGES = {
  "speed_scale": (0.7, 1.3), "preapproach_height": (0.05, 0.10), "hover_height": (0.03, 0.06),
  "side_offset": (0.02, 0.06), "approach_tilt_max_deg": (4.0, 16.0),
  "pre_close_s": (0.15, 0.6), "pre_release_s": (0.1, 0.6), "post_release_s": (0.3, 0.7),
  "hover_pause_s": (0.2, 0.8),  # applied with probability HOVER_PAUSE_P, else 0
}
HOVER_PAUSE_P = 0.35


def plan_config_for(v: Variation) -> PlanConfig:
  """PlanConfig for one episode: defaults, overridden by the variation's drawn plan params."""
  cfg = PlanConfig()
  p = (v.extras or {}).get("plan")
  if not p:
    return cfg
  k = p["speed_scale"]
  for name in ("descend_speed", "line_speed", "lift_speed", "lower_speed"):
    setattr(cfg, name, getattr(cfg, name) * k)
  for name in ("preapproach_height", "hover_height", "side_offset", "approach_tilt_max_deg",
               "pre_close_s", "pre_release_s", "post_release_s", "hover_pause_s"):
    setattr(cfg, name, float(p[name]))
  return cfg


class PlanError(RuntimeError):
  def __init__(self, reason: str, detail: str = ""):
    super().__init__(f"{reason}: {detail}")
    self.reason = reason
    self.detail = detail


def mouth_to_jaw_angle(width: float) -> float:
  """Jaw angle whose moving-tip mouth equals ``width`` (m), from the measured table in
  manip/task/README.md section 4 (0 mm at the closed stop)."""
  w = np.array([0.0, 0.048, 0.063, 0.070, 0.087, 0.107])
  q = np.array([rm.GRIPPER_CLOSED, 0.42, 0.61, 0.71, 1.0, 1.745])
  return float(np.interp(width, w, q))


def azimuth(p: np.ndarray) -> float:
  return math.atan2(p[1] - PAN_AXIS[1], p[0] - PAN_AXIS[0])


def rot_z(a: float) -> np.ndarray:
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def min_jerk(n: int) -> np.ndarray:
  tau = np.arange(1, n + 1) / n
  return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


@dataclass
class Plan:
  arm: list[np.ndarray] = field(default_factory=list)  # each (k, 5)
  jaw: list[np.ndarray] = field(default_factory=list)  # each (k,)
  phase: list[str] = field(default_factory=list)  # one label per frame

  def add(self, arm: np.ndarray, jaw: np.ndarray, label: str) -> None:
    assert len(arm) == len(jaw)
    self.arm.append(np.asarray(arm, float))
    self.jaw.append(np.asarray(jaw, float))
    self.phase += [label] * len(arm)

  def last(self) -> tuple[np.ndarray, float]:
    return self.arm[-1][-1].copy(), float(self.jaw[-1][-1])

  def arrays(self) -> tuple[np.ndarray, list[str]]:
    a = np.concatenate(self.arm, axis=0)
    j = np.concatenate(self.jaw)[:, None]
    return np.concatenate([a, j], axis=1), list(self.phase)


class Expert:
  def __init__(self, sc: EpisodeScene, spec: TaskSpec, v: Variation, cfg: PlanConfig,
               pinch_width: float | None = None):
    self.sc, self.spec, self.v, self.cfg = sc, spec, v, cfg
    m = sc.model
    self.lo = sc.joint_lo[:5]
    self.hi = sc.joint_hi[:5]
    mid, half = (self.lo + self.hi) / 2, (self.hi - self.lo) / 2
    self.soft_lo, self.soft_hi = mid - SOFT_LIMIT * half, mid + SOFT_LIMIT * half
    self.half = v.obj_size / 2
    self.half_w = v.obj_size / 2
    self.rest_z = self.half
    width = pinch_width if pinch_width is not None else v.obj_size
    width = float(np.clip(width, v.obj_size, 0.050))  # never open less than the object
    self.g_open = float(np.clip(mouth_to_jaw_angle(width + cfg.open_margin), 0.6, 1.0))
    self.g_close = float(max(rm.GRIPPER_CLOSED,
                             mouth_to_jaw_angle(v.obj_size) - cfg.squeeze))
    self.g_release = float(np.clip(mouth_to_jaw_angle(v.obj_size + cfg.release_mouth_margin),
                                   0.25, self.g_open))
    if not cfg.close_jaw:
      self.g_close = self.g_open
    # Grasp-site height: object centre, but the lowest fixed-jaw point stays
    # cfg.tip_clearance above the table (the grasp gate uses 3 mm; 2 mm puts the pads
    # lower on small cubes, which stops them squeezing out at the tip spheres).
    self.grasp_z = max(self.half, rm.JAW_TIP_BELOW_SITE + cfg.tip_clearance)
    # IK runs on a kinematic copy whose arm ranges are shrunk to 95 %, so solutions
    # respect the soft limits instead of being rejected after the fact.
    self.plan_model = copy.deepcopy(m)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in rm.ARM_JOINTS]
    self.plan_model.jnt_range[jids, 0] = self.soft_lo
    self.plan_model.jnt_range[jids, 1] = self.soft_hi
    self._scratch = mujoco.MjData(m)
    self._coll = mujoco.MjData(m)
    self._coll.qpos[:] = m.qpos0
    self.g_now = 0.3  # jaw angle used by the collision check (updated as plans are built)
    self.check_object = False
    self.tilt = 0.0  # signed finger tilt (rad) in the radial plane; + = fingertips outward
    self.app_tol = cfg.ik_app_tol_deg
    self.retilts = 0
    self.ik_calls: list[int] = []
    self.ik_worst = {"pos_mm": 0.0, "app_deg": 0.0}
    d = np.asarray(spec.approach.direction, float)
    self.spec_dir = d / np.linalg.norm(d)

  # --- geometry helpers ---------------------------------------------------------
  def spec_tilt(self, pick_az: float) -> float:
    """Finger tilt towards the spec's approach side, clamped to cfg max, in the arm's
    radial plane (a 5-DoF arm can only tilt the fingers radially)."""
    u_pick = np.array([math.cos(pick_az), math.sin(pick_az)])
    h = self.spec_dir[:2]
    tilt_spec = math.acos(np.clip(-self.spec_dir[2], -1, 1))
    radial = float(np.dot(h, u_pick)) / max(np.linalg.norm(h), 1e-9)
    return min(tilt_spec, math.radians(self.cfg.approach_tilt_max_deg)) * radial

  def approach_at(self, p: np.ndarray, pick_az: float) -> np.ndarray:
    """Finger axis at site position p: the episode's tilt, carried round with the pan."""
    tilt = self.tilt
    az = azimuth(p)
    u = np.array([math.cos(az), math.sin(az), 0.0])
    return np.array([0, 0, -math.cos(tilt)]) + u * math.sin(tilt)

  def collides(self, q: np.ndarray) -> bool:
    """Kinematic check of a planned pose: robot-robot or robot-table contact."""
    d, sc = self._coll, self.sc
    d.qpos[sc.arm_qadr] = q
    d.qpos[sc.g_qadr] = self.g_now
    mujoco.mj_fwdPosition(sc.model, d)
    n = d.ncon
    if not n:
      return False
    g = d.contact.geom[:n]
    c1, c2 = sc.geom_cat[g[:, 0]], sc.geom_cat[g[:, 1]]
    robot = (5, 6, 7)
    r1, r2 = np.isin(c1, robot), np.isin(c2, robot)
    bad = (r1 & r2) | (r1 & (c2 == 1)) | (r2 & (c1 == 1))
    if self.check_object:  # approach: nothing may touch the object yet (object at rest pose)
      bad |= (r1 & (c2 == 2)) | (r2 & (c1 == 2))
    return bool(np.any(bad))

  def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = self._scratch
    d.qpos[self.sc.arm_qadr] = q
    mujoco.mj_kinematics(self.sc.model, d)
    return d.site_xpos[self.sc.site].copy(), d.xmat[self.sc.grip_body].reshape(3, 3).copy()

  def solve(self, pos, approach, closing, q_seed, multi: bool = False) -> np.ndarray:
    """IK on the soft-limited planning model. Warm start from q_seed first; if that
    misses the tolerances (or ``multi``), add random restarts and take the solution
    within tolerance that is NEAREST the seed (so the arm never swaps elbow/wrist
    branch mid-motion)."""
    m, cfg = self.plan_model, self.cfg

    def ok(r):
      return r.pos_err <= cfg.ik_pos_tol and r.approach_err_deg <= self.app_tol

    best_err = None
    r = ik.solve(m, q_seed, pos, approach=approach, closing=closing, iters=200)
    cands = [r] if ok(r) else []
    best_err = r
    if not cands or multi:
      rng = np.random.default_rng(self.v.seed % 100000 + len(self.ik_calls))
      for _ in range(6):
        q0 = rng.uniform(self.soft_lo * 0.8, self.soft_hi * 0.8)
        rr = ik.solve(m, q0, pos, approach=approach, closing=closing, iters=200)
        if ok(rr):
          cands.append(rr)
        elif rr.pos_err < best_err.pos_err:
          best_err = rr
    self.ik_calls.append(1)
    if not cands:
      raise PlanError("ik_unreachable",
                      f"target {np.round(pos, 3)} err {best_err.pos_err*1e3:.1f} mm "
                      f"{best_err.approach_err_deg:.1f} deg")
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
    cands.sort(key=lambda c: float(np.abs((c.q - q_seed) * w).max()))
    free = [c for c in cands if not self.collides(c.q)]
    if not free:
      raise PlanError("ik_unreachable", f"target {np.round(pos, 3)}: every IK solution "
                      "puts the arm into itself or the table")
    r = free[0]
    self.ik_worst["pos_mm"] = max(self.ik_worst["pos_mm"], r.pos_err * 1e3)
    self.ik_worst["app_deg"] = max(self.ik_worst["app_deg"], r.approach_err_deg)
    return r.q

  # --- trajectory primitives ------------------------------------------------------
  def joint_move(self, plan: Plan, q1: np.ndarray, g1: float, label: str,
                 min_time: float = 0.4) -> None:
    q0, g0 = plan.last()
    dq = np.abs(q1 - q0).max()
    T = max(1.875 * dq / (ARM_VMAX * PLAN_MARGIN),
            1.875 * abs(g1 - g0) / (JAW_VMAX * PLAN_MARGIN), min_time)
    n = int(math.ceil(T * FPS))
    s = min_jerk(n)
    plan.add(q0 + np.outer(s, q1 - q0), g0 + s * (g1 - g0), label)

  def hold(self, plan: Plan, seconds: float, label: str, g: float | None = None) -> None:
    q0, g0 = plan.last()
    n = max(1, int(round(seconds * FPS)))
    g1 = g0 if g is None else g
    plan.add(np.repeat(q0[None], n, 0), np.full(n, g1), label)

  def jaw_move(self, plan: Plan, g1: float, label: str) -> None:
    q0, g0 = plan.last()
    n = max(1, int(math.ceil(abs(g1 - g0) / (JAW_VMAX * PLAN_MARGIN) * FPS)))
    plan.add(np.repeat(q0[None], n, 0), g0 + (g1 - g0) * np.arange(1, n + 1) / n, label)

  def cart_path(self, plan: Plan, pts: np.ndarray, frames_of, label: str,
                min_dt: np.ndarray | None = None, speed: float = 0.04,
                to_site=None, retilt: bool = False) -> None:
    """Follow Cartesian targets ``pts`` (k, 3) with dense IK, then time-parametrise.

    frames_of(site) -> (approach, closing). ``to_site`` maps an object-centre target to
    a grasp-site target (when carrying). With ``retilt``, a waypoint the arm cannot
    reach with the current finger tilt is retried with other tilts; the new tilt is kept
    for the following waypoints (the object is held, so the finger tilt is free).
    """
    q, g = plan.last()
    self.g_now = g
    qs, sites = [], []
    for p in pts:
      site = to_site(p) if to_site else p
      try:
        q = self.solve(site, *frames_of(site), q)
      except PlanError:
        if not retilt:
          raise
        base, ok = self.tilt, False
        for t in sorted(RETILT_CANDIDATES, key=lambda t: abs(t - base)):
          if abs(t - base) < 1e-6:
            continue
          self.tilt = t
          site = to_site(p) if to_site else p
          try:
            q = self.solve(site, *frames_of(site), q)
            ok = True
            self.retilts += 1
            break
          except PlanError:
            continue
        if not ok:
          self.tilt = base
          raise
      qs.append(q)
      sites.append(site)
    qs = np.array(qs)
    prev_q = plan.last()[0]
    times = [0.0]
    p_prev = self.fk(prev_q)[0]
    for i, (p, qq) in enumerate(zip(sites, qs)):
      dt = max(np.linalg.norm(p - p_prev) / speed,
               np.abs(qq - prev_q).max() / (ARM_VMAX * PLAN_MARGIN), 1e-3)
      if min_dt is not None:
        dt = max(dt, float(min_dt[i]))
      times.append(times[-1] + dt)
      prev_q, p_prev = qq, p
    times = np.array(times)
    knots = np.concatenate([plan.last()[0][None], qs], axis=0)
    n = int(math.ceil(times[-1] * FPS))
    t = np.arange(1, n + 1) * DT
    t[-1] = times[-1]
    arm = np.stack([np.interp(t, times, knots[:, j]) for j in range(5)], axis=1)
    plan.add(arm, np.full(n, g), label)

  # --- stage 1 --------------------------------------------------------------------
  def plan_pick(self, q_start: np.ndarray, g_start: float, obj_pos: np.ndarray,
                obj_yaw: float) -> tuple[Plan, dict]:
    cfg = self.cfg
    plan = Plan()
    plan.add(q_start[None], np.array([g_start]), "start")
    pick_az = azimuth(obj_pos)
    # Natural closing axis with the arm panned towards the object.
    q_nat = q_start.copy()
    q_nat[0] = np.clip(pick_az, self.soft_lo[0], self.soft_hi[0])
    _, R_nat = self.fk(q_nat)
    c_nat = -R_nat[:, 0]
    # Cube faces: try the 4 face normals, nearest to the natural closing axis first.
    normals = [np.array([math.cos(obj_yaw + k * math.pi / 2),
                         math.sin(obj_yaw + k * math.pi / 2), 0.0]) for k in range(4)]
    normals.sort(key=lambda n: -float(np.dot(n, c_nat)))
    # Candidate finger tilts: the spec's approach side first, then vertical, then
    # outward / inward 15 deg. Candidate faces: the two nearest the natural closing axis.
    t_spec = self.spec_tilt(pick_az)
    tilts = [t_spec] + [t for t in (0.0, math.radians(15), -math.radians(15))
                        if abs(t - t_spec) > 1e-3]
    last_err = None
    for tilt in tilts:
      for closing in normals[:2]:
        self.tilt = tilt
        try:
          plan, info = self._plan_pick_with(plan, q_start, obj_pos, closing, pick_az)
          info["tilt_deg"] = math.degrees(tilt)
          info["spec_tilt_deg"] = math.degrees(t_spec)
          return plan, info
        except PlanError as e:  # next candidate
          last_err = e
          plan = Plan()
          plan.add(q_start[None], np.array([g_start]), "start")
    raise last_err

  def _plan_pick_with(self, plan: Plan, q_start, obj_pos, closing, pick_az):
    cfg = self.cfg
    # Put the fixed jaw's inner face (mid-pad point, gripper frame) cfg clearance
    # outside the object's face, using the actual (possibly tilted) gripper rotation.
    R = ik.gripper_rotation(self.approach_at(obj_pos, pick_az), closing)
    pad = np.array([rm.FIXED_JAW_INNER_X, 0.0, -0.098])
    v = R @ (np.array(rm.GRASP_SITE_POS) - pad)  # site relative to the pad point
    F = obj_pos + closing * (self.half_w + cfg.fixed_jaw_clearance)
    grasp = np.array([F[0] + v[0], F[1] + v[1], self.grasp_z])
    hover = grasp + np.array([0, 0, cfg.hover_height])
    pre = grasp + np.array([0, 0, cfg.preapproach_height])
    h = self.spec_dir[:2]
    side_used = bool(np.linalg.norm(h) > 0.2)
    if side_used:
      pre[:2] -= h / np.linalg.norm(h) * cfg.side_offset
    rel_az = lambda p: azimuth(p) - pick_az  # noqa: E731

    def frames(p):
      return self.approach_at(p, pick_az), rot_z(rel_az(p)) @ closing

    try:
      q_pre = self.solve(pre, *frames(pre), q_start, multi=True)
    except PlanError:
      if not side_used:
        raise
      side_used = False
      pre = grasp + np.array([0, 0, cfg.preapproach_height])
      q_pre = self.solve(pre, *frames(pre), q_start, multi=True)
    self.joint_move(plan, q_pre, self.g_open, "approach_transit")
    self.check_object = True
    try:
      self.cart_path(plan, _line(pre, hover, 0.005), frames, "approach_line",
                     speed=cfg.line_speed)
      if cfg.hover_pause_s > 0:  # a hesitation at the hover point (same phase label)
        self.hold(plan, cfg.hover_pause_s, "approach_line")
      self.cart_path(plan, _line(hover, grasp, 0.004), frames, "descend",
                     speed=cfg.descend_speed)
    finally:
      self.check_object = False
    self.hold(plan, cfg.pre_close_s, "pre_close")
    self.jaw_move(plan, self.g_close, "close")
    self.hold(plan, 0.35, "squeeze")
    info = {"closing": closing.tolist(), "approach": self.approach_at(grasp, pick_az).tolist(),
            "pick_az": pick_az,
            "grasp_site": grasp.tolist(), "approach_side_used": side_used,
            "g_open": self.g_open, "g_close": self.g_close, "g_release": self.g_release}
    return plan, info

  # --- stages 2 and 3: re-planned from the simulated state ----------------------
  def attach_object(self, pick_info: dict, obj_pos, site_pos, R_grip) -> None:
    """Record where the object actually sits in the gripper frame (measured in sim)."""
    self.pick_az = pick_info["pick_az"]
    self.closing0 = np.array(pick_info["closing"])
    self.off_local = R_grip.T @ (np.asarray(obj_pos) - np.asarray(site_pos))
    # While carrying, the finger axis may deviate from the planned one (the object is
    # held; physics checks the result). Position stays strict.
    self.app_tol = self.cfg.carry_app_tol_deg

  def frames(self, p_site):
    a = self.approach_at(p_site, self.pick_az)
    c = rot_z(azimuth(p_site) - self.pick_az) @ self.closing0
    return a, c

  def site_for(self, p_obj):
    """Grasp-site target that puts the held object's centre at p_obj."""
    p_obj = np.asarray(p_obj, float)
    p = p_obj.copy()
    for _ in range(2):  # the gripper rotation depends on the site position
      a, c = self.frames(p)
      p = p_obj - ik.gripper_rotation(a, c) @ self.off_local
    return p

  def carry_height(self) -> float:
    rim = mj_scene.BIN_CLEARANCE + mj_scene.BIN_FLOOR_THICKNESS + mj_scene.BIN_WALL_HEIGHT
    return min(rim + self.cfg.wall_clearance + self.half, self.cfg.carry_z_max)

  def plan_lift(self, plan: Plan, obj_pos: np.ndarray) -> dict:
    """Vertical lift until the object's bottom is cfg.lift_clearance above the table."""
    self.pick_rest = np.asarray(obj_pos, float).copy()
    A = np.array([obj_pos[0], obj_pos[1], self.half + self.cfg.lift_clearance])
    self.cart_path(plan, _line(np.asarray(obj_pos, float), A, 0.004), self.frames, "lift",
                   speed=self.cfg.lift_speed, to_site=self.site_for, retilt=True)
    self.A = A
    return {"lift_to": A.tolist(), "obj_in_gripper_mm": (self.off_local * 1e3).round(2).tolist()}

  def plan_carry(self, plan: Plan, bin_xy: np.ndarray) -> dict:
    """Carry along the mapped human object path. If the human's lift arc goes above
    what the arm can reach, shrink the arc (shape kept, height scaled) and retry."""
    cfg = self.cfg
    B = np.array([bin_xy[0], bin_xy[1], self.carry_height()])
    P1 = np.array([bin_xy[0], bin_xy[1], mj_scene.bin_floor_top() + self.half])
    err, tilt0 = None, self.tilt
    for arc in cfg.arc_scales:
      path, dt_h, shape = self.mapped_human_path(self.pick_rest, P1, self.A, B,
                                                 cfg.carry_z_max, arc_scale=arc)
      trial = Plan(arm=list(plan.arm), jaw=list(plan.jaw), phase=list(plan.phase))
      self.tilt = tilt0
      try:
        self.cart_path(trial, path, self.frames, "carry", min_dt=dt_h, speed=0.15,
                       to_site=self.site_for, retilt=True)
      except PlanError as e:
        err = e
        continue
      plan.arm, plan.jaw, plan.phase = trial.arm, trial.jaw, trial.phase
      shape["arc_scale"] = arc
      shape["tilt_after_carry_deg"] = math.degrees(self.tilt)
      return shape
    raise err

  def plan_release(self, plan: Plan, obj_pos, site_pos, R_grip, bin_xy, home,
                   R_obj: np.ndarray | None = None) -> dict:
    """Stage 3, re-planned at the end of the carry from the measured in-hand pose:
    re-centre over the bin, lower until the object's LOWEST point (from its measured
    orientation; a cube may have turned in the jaws) is release_gap above the floor,
    open, retreat, go home."""
    cfg = self.cfg
    self.off_local = R_grip.T @ (np.asarray(obj_pos) - np.asarray(site_pos))
    below = self.half if R_obj is None else float(np.abs(np.asarray(R_obj)[2, :]).sum() * self.half)
    release_z = mj_scene.bin_floor_top() + below + cfg.release_gap
    obj_pos = np.asarray(obj_pos, float)
    top = np.array([bin_xy[0], bin_xy[1], max(float(obj_pos[2]), self.carry_height())])
    R_pt = np.array([bin_xy[0], bin_xy[1], release_z])
    pts = np.concatenate([_line(obj_pos, top, 0.004), _line(top, R_pt, 0.004)])
    self.cart_path(plan, pts, self.frames, "lower", speed=cfg.lower_speed,
                   to_site=self.site_for, retilt=True)
    self.hold(plan, cfg.pre_release_s, "pre_release")
    # Open just enough to let go (mouth = object + release_mouth_margin): the moving
    # jaw swings outwards as it opens and would otherwise hit the bin wall.
    self.jaw_move(plan, self.g_release, "release")
    self.hold(plan, cfg.post_release_s, "post_release")
    s_rel = self.site_for(R_pt)
    up = s_rel + np.array([0, 0, cfg.retreat_height])
    self.cart_path(plan, _line(s_rel, up, 0.005), self.frames, "retreat",
                   speed=cfg.line_speed, retilt=True)
    home = np.asarray(home, float).copy()
    home[4] = plan.last()[0][4]  # "home" for the arm; the wrist roll stays where it is
    self.joint_move(plan, home, 0.3, "return_home")
    self.hold(plan, 0.5, "rest")
    return {"release_z": release_z, "object_tilt_in_hand_deg": round(math.degrees(
              math.acos(min(1.0, abs(float(np.asarray(R_obj)[2, 2]))))), 1) if R_obj is not None else None,
            "obj_in_gripper_before_lower_mm": (self.off_local * 1e3).round(2).tolist()}

  def mapped_human_path(self, P0r: np.ndarray, P1r: np.ndarray, A: np.ndarray,
                        B: np.ndarray, z_max: float, arc_scale: float = 1.0):
    """Map the spec's object path (lift-off -> release) onto this episode.

    Same convention as the mjlab task's tracking reference: the human path's end points
    (lift-off, release) are moved onto this episode's pick (object at rest, P0r) and
    place (object resting on the bin floor, P1r). Concretely the path is split into its
    chord and its deviation from the chord; the chord becomes P0r -> P1r, the xy
    deviation is rotated and scaled like the chord (2-D similarity) and the z deviation
    is kept in metres (the human's lift arc). Then three safety floors/ceilings, each
    applied only where needed and logged: z >= the lift height A (object clear of the
    table), z ramps up to the carry height B near the bin (clear the walls), and the
    measured reach ceiling. Finally it is smoothed and densely resampled (<= 4 mm), with
    the demo timing x robot_time_scale as a lower bound on each step's duration. The
    path starts at A (end of the vertical lift) and ends at B (above the bin centre).
    """
    traj = self.spec.trajectory
    half = self.half
    zone = mj_scene.BIN_INNER_HALF + mj_scene.BIN_WALL_THICKNESS + half + 0.015

    def floor_z(xy: np.ndarray) -> np.ndarray:
      d = np.max(np.abs(xy - B[:2]), axis=1)  # Chebyshev distance to the bin centre
      w = np.clip((zone + 0.04 - d) / 0.04, 0.0, 1.0)
      w = w * w * (3 - 2 * w)  # smoothstep
      return A[2] + (B[2] - A[2]) * w

    if traj is None or len(traj.waypoints) < 3:
      n = 30
      s = np.linspace(0, 1, n)
      path = P0r + np.outer(s, P1r - P0r)
      path[:, 2] += 0.06 * arc_scale * np.sin(np.pi * s)
      path[:, 2] = np.maximum(path[:, 2], floor_z(path[:, :2]))
      path[0], path[-1] = A, B
      return _densify(path, np.full(n, 0.05), 0.004) + ({"source": "synthetic_arc"},)
    t, pos, g, r = traj.reference()
    ig = int(np.searchsorted(t, g))
    ir = min(int(np.searchsorted(t, r)), len(t) - 1)
    z0 = pos[ig, 2]
    il = ig
    while il < ir and pos[il, 2] < z0 + 0.003:
      il += 1
    il = max(ig, il - 1)
    seg, ts = pos[il:ir + 1], t[il:ir + 1]
    if len(seg) < 3:
      seg, ts = pos[ig:ir + 1], t[ig:ir + 1]
    s = (ts - ts[0]) / max(ts[-1] - ts[0], 1e-9)
    H0, H1 = seg[0], seg[-1]
    dev = seg - (H0 + np.outer(s, H1 - H0))
    ch, cr = (H1 - H0)[:2], (P1r - P0r)[:2]
    nh, nr = np.linalg.norm(ch), np.linalg.norm(cr)
    if nh > 1e-3:
      rot = math.atan2(cr[1], cr[0]) - math.atan2(ch[1], ch[0])
      scale = float(np.clip(nr / nh, 0.5, 2.0))
    else:
      rot, scale = 0.0, 1.0
    Rz = rot_z(rot)[:2, :2]
    dev_xy = (dev[:, :2] @ Rz.T) * scale
    path = P0r + np.outer(s, P1r - P0r)
    path[:, :2] += dev_xy
    path[:, 2] += dev[:, 2] * arc_scale
    fz = floor_z(path[:, :2])
    raised = int(np.sum(path[:, 2] < fz - 1e-4))
    path[:, 2] = np.minimum(np.maximum(path[:, 2], fz), max(z_max, B[2]))
    path[0], path[-1] = A, B
    # Per-radius reach ceiling (measured workspace table, 1.5 cm margin, object centre
    # sits at most ~1 cm below the site): flatten only where the arm cannot follow.
    r_xy = np.linalg.norm(path[:, :2] - PAN_AXIS, axis=1)
    ceil_z = np.interp(np.maximum(r_xy, self.cfg.min_carry_radius), REACH_R, REACH_Z) - 0.015
    fz = floor_z(path[:, :2])
    clipped_by_reach = int(np.sum(path[:, 2] > np.maximum(ceil_z, fz)))
    path[:, 2] = np.minimum(path[:, 2], np.maximum(ceil_z, fz))
    # Keep the carried object out of the arm's own footprint: points closer than
    # min_carry_radius to the pan axis are pushed radially outward (the arc bends round
    # the base, as a person would route it for this robot).
    rel = path[:, :2] - PAN_AXIS
    r = np.linalg.norm(rel, axis=1)
    # (never closer than the pick or place themselves, so the end points stay put)
    rmin = min(self.cfg.min_carry_radius, float(np.linalg.norm(A[:2] - PAN_AXIS)),
               float(np.linalg.norm(B[:2] - PAN_AXIS)))
    pushed = int(np.sum(r < rmin))
    if pushed:
      out = PAN_AXIS + rel / np.maximum(r, 1e-9)[:, None] * rmin
      path[:, :2] = np.where((r < rmin)[:, None], out, path[:, :2])
      r_xy = np.linalg.norm(path[:, :2] - PAN_AXIS, axis=1)
      ceil_z = np.interp(r_xy, REACH_R, REACH_Z) - 0.015
      path[:, 2] = np.minimum(path[:, 2], np.maximum(ceil_z, floor_z(path[:, :2])))
    # light extra smoothing (ends pinned)
    if len(path) >= 5:
      k = np.ones(3) / 3
      for j in range(3):
        sm = np.convolve(np.pad(path[:, j], 1, mode="edge"), k, mode="valid")
        sm[0], sm[-1] = path[0, j], path[-1, j]
        path[:, j] = sm
    dt = np.diff(ts, prepend=ts[0])
    dt[0] = 0.0
    shape = {"source": "spec_object_trajectory", "human_frames": int(len(seg)),
             "human_seconds": float(ts[-1] - ts[0]),
             "robot_time_scale": float(traj.robot_time_scale),
             "rotation_deg": math.degrees(rot), "xy_scale": scale,
             "apex_above_chord_mm": float(dev[:, 2].max() * 1e3),
             "points_pushed_out_of_base_radius": pushed,
             "points_lowered_to_reach_ceiling": clipped_by_reach,
             "points_raised_for_table_or_bin_clearance": raised}
    return _densify(path, dt, 0.004) + (shape,)


def _line(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
  n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
  s = np.arange(1, n + 1) / n
  return a + np.outer(s, b - a)


def _densify(path: np.ndarray, dt_demo: np.ndarray, step: float):
  """Resample so consecutive points are <= step apart; spread each original segment's
  demo-clock duration over its sub-steps. Returns (points (k,3), min_dt (k,))."""
  pts, dts = [path[0]], [0.0]
  for i in range(1, len(path)):
    seg = path[i] - path[i - 1]
    n = max(1, int(math.ceil(np.linalg.norm(seg) / step)))
    for k in range(1, n + 1):
      pts.append(path[i - 1] + seg * k / n)
      dts.append(dt_demo[i] / n)
  return np.array(pts[1:]), np.array(dts[1:])


def rate_limit(targets: np.ndarray, start: np.ndarray, frac: float = 0.98) -> np.ndarray:
  """Clip every frame-to-frame change to frac x the servo speed caps (arm 0.5, jaw 2.0
  rad/s). The plans run at 90 % of the cap, so this only trims smoothing overshoot at
  stage seams; it guarantees the recorded actions respect the caps."""
  out = np.empty_like(targets)
  cap = np.array([ARM_VMAX] * 5 + [JAW_VMAX]) * frac * DT
  prev = np.asarray(start, float)
  for k in range(len(targets)):
    prev = prev + np.clip(targets[k] - prev, -cap, cap)
    out[k] = prev
  return out


def smooth_targets(targets: np.ndarray, window: int) -> np.ndarray:
  """Centred moving average on the arm columns (ends held). Never raises a speed."""
  if window < 2 or len(targets) < window:
    return targets
  out = targets.copy()
  k = np.ones(window) / window
  pad = (window // 2, window - 1 - window // 2)
  for j in range(5):
    out[:, j] = np.convolve(np.pad(targets[:, j], pad, mode="edge"), k, mode="valid")
  return out
