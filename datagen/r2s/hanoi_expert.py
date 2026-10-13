"""Scripted expert for one Tower-of-Hanoi move, and the episode runners built on it.

One move = pick the top cube of the source spot and place it on the destination spot (on
the table, or on top of a larger cube). The expert reuses the pick-and-place planner
(planner.Expert: IK, collision-checked waypoints, speed caps, finger re-tilt) and changes
only what stacking needs:

  grasp height   the grasp site goes GRASP_ABOVE_CENTRE (5-12 mm, per cube) above the
                 cube's centre, so the jaw pads (which reach 12 mm below to 2 mm above the
                 site) hold it at or just above the centre of mass; but never so low
                 that the jaw tips come within 2 mm of whatever the cube stands on (the
                 table or the larger cube below it)
  approach       straight down from above the cube (no sideways offset towards the human's
                 approach side: there are other stacks next to it); pre-approach and hover
                 heights capped by the arm's measured reach ceiling
  lift           vertical, until the cube's bottom is CARRY_CLEARANCE above the tallest
                 other stack
  carry          along an arc around the shoulder-pan axis at that height (the spots lie on
                 such an arc), to above the destination
  place          re-planned from the cube's MEASURED pose in the jaws: centre over the
                 destination (the top cube there, or the spot centre), lower until the cube's
                 lowest corner is SLOW_ZONE above the landing surface, then slowly
                 (SLOW_LOWER_SPEED) until it is RELEASE_GAP above it, open, retreat
                 upwards, go to the Hanoi ready pose, rest
  ready pose     HANOI_HOME_Q, higher than the pick-and-place home so the jaws clear a
                 three-cube stack (105 mm) under it, with the start jitter of --variety too

Physics honesty is the pick-and-place rule: the expert only writes servo targets
(data.ctrl). Cubes are placed in the model before the episode and settle (unrecorded);
after that nothing writes a position or velocity, and the model has no weld, equality
constraint or mocap body.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time

import mujoco
import numpy as np

from . import hanoi as H
from . import scene as S
from .hanoi_verify import HanoiMonitor, cube_body_ids, read_state
from .paths import TaskSpec, rm
from .planner import (REACH_R, REACH_Z, Expert, Plan, PlanConfig, PlanError, _line, azimuth,
                      plan_config_for, rate_limit, smooth_targets)
from .rollout import EpisodeResult, Recorder, _jsonable, _run_frames, _speed_ok
from .variations import HOME_Q, PAN_AXIS, Variation

# Grasp-site height above the cube's centre, per cube. The jaw pads reach 12 mm below to
# 2 mm above the site, so 5-12 mm puts the contact patch on or just above the centre of
# mass. Measured (8 large-cube moves each): at 0 / 5 mm the upper moving jaw (mesh1)
# bites the large cube's top edge, 2.1-3.0 mm deep (penetration failure 8/8, 4/8); at
# 10-15 mm 0/8 fail. Medium and small cubes pass at 0-10 mm.
GRASP_ABOVE_CENTRE = {"L": 0.012, "M": 0.008, "S": 0.005}
CARRY_CLEARANCE = 0.020
RELEASE_GAP = 0.003
SLOW_ZONE = 0.020
SLOW_LOWER_SPEED = 0.012
MIN_LIFT = 0.030  # the verifier wants >= 2 cm; the in-hand offset can eat a few mm
REACH_MARGIN = 0.005
SETTLE_S = 0.6  # unrecorded settle of the stacks before the episode starts
TAIL_S = 1.0  # recorded, after the expert's 0.5 s rest: the settled end of the move
HANOI_TILTS_DEG = (15.0, 20.0, 10.0, 0.0)  # fingertips outward; tried in this order
# Hanoi ready pose (rad): wrist camera looking at the spots, lowest jaw point ~0.17 m up.
HANOI_HOME_Q = np.array([0.0, -0.9, 0.15, 1.45, -1.55])


def reach_ceiling(xy) -> float:
  """Highest grasp-site height with near-vertical fingers at this xy (measured table)."""
  r = float(np.linalg.norm(np.asarray(xy, float)[:2] - PAN_AXIS))
  return float(np.interp(r, REACH_R, REACH_Z)) - REACH_MARGIN


def hanoi_start_pose(v: Variation) -> np.ndarray:
  """The episode's start pose: the drawn start jitter around HANOI_HOME_Q instead of
  around the pick-and-place home (same draw, so every other number is unchanged)."""
  return HANOI_HOME_Q + (np.asarray(v.q_start, float) - HOME_Q)


def cube_top(d, body: int, half: float) -> float:
  R = d.xmat[body].reshape(3, 3)
  return float(d.xpos[body, 2]) + half * float(np.abs(R[2, :]).sum())


@dataclasses.dataclass
class MoveContext:
  """Everything the expert needs about one move, measured from the simulation."""
  state: str
  move: tuple[str, str]
  cube: str
  obj_pos: np.ndarray  # settled centre of the cube to move
  obj_yaw: float
  surface_below: float  # top of what it stands on (0 = table)
  dest_xy: np.ndarray  # where its centre should end up
  dest_surface: float  # top of the destination stack (0 = empty spot)
  obstacle_top: float  # top of the tallest other cube anywhere


def move_context(sc: S.EpisodeScene, v: Variation, ids: dict[str, int], state: str,
                 move: tuple[str, str]) -> MoveContext:
  d = sc.data
  cube = H.moved_cube(state, move)
  stacks = H.stacks(state)
  half = {c: H.SIZES[c] / 2 for c in H.CUBES}
  under = [c for c in stacks[move[0]] if c != cube]
  surface_below = max((cube_top(d, ids[c], half[c]) for c in under), default=0.0)
  dest = stacks[move[1]]
  if dest:
    top = dest[-1]
    dest_xy = d.xpos[ids[top], :2].copy()
    dest_surface = cube_top(d, ids[top], half[top])
  else:
    dest_xy = np.asarray(H.hanoi_info(v)["spots"][move[1]], float)
    dest_surface = 0.0
  obstacle_top = max(cube_top(d, ids[c], half[c]) for c in H.CUBES if c != cube)
  q = d.xquat[ids[cube]]
  return MoveContext(state=state, move=tuple(move), cube=cube, obj_pos=d.xpos[ids[cube]].copy(),
                     obj_yaw=2 * math.atan2(q[3], q[0]), surface_below=surface_below,
                     dest_xy=dest_xy, dest_surface=dest_surface, obstacle_top=obstacle_top)


def scene_for_cube(sc: S.EpisodeScene, ids: dict[str, int], cube: str) -> S.EpisodeScene:
  """A view of the same scene in which ``cube`` is the planner's object and the other two
  are extra bodies (only the planner's bookkeeping changes; same model and data)."""
  m = sc.model
  cat = sc.geom_cat.copy()
  for c, b in ids.items():
    cat[int(m.body_geomadr[b])] = S.CAT_OBJECT if c == cube else S.CAT_EXTRA
  b = ids[cube]
  j = int(m.body_jntadr[b])
  return dataclasses.replace(sc, obj_body=b, obj_qadr=int(m.jnt_qposadr[j]),
                             obj_dadr=int(m.jnt_dofadr[j]), geom_cat=cat,
                             extra_bodies=tuple(ids[c] for c in H.CUBES if c != cube))


class HanoiExpert(Expert):
  """planner.Expert with the stacking changes listed in the module docstring.

  Finger tilt. A 5-joint arm can tilt its fingers only in its own vertical (radial)
  plane. With vertical fingers the grasp site reaches only ~85 mm up at the spots'
  radius (measured: IK within 3 mm / 5 deg), below the top cube of a three-cube stack
  (site ~97 mm) and the carry heights; tilting the fingertips 15 deg outward raises
  that to ~125 mm. A radial tilt keeps the jaw pads vertical only if the jaws close
  TANGENTIALLY (perpendicular to the tilt plane), so the expert grasps the cube by its
  pair of faces closest to tangential (cube yaws are drawn within +-10 deg of the spot's
  radial direction, so that pair is never more than 10 deg off). Otherwise the squeeze
  turns the cube by the tilt angle and it lands crooked (seen: 14 deg in the jaws).
  """
  def __init__(self, sc: S.EpisodeScene, spec: TaskSpec, v: Variation, cfg: PlanConfig,
               ctx: MoveContext, pinch_width: float | None = None):
    cfg = copy.copy(cfg)
    cfg.side_offset = 0.0  # straight down: other stacks stand next to this one
    super().__init__(sc, spec, v, cfg, pinch_width=pinch_width)
    self.ctx = ctx
    # Collision checks see every cube where it is NOW (not at the model's initial pose).
    self._coll.qpos[:] = sc.data.qpos
    self.grasp_z = max(float(ctx.obj_pos[2]) + GRASP_ABOVE_CENTRE[ctx.cube],
                       ctx.surface_below + rm.JAW_TIP_BELOW_SITE + cfg.tip_clearance)
    room = max(reach_ceiling(ctx.obj_pos) - self.grasp_z, 0.015)
    self.cfg.preapproach_height = min(cfg.preapproach_height, room)
    self.cfg.hover_height = min(cfg.hover_height, self.cfg.preapproach_height)

  def plan_pick(self, q_start: np.ndarray, g_start: float, obj_pos: np.ndarray,
                obj_yaw: float) -> tuple[Plan, dict]:
    pick_az = azimuth(obj_pos)
    tangential = np.array([-math.sin(pick_az), math.cos(pick_az), 0.0])
    q_nat = np.asarray(q_start, float).copy()
    q_nat[0] = np.clip(pick_az, self.soft_lo[0], self.soft_hi[0])
    c_nat = -self.fk(q_nat)[1][:, 0]
    normals = [np.array([math.cos(obj_yaw + k * math.pi / 2), math.sin(obj_yaw + k * math.pi / 2), 0.0])
               for k in range(4)]
    # the two normals of the face pair closest to tangential; the one needing less wrist
    # roll first
    normals.sort(key=lambda n: (-round(abs(float(n @ tangential)), 6), -float(n @ c_nat)))
    last_err = None
    for tilt_deg in HANOI_TILTS_DEG:
      for closing in normals[:2]:
        self.tilt = math.radians(tilt_deg)
        plan = Plan()
        plan.add(np.asarray(q_start, float)[None], np.array([g_start]), "start")
        try:
          plan, info = self._plan_pick_with(plan, q_start, obj_pos, closing, pick_az)
        except PlanError as e:
          last_err = e
          continue
        finally:
          self.check_object = False
        info["tilt_deg"] = tilt_deg
        info["closing_vs_tangential_deg"] = round(math.degrees(math.acos(min(1.0, abs(
          float(closing @ tangential))))), 1)
        return plan, info
    raise last_err

  def carry_z(self) -> float:
    """Object-centre height while carrying: its bottom CARRY_CLEARANCE above every other
    cube (the cube's own source stack included)."""
    return max(self.ctx.obstacle_top + CARRY_CLEARANCE + self.half,
               float(self.ctx.obj_pos[2]) + MIN_LIFT)

  def plan_lift(self, plan: Plan, obj_pos: np.ndarray) -> dict:
    obj_pos = np.asarray(obj_pos, float)
    top = np.array([obj_pos[0], obj_pos[1], self.carry_z()])
    self.cart_path(plan, _line(obj_pos, top, 0.004), self.frames, "lift",
                   speed=self.cfg.lift_speed, to_site=self.site_for, retilt=True)
    return {"lift_to": top.tolist(), "obj_in_gripper_mm": (self.off_local * 1e3).round(2).tolist(),
            "grasp_site_z": self.grasp_z}

  def plan_carry(self, plan: Plan, obj_pos: np.ndarray) -> dict:
    """Arc around the pan axis (radius and azimuth interpolated) at the carry height."""
    z = max(self.carry_z(), float(obj_pos[2]))
    a = np.asarray(obj_pos[:2], float) - PAN_AXIS
    b = np.asarray(self.ctx.dest_xy, float) - PAN_AXIS
    r0, r1 = np.linalg.norm(a), np.linalg.norm(b)
    t0, t1 = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
    arc_len = max(abs(t1 - t0) * max(r0, r1), abs(r1 - r0))
    n = max(2, int(math.ceil(arc_len / 0.004)))
    s = np.arange(1, n + 1) / n
    r = r0 + s * (r1 - r0)
    t = t0 + s * (t1 - t0)
    pts = np.stack([PAN_AXIS[0] + r * np.cos(t), PAN_AXIS[1] + r * np.sin(t), np.full(n, z)], 1)
    self.cart_path(plan, pts, self.frames, "carry", speed=self.cfg.line_speed,
                   to_site=self.site_for, retilt=True)
    return {"carry_z": z, "arc_deg": round(math.degrees(t1 - t0), 1),
            "tilt_after_carry_deg": round(math.degrees(self.tilt), 1)}

  def plan_place(self, plan: Plan, obj_pos, site_pos, R_grip, R_obj, home) -> dict:
    """Re-planned from the measured in-hand pose at the end of the carry."""
    cfg, ctx = self.cfg, self.ctx
    self.off_local = R_grip.T @ (np.asarray(obj_pos) - np.asarray(site_pos))
    below = float(np.abs(np.asarray(R_obj)[2, :]).sum() * self.half)  # centre to lowest corner
    release_z = ctx.dest_surface + below + RELEASE_GAP
    obj_pos = np.asarray(obj_pos, float)
    dx, dy = ctx.dest_xy
    top = np.array([dx, dy, max(float(obj_pos[2]), self.carry_z())])
    slow = np.array([dx, dy, release_z + SLOW_ZONE])
    rel = np.array([dx, dy, release_z])
    self.cart_path(plan, np.concatenate([_line(obj_pos, top, 0.004), _line(top, slow, 0.004)]),
                   self.frames, "lower", speed=cfg.lower_speed, to_site=self.site_for, retilt=True)
    self.cart_path(plan, _line(slow, rel, 0.002), self.frames, "lower", speed=SLOW_LOWER_SPEED,
                   to_site=self.site_for, retilt=True)
    self.hold(plan, cfg.pre_release_s, "pre_release")
    self.jaw_move(plan, self.g_release, "release")
    self.hold(plan, cfg.post_release_s, "post_release")
    s_rel = self.site_for(rel)
    rise = min(cfg.retreat_height, max(reach_ceiling(s_rel) - float(s_rel[2]), 0.02))
    self.cart_path(plan, _line(s_rel, s_rel + np.array([0, 0, rise]), 0.005), self.frames,
                   "retreat", speed=cfg.line_speed, retilt=True)
    home = np.asarray(home, float).copy()
    home[4] = plan.last()[0][4]  # the wrist roll stays where it is
    self.joint_move(plan, home, 0.3, "return_home")
    self.hold(plan, 0.5, "rest")
    return {"release_z": release_z, "dest_surface": ctx.dest_surface,
            "object_tilt_in_hand_deg": round(math.degrees(math.acos(min(1.0, float(
              np.max(np.abs(np.asarray(R_obj)[2, :])))))), 1),
            "obj_in_gripper_before_lower_mm": (self.off_local * 1e3).round(2).tolist()}


class _MonitorTap:
  """Lets rollout._run_frames drive the HanoiMonitor (it expects the expert Monitor's
  interface). The monitor can be swapped between moves (full-puzzle runs)."""

  def __init__(self, sim: "MoveSim"):
    self.sim = sim
    self.max_track_err = 0.0
    self.prev_still = np.zeros(5, bool)
    self.c = {"unstable": 0}

  def step(self, phase, holding=None) -> None:
    self.sim.mon.step()
    self.c["unstable"] = self.sim.mon.base.c["unstable"]


class _TapRecorder(Recorder):
  def __init__(self, sc, sim: "MoveSim"):
    super().__init__(sc)
    self.sim = sim

  def frame(self, target, phase) -> None:
    super().frame(target, phase)
    self.sim.mon.frame_start(target)


class MoveSim:
  """One continuous simulated scene in which the expert makes moves. Recording runs
  through every move; the HanoiMonitor is renewed per move (``new_move``)."""

  def __init__(self, sc: S.EpisodeScene, v: Variation, servo_foh: bool = True):
    self.sc, self.v, self.servo_foh = sc, v, servo_foh
    self.ids = cube_body_ids(sc, v)
    self.mon = HanoiMonitor(sc, v)
    self.tap = _MonitorTap(self)
    self.rec = _TapRecorder(sc, self)
    self.speed_ok = True
    self.speed_max = [0.0, 0.0]

  def new_move(self) -> None:
    self.mon = HanoiMonitor(self.sc, self.v)

  def positions(self) -> dict[str, np.ndarray]:
    return {c: self.sc.data.xpos[b].copy() for c, b in self.ids.items()}

  def unstable(self) -> bool:
    return bool(self.mon.base.c["unstable"])

  def run(self, targets: np.ndarray, phases: list[str], prev: np.ndarray | None) -> None:
    ok, a, j = _speed_ok(targets, prev)
    self.speed_ok &= ok
    self.speed_max = [max(self.speed_max[0], a), max(self.speed_max[1], j)]
    _run_frames(self.sc, self.tap, self.rec, targets, phases, self.servo_foh)

  def run_stage(self, plan_fn, cfg: PlanConfig, q_start=None, g_start=None) -> dict:
    """Plan from the last applied target (or the start pose before the first frame),
    smooth across the seam, enforce the speed caps, simulate."""
    if not self.rec.action:
      p = Plan()
      p.add(np.asarray(q_start, float)[None], np.array([g_start]), "start")
      info = plan_fn(p)
      t, ph = p.arrays()
      t = rate_limit(smooth_targets(t, cfg.smooth_window), t[0])
      self.run(t, ph, None)
      return info
    last = np.asarray(self.rec.action[-1], float)
    p = Plan()
    p.add(last[None, :5], np.array([last[5]]), "anchor")
    info = plan_fn(p)
    t, ph = p.arrays()
    t, ph = t[1:], ph[1:]
    prev = np.asarray(self.rec.action[-3:], float)
    t = smooth_targets(np.concatenate([prev, t], 0), cfg.smooth_window)[len(prev):]
    t = rate_limit(t, last)
    self.run(t, ph, last)
    return info

  def hold(self, seconds: float, label: str = "tail") -> None:
    last = np.asarray(self.rec.action[-1], float)
    n = int(round(seconds * S.FPS))
    self.run(np.repeat(last[None], n, 0), [label] * n, last)


def place_and_settle(sc: S.EpisodeScene, v: Variation, settle_s: float = SETTLE_S) -> None:
  """The only state write of an episode: arm at its start pose, then let every cube settle
  (unrecorded). Cubes start just above their resting heights (hanoi.START_GAP)."""
  m, d = sc.model, sc.data
  assert m.neq == 0 and m.nmocap == 0, "model must have no equality constraints or mocap"
  mujoco.mj_resetData(m, d)
  q0 = hanoi_start_pose(v)
  d.qpos[sc.arm_qadr] = q0
  d.qpos[sc.g_qadr] = v.gripper_start
  d.ctrl[sc.arm_act] = q0
  d.ctrl[sc.g_act] = v.gripper_start
  mujoco.mj_forward(m, d)
  for _ in range(int(round(settle_s / m.opt.timestep))):
    mujoco.mj_step(m, d)
  d.time = 0.0


def expert_move(sim: MoveSim, spec: TaskSpec, v: Variation, state: str, move: tuple[str, str],
                cfg: PlanConfig, pinch_width: float | None = None) -> tuple[dict, list[str]]:
  """Plan and simulate one move from the current simulated state. Returns (plan info,
  planner failure reasons); raises nothing (a PlanError becomes a reason)."""
  sc, d = sim.sc, sim.sc.data
  ctx = move_context(sc, v, sim.ids, state, move)
  view = scene_for_cube(sc, sim.ids, ctx.cube)
  vm = H.variation_for_move(v, sim.positions(), state, move)
  ex = HanoiExpert(view, spec, vm, cfg, ctx, pinch_width=pinch_width)
  info: dict = {"move": list(move), "cube": ctx.cube, "surface_below": ctx.surface_below,
                "dest_surface": ctx.dest_surface, "obstacle_top": ctx.obstacle_top,
                "dest_xy": ctx.dest_xy.tolist()}
  fails: list[str] = []

  def grip_state():
    return (d.xpos[view.obj_body].copy(), d.site_xpos[view.site].copy(),
            d.xmat[view.grip_body].reshape(3, 3).copy())

  def jaws_on_cube() -> bool:
    g = int(sc.model.body_geomadr[view.obj_body])
    fixed = moving = False
    for i in range(d.ncon):
      a, b = int(d.contact.geom[i, 0]), int(d.contact.geom[i, 1])
      if g in (a, b):
        other = b if a == g else a
        fixed |= sc.geom_cat[other] == S.CAT_FIXED_JAW
        moving |= sc.geom_cat[other] == S.CAT_MOVING_JAW
    return fixed and moving

  try:
    if sim.rec.action:
      q0, g0 = np.asarray(sim.rec.action[-1][:5], float), float(sim.rec.action[-1][5])
    else:
      q0, g0 = hanoi_start_pose(v), float(v.gripper_start)
    holder = {}

    def pick(p):
      plan, pinfo = ex.plan_pick(p.last()[0], p.last()[1], ctx.obj_pos, ctx.obj_yaw)
      # plan_pick builds its own Plan starting with a "start" frame at the given pose;
      # move its frames (after that first one) onto p.
      a, ph = plan.arrays()
      p.arm.append(a[1:, :5])
      p.jaw.append(a[1:, 5])
      p.phase += ph[1:]
      holder["pick"] = pinfo
      return pinfo

    info["pick"] = sim.run_stage(pick, cfg, q_start=q0, g_start=g0)
    if sim.unstable():
      return info, ["unstable"]
    if not jaws_on_cube():
      fails.append("grasp_failed")
    ex.attach_object(holder["pick"], *grip_state())
    info["lift"] = sim.run_stage(lambda p: ex.plan_lift(p, d.xpos[view.obj_body].copy()), cfg)
    lifted = float(d.xpos[view.obj_body][2] - ctx.obj_pos[2])
    info["lift_check_mm"] = round(lifted * 1e3, 2)
    if not jaws_on_cube() or lifted < 0.01:
      return info, sorted(set(fails + ["grasp_failed"]))
    info["carry"] = sim.run_stage(lambda p: ex.plan_carry(p, d.xpos[view.obj_body].copy()), cfg)
    if sim.unstable():
      return info, ["unstable"]
    o, st, R = grip_state()
    R_obj = d.xmat[view.obj_body].reshape(3, 3).copy()
    info["place"] = sim.run_stage(lambda p: ex.plan_place(p, o, st, R, R_obj, HANOI_HOME_Q), cfg)
  except PlanError as e:
    info["error"] = str(e)
    return info, sorted(set(fails + [e.reason]))
  info["retilts"] = ex.retilts
  info["ik_worst"] = {k: round(x, 3) for k, x in ex.ik_worst.items()}
  if "lift" in info and "place" in info:
    a0 = np.array(info["lift"]["obj_in_gripper_mm"])
    a1 = np.array(info["place"]["obj_in_gripper_before_lower_mm"])
    info["in_hand_slip_mm"] = round(float(np.linalg.norm(a1 - a0)), 2)
  return info, fails


def run_hanoi_episode(spec: TaskSpec, v: Variation, cfg: PlanConfig | None = None,
                      pinch_width: float | None = None, render_size=(256, 256),
                      keep_arrays: bool = True, servo_foh: bool = True) -> EpisodeResult:
  """One single-move Hanoi episode (datagen): settle the drawn start state, let the expert
  make the solver's next move, record a settled tail, judge with the Hanoi verifier.
  accepted = L_full (and the planner found a plan)."""
  wall0 = time.perf_counter()
  h = H.hanoi_info(v)
  cfg = cfg or plan_config_for(v)
  sc = S.build_episode_scene(v, object_kind="box", render_size=render_size)
  place_and_settle(sc, v)
  sim = MoveSim(sc, v, servo_foh)
  state, move = h["start_state"], tuple(h["move"])
  plan_info, plan_fails = expert_move(sim, spec, v, state, move, cfg, pinch_width)
  if sim.rec.action and not sim.unstable():
    sim.hold(TAIL_S)
  verdict = sim.mon.verdict(h["end_state"], h["moved_cube"])
  if not sim.speed_ok and "speed_cap" not in verdict["reasons"]:
    verdict["reasons"].append("speed_cap")
    verdict["L_full"] = verdict["passed"] = False
  reasons = list(plan_fails) + [r for r in verdict["reasons"] if r not in plan_fails]
  accepted = bool(verdict["L_full"] and not plan_fails)
  stats = {"hanoi": {k: verdict[k] for k in ("L_task", "L_full", "reasons", "task_reasons",
                                             "style_only", "observed_end_state",
                                             "expected_end_state", "moved_cube")},
           **verdict["stats"],
           "planner_failures": plan_fails, "reason_times_s": verdict["reason_times_s"],
           "max_target_speed_arm_rad_s": round(sim.speed_max[0], 4),
           "max_target_speed_jaw_rad_s": round(sim.speed_max[1], 4)}
  res = EpisodeResult(
    attempt=v.attempt, accepted=accepted, reasons=reasons,
    primary_reason=reasons[0] if reasons else None, stats=_jsonable(stats),
    variation=v.to_dict(), plan_info=_jsonable(plan_info),
    sim_seconds=len(sim.rec.state) / S.FPS, wall_seconds=time.perf_counter() - wall0,
    phases=sim.rec.phase)
  if keep_arrays and sim.rec.state:
    res.state = np.array(sim.rec.state, np.float32)
    res.action = np.array(sim.rec.action, np.float32)
    res.env_state = np.array(sim.rec.env, np.float32)
    res.qpos_full = np.array(sim.rec.qfull, np.float64)
  return res


def run_hanoi_puzzle_expert(spec: TaskSpec, v: Variation, max_moves: int = 12,
                            pinch_width: float | None = None, servo_foh: bool = True) -> dict:
  """The whole puzzle with the scripted expert in ONE continuous scene (no resets between
  moves): read the state from the simulation, make the solver's next move, repeat until
  the goal or ``max_moves``. ``v`` should start at "AAA" (sample_hanoi_variation(...,
  start_state="AAA")). Returns per-move verdicts and the outcome."""
  wall0 = time.perf_counter()
  cfg = plan_config_for(v)
  sc = S.build_episode_scene(v, object_kind="box")
  place_and_settle(sc, v)
  sim = MoveSim(sc, v, servo_foh)
  seen = read_state(v, sim.positions())
  state = seen["state"]
  moves, illegal = [], 0
  while state is not None and state != H.GOAL_STATE and len(moves) < max_moves:
    move = H.next_move(state)
    expected = H.apply_move(state, move)
    sim.new_move()
    info, fails = expert_move(sim, spec, v, state, move, cfg, pinch_width)
    if sim.rec.action and not sim.unstable():
      sim.hold(TAIL_S)
    vd = sim.mon.verdict(expected, H.moved_cube(state, move))
    seen = read_state(v, sim.positions())
    if not seen["legal"] or seen["state"] is None:
      illegal += 1
    moves.append({"from_state": state, "move": list(move), "expected": expected,
                  "observed": seen["state"], "L_task": vd["L_task"], "L_full": vd["L_full"],
                  "reasons": fails + vd["reasons"], "planner_failures": fails,
                  "sim_s": round(float(sc.data.time), 2)})
    if fails or sim.unstable():
      break
    state = seen["state"] if seen["heights_ok"] else None
  return {"solved": state == H.GOAL_STATE, "moves_used": len(moves), "illegal_states": illegal,
          "final_state": state, "moves": moves, "sim_s": round(float(sc.data.time), 2),
          "wall_s": round(time.perf_counter() - wall0, 2), "frames": len(sim.rec.state)}
