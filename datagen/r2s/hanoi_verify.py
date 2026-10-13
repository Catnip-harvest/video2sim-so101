"""Hanoi verifier: judge one move from physics alone (no planner phase labels), so the same
code scores the scripted expert, a 10 Hz replay and a learned policy.

It watches all three cubes the same way (it does not need to know which cube "should"
move until the verdict), on every physics step:
  * contacts: which cube touches which jaw, the table, another cube or a non-jaw arm link;
  * carries: a cube is CARRIED while both jaws touch it, it is > 5 mm above where it
    settled and it touches neither the table nor another cube (same rule as the
    pick-and-place phase-free verifier, verify_policy.py). A carry that ends while the jaw
    is not opening is a drop; a carry that ends with the cube's bottom more than 10 mm
    above whatever is under it (table or the top of the cube below) is a release too high;
  * how far every cube moved (3-D) and tipped (tilt of its nearest-to-vertical face
    normal, so a cube sitting flat on any face reads 0 deg) since the start;
  * legality: a LARGER cube resting on a SMALLER one (they touch, the larger is above and
    not held by both jaws) at any moment.
The robot-side checks (arm-table / self collision, joint limit, torque saturation,
instability) come from the pick-and-place Monitor (rollout.py), unchanged.

Verdict for one move (``verdict``):
  L_task  the final arrangement IS the expected next state: every cube's centre within
          2 cm (xy) of its spot centre and within 6 mm (z) of its stack height, all cubes
          at rest (< 2 cm/s), and no jaw touching any cube in the last 0.5 s.
          Reasons: unstable, cube_off_spot, wrong_arrangement, wrong_stack_height,
          not_resting, not_released.
  L_full  L_task and: the moved cube was carried by both jaws (never_carried, not_lifted
          < 2 cm, dropped_or_slipped, release_too_high), no other cube moved > 1 cm or
          tipped > 15 deg (other_cube_moved, other_cube_tipped), never a larger cube on a
          smaller (illegal_stack), and the physical-honesty clauses of the pick-and-place
          verifier: foreign_contact_while_carried (carried cube on the table or an arm
          link), arm_body_touched_object (a non-jaw arm link on any cube), object_fling,
          arm_table_collision, self_collision, joint_limit, torque_saturation,
          tracking_error, speed_cap, penetration, object_disturbed_before_grasp.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import hanoi as H
from . import scene as S
from .planner import ARM_VMAX, JAW_VMAX
from .rollout import ROBOT_CATS, Monitor

XY_TOL = 0.020        # cube centre vs. its spot centre
Z_TOL = 0.006         # cube centre vs. its stack height
REST_SPEED = 0.02     # m/s
OTHER_MOVE_MAX = 0.010
TIP_MAX_DEG = 15.0
LIFT_Z = 0.005
OPENING_RAD = 0.10
RELEASE_MAX_H = 0.010
FLING_V = 0.5
DISTURB_XY = 0.010
PENETRATION = -0.002
PENETRATION_STEPS = 3
NOT_LIFTED = 0.02
END_WINDOW_S = 0.5
TRACK_ERR = 0.1
SAT_WINDOW_S = 0.25
STACK_TOUCH_SLACK = 0.004  # "resting on": bottom of the upper cube within 4 mm of the lower's top

TASK_CLAUSES = ("unstable", "cube_off_spot", "wrong_arrangement", "wrong_stack_height",
                "not_resting", "not_released")
FULL_CLAUSES = ("never_carried", "not_lifted", "dropped_or_slipped", "release_too_high",
                "other_cube_moved", "other_cube_tipped", "illegal_stack",
                "foreign_contact_while_carried", "arm_body_touched_object", "object_fling",
                "arm_table_collision", "self_collision", "joint_limit", "torque_saturation",
                "tracking_error", "speed_cap", "penetration", "object_disturbed_before_grasp")
STYLE_CLAUSES = ("object_disturbed_before_grasp", "tracking_error")
ALL_CLAUSES = TASK_CLAUSES + FULL_CLAUSES


def cube_body_ids(sc: S.EpisodeScene, v) -> dict[str, int]:
  """cube letter -> body id, for a Hanoi or a scattered-stacking episode."""
  extras = v.extras or {}
  info = extras.get("hanoi") or extras.get("stack")
  return {c: mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_BODY, name)
          for name, c in info["bodies"].items()}


def tilt_deg(xmat: np.ndarray) -> float:
  """Angle between vertical and the cube's face normal closest to vertical."""
  R = np.asarray(xmat).reshape(3, 3)
  return math.degrees(math.acos(min(1.0, float(np.max(np.abs(R[2, :]))))))


def read_state(v, positions: dict[str, np.ndarray]) -> dict:
  """What arrangement is on the table, read from cube centres (m).

  Each cube is assigned to the nearest spot; ``state`` is None if a cube is more than
  XY_TOL from every spot. ``legal`` is False if, on some spot, a larger cube sits above a
  smaller one. ``heights_ok`` checks every cube against the stack height its state
  implies (a tilted or half-fallen cube fails it)."""
  spots = {s: np.asarray(xy, float) for s, xy in H.hanoi_info(v)["spots"].items()}
  letters, off = [], {}
  for cube in H.CUBES:
    p = np.asarray(positions[cube], float)
    dist = {s: float(np.linalg.norm(p[:2] - c)) for s, c in spots.items()}
    near = min(dist, key=dist.get)
    letters.append(near)
    if dist[near] > XY_TOL:
      off[cube] = round(dist[near] * 1e3, 1)
  state = "".join(letters)
  legal = True
  for spot in H.SPOTS:
    here = sorted((c for c in H.CUBES if state[H.CUBES.index(c)] == spot),
                  key=lambda c: positions[c][2])  # bottom to top by height
    if any(H.is_larger(here[k + 1], here[k]) for k in range(len(here) - 1)):
      legal = False
  dz = {c: round((float(positions[c][2]) - H.centre_height(state, c)) * 1e3, 2) for c in H.CUBES}
  heights_ok = legal and all(abs(x) <= Z_TOL * 1e3 for x in dz.values())
  return {"state": None if off else state, "nearest": state, "off_spot_mm": off,
          "legal": legal, "heights_ok": heights_ok, "dz_mm": dz}


class HanoiMonitor:
  """Per-physics-step bookkeeping for all three cubes. Nothing here writes to the sim.

  Usage: mon = HanoiMonitor(sc, v); every 30 Hz frame start (before its substeps)
  mon.frame_start(target6); every physics step after mj_step mon.step(); at the end
  mon.verdict(expected_end_state, moved_cube)."""
  check_legality = True  # Hanoi rule; scattered stacking (stack_verify.py) turns it off
  min_carry_s = 0.0  # stacking ignores shorter "carries" (stack_verify.StackMonitor)
  release_touchdown_rad = None  # stacking: table touch allowed once the jaw goal opened this much

  def __init__(self, sc: S.EpisodeScene, v):
    self.sc, self.v = sc, v
    m, d = sc.model, sc.data
    self.base = Monitor(sc)  # robot-side clauses, holding=None (no phase labels)
    self.ids = cube_body_ids(sc, v)
    self.cube_list = list(H.CUBES)
    self.geom_label = np.full(m.ngeom, -1, int)
    self.half = {}
    for k, c in enumerate(self.cube_list):
      g = int(m.body_geomadr[self.ids[c]])
      self.geom_label[g] = k
      self.half[c] = float(m.geom_size[g][0])
    self.dofadr = {c: int(m.body_dofadr[self.ids[c]]) for c in self.cube_list}
    self.rest = {c: d.xpos[b].copy() for c, b in self.ids.items()}
    self.max_disp = {c: 0.0 for c in self.cube_list}
    self.max_tilt = {c: tilt_deg(d.xmat[b]) for c, b in self.ids.items()}
    self.max_lift = {c: 0.0 for c in self.cube_list}
    self.max_speed = {c: 0.0 for c in self.cube_list}
    self.carrying = {c: False for c in self.cube_list}
    self.carry_min_goal = {c: math.inf for c in self.cube_list}
    self.cur: dict[str, dict | None] = {c: None for c in self.cube_list}
    self.carries: dict[str, list] = {c: [] for c in self.cube_list}
    self.first_both_jaw: dict[str, dict] = {}
    self.illegal_pairs: dict[str, float] = {}
    self.robot_link_cube: dict[str, int] = {c: 0 for c in self.cube_list}
    self.jaw_cube_steps: dict[str, int] = {c: 0 for c in self.cube_list}
    self.foreign: dict[str, int] = {}
    self.first: dict[str, float] = {}
    self.last_jaw_contact_t = -math.inf
    self.last_motion_t = -math.inf  # last physics step any cube moved faster than REST_SPEED
    self.pen_run = self.max_pen_run = 0
    self.min_contact_dist = 0.0
    self.min_contact_pair = ""
    self.max_track_err = 0.0
    self.max_goal_speed = [0.0, 0.0]
    self.prev_target = None
    self.n_steps = 0

  # ---- per 30 Hz frame, BEFORE its substeps ------------------------------------------------
  def frame_start(self, target) -> None:
    sc, d = self.sc, self.sc.data
    target = np.asarray(target, float)
    self.max_track_err = max(self.max_track_err,
                             float(np.max(np.abs(d.qpos[sc.arm_qadr] - d.ctrl[sc.arm_act]))))
    prev = self.prev_target
    if prev is None:
      prev = np.r_[d.ctrl[sc.arm_act], d.ctrl[sc.g_act]]
    speed = np.abs(target - prev) * S.FPS
    self.max_goal_speed = [max(self.max_goal_speed[0], float(speed[:5].max())),
                           max(self.max_goal_speed[1], float(speed[5]))]
    if speed[:5].max() > ARM_VMAX + 1e-6 or speed[5] > JAW_VMAX + 1e-6:
      self._mark("speed_cap")
    self.prev_target = target.copy()

  def _mark(self, key: str, t: float | None = None) -> None:
    self.first.setdefault(key, round(float(self.sc.data.time if t is None else t), 3))

  def _bottom(self, c: str) -> float:
    d = self.sc.data
    R = d.xmat[self.ids[c]].reshape(3, 3)
    return float(d.xpos[self.ids[c], 2]) - self.half[c] * float(np.abs(R[2, :]).sum())

  def _top(self, c: str) -> float:
    d = self.sc.data
    R = d.xmat[self.ids[c]].reshape(3, 3)
    return float(d.xpos[self.ids[c], 2]) + self.half[c] * float(np.abs(R[2, :]).sum())

  def _surface_under(self, c: str) -> tuple[float, str]:
    """Top of the highest other cube whose footprint is under cube c's centre, else table."""
    d = self.sc.data
    p = d.xpos[self.ids[c]]
    best, name = 0.0, "table"
    for o in self.cube_list:
      if o == c:
        continue
      q = d.xpos[self.ids[o]]
      if np.max(np.abs(p[:2] - q[:2])) <= self.half[o] and self._top(o) > best:
        best, name = self._top(o), H.CUBE_NAMES[o]
    return best, name

  # ---- per physics step, AFTER mj_step -------------------------------------------------------
  def step(self) -> None:
    sc, d, cat = self.sc, self.sc.data, self.sc.geom_cat
    self.base.step("none", None)
    self.n_steps += 1
    t = float(d.time)
    n = d.ncon
    k3 = len(self.cube_list)
    fixed = np.zeros(k3, bool)
    moving = np.zeros(k3, bool)
    table = np.zeros(k3, bool)
    robot = np.zeros(k3, bool)
    touch = np.zeros((k3, k3), bool)
    if n:
      g = d.contact.geom[:n]
      c1, c2 = cat[g[:, 0]], cat[g[:, 1]]
      l1, l2 = self.geom_label[g[:, 0]], self.geom_label[g[:, 1]]
      for la, lb, cb in ((l1, l2, c2), (l2, l1, c1)):
        sel = la >= 0
        if not np.any(sel):
          continue
        for i, other_lab, other_cat in zip(la[sel], lb[sel], cb[sel]):
          if other_lab >= 0:
            touch[i, other_lab] = True
          elif other_cat == S.CAT_FIXED_JAW:
            fixed[i] = True
          elif other_cat == S.CAT_MOVING_JAW:
            moving[i] = True
          elif other_cat == S.CAT_ROBOT:
            robot[i] = True
          elif other_cat == S.CAT_TABLE:
            table[i] = True
      dist = d.contact.dist[:n]
      rob = np.isin(c1, ROBOT_CATS) | np.isin(c2, ROBOT_CATS)
      worst = 0.0
      if np.any(rob):
        i = int(np.flatnonzero(rob)[np.argmin(dist[rob])])
        worst = float(dist[i])
        if worst < self.min_contact_dist:
          self.min_contact_dist = worst
          self.min_contact_pair = (f"{self.sc.model.geom(g[i, 0]).name}|"
                                   f"{self.sc.model.geom(g[i, 1]).name}@{t:.2f}s")
      self.pen_run = self.pen_run + 1 if worst < PENETRATION else 0
    else:
      self.pen_run = 0
    self.max_pen_run = max(self.max_pen_run, self.pen_run)
    if self.pen_run >= PENETRATION_STEPS:
      self._mark("penetration", t)
    if np.any(fixed | moving):
      self.last_jaw_contact_t = t
    goal = float(d.ctrl[sc.g_act])
    for i, c in enumerate(self.cube_list):
      b = self.ids[c]
      p = d.xpos[b]
      both = bool(fixed[i] and moving[i])
      self.max_disp[c] = max(self.max_disp[c], float(np.linalg.norm(p - self.rest[c])))
      self.max_lift[c] = max(self.max_lift[c], float(p[2] - self.rest[c][2]))
      self.max_tilt[c] = max(self.max_tilt[c], tilt_deg(d.xmat[b]))
      vel = float(np.linalg.norm(d.qvel[self.dofadr[c]:self.dofadr[c] + 3]))
      self.max_speed[c] = max(self.max_speed[c], vel)
      if vel >= REST_SPEED:
        self.last_motion_t = t
      if vel > FLING_V:
        self._mark("object_fling", t)
      if robot[i]:
        self.robot_link_cube[c] += 1
        self._mark("arm_body_touched_object", t)
      if fixed[i] or moving[i]:
        self.jaw_cube_steps[c] += 1
      if both and c not in self.first_both_jaw:
        self.first_both_jaw[c] = {"t": round(t, 3), "xy_shift_mm": round(
          float(np.linalg.norm(p[:2] - self.rest[c][:2])) * 1e3, 2)}
      # legality: this (larger) cube resting on a smaller one it touches
      if self.check_legality and not both:
        for j, o in enumerate(self.cube_list):
          if touch[i, j] and H.is_larger(c, o) and self._bottom(c) >= self._top(o) - STACK_TOUCH_SLACK:
            self.illegal_pairs.setdefault(f"{c}_on_{o}", round(t, 3))
            self._mark("illegal_stack", t)
      touching_cube = bool(touch[i].any())
      if not self.carrying[c]:
        if both and p[2] > self.rest[c][2] + LIFT_Z and not table[i] and not touching_cube:
          self.carrying[c] = True
          self.carry_min_goal[c] = goal
          self.cur[c] = {"t_start": round(t, 3), "steps": 0, "foreign_steps": 0}
      if self.carrying[c]:
        cur = self.cur[c]
        if both:
          cur["steps"] += 1
          self.carry_min_goal[c] = min(self.carry_min_goal[c], goal)
          # another cube under it is allowed: that is placing; so is the table when the
          # monitor allows a touchdown during a slow release (jaws opening, cube not sliding)
          touchdown = (self.release_touchdown_rad is not None and table[i] and vel < REST_SPEED
                       and goal >= self.carry_min_goal[c] + self.release_touchdown_rad)
          if (table[i] and not touchdown) or robot[i]:
            cur["foreign_steps"] += 1
            self._mark("foreign_contact_while_carried", t)
            for k, flag in (("table", table[i]), ("arm", robot[i])):
              if flag:
                self.foreign[k] = self.foreign.get(k, 0) + 1
        elif cur["steps"] < self.min_carry_s * S.FPS * S.SUBSTEPS:
          # both jaws touched a falling cube for a moment while opening: not a grasp
          self.cur[c] = None
          self.carrying[c] = False
          self.carry_min_goal[c] = math.inf
        else:  # the carry ended this step
          opening = goal >= self.carry_min_goal[c] + OPENING_RAD
          surf, surf_name = self._surface_under(c)
          h = self._bottom(c) - surf
          cur.update({"t_end": round(t, 3), "opening": bool(opening),
                      "bottom_above_surface_mm": round(h * 1e3, 2), "surface": surf_name,
                      "end_xy": [round(float(p[0]), 4), round(float(p[1]), 4)]})
          if not opening:
            self._mark("dropped_or_slipped", t)
          if h > RELEASE_MAX_H:
            self._mark("release_too_high", t)
          self.carries[c].append(cur)
          self.cur[c] = None
          self.carrying[c] = False
          self.carry_min_goal[c] = math.inf

  # ---- end of the move ---------------------------------------------------------------------
  def positions(self) -> dict[str, np.ndarray]:
    return {c: self.sc.data.xpos[b].copy() for c, b in self.ids.items()}

  def at_rest(self) -> bool:
    d = self.sc.data
    return all(float(np.linalg.norm(d.qvel[self.dofadr[c]:self.dofadr[c] + 3])) < REST_SPEED
               for c in self.cube_list)

  def physics_clauses(self, mover: str) -> tuple[list, list[str]]:
    """Mark every L_full clause that does not depend on the task's arrangement (robot
    collisions and limits, the moved cube's carry, the other cubes). Returns (carries of
    the moved cube, the other cubes)."""
    d, m = self.sc.data, self.sc.model
    b = self.base
    t_end = float(d.time)
    for key, reason in (("arm_table", "arm_table_collision"), ("self", "self_collision"),
                        ("joint_limit", "joint_limit"), ("unstable", "unstable")):
      if b.c[key]:
        self._mark(reason, b.first.get(key))
    if np.any(b.sat_max_run * m.opt.timestep > SAT_WINDOW_S):
      self._mark("torque_saturation", t_end)
    if self.max_track_err > TRACK_ERR:
      self._mark("tracking_error", t_end)
    # the moved cube: carried by both jaws, lifted, released low enough (marked above)
    carries = list(self.carries[mover])
    if self.carrying[mover] and self.cur[mover] is not None:
      carries.append({**self.cur[mover], "t_end": None, "still_held": True})
    if not carries:
      self._mark("never_carried", t_end)
    if self.max_lift[mover] < NOT_LIFTED:
      self._mark("not_lifted", t_end)
    grasp = self.first_both_jaw.get(mover)
    if grasp and grasp["xy_shift_mm"] > DISTURB_XY * 1e3:
      self._mark("object_disturbed_before_grasp", grasp["t"])
    others = [c for c in self.cube_list if c != mover]
    if any(self.max_disp[c] > OTHER_MOVE_MAX for c in others):
      self._mark("other_cube_moved", t_end)
    if any(self.max_tilt[c] > TIP_MAX_DEG for c in others):
      self._mark("other_cube_tipped", t_end)
    return carries, others

  def physics_stats(self) -> dict:
    b, m = self.base, self.sc.model
    return {
      "max_move_mm": {c: round(x * 1e3, 2) for c, x in self.max_disp.items()},
      "max_tilt_deg": {c: round(x, 2) for c, x in self.max_tilt.items()},
      "max_lift_mm": {c: round(x * 1e3, 2) for c, x in self.max_lift.items()},
      "max_speed_m_s": {c: round(x, 4) for c, x in self.max_speed.items()},
      "first_both_jaw": self.first_both_jaw, "illegal_pairs": self.illegal_pairs,
      "robot_link_cube_steps": self.robot_link_cube, "jaw_cube_steps": self.jaw_cube_steps,
      "foreign_contact_kinds": self.foreign,
      "min_robot_contact_dist_mm": round(self.min_contact_dist * 1e3, 3),
      "min_robot_contact_pair": self.min_contact_pair,
      "max_penetration_run_steps": self.max_pen_run,
      "max_tracking_error_rad": round(self.max_track_err, 4),
      "max_goal_speed_rad_s": [round(x, 4) for x in self.max_goal_speed],
      "max_arm_torque_frac": np.round(b.max_torque_frac, 3).tolist(),
      "max_saturated_sag_s": np.round(b.sat_max_run * m.opt.timestep, 3).tolist(),
      "arm_table_contacts": int(b.c["arm_table"]), "self_contacts": int(b.c["self"]),
    }

  def verdict(self, expected_end: str, mover: str) -> dict:
    d = self.sc.data
    t_end = float(d.time)
    carries, others = self.physics_clauses(mover)
    # the final arrangement
    pos = self.positions()
    seen = read_state(self.v, pos)
    spots = H.hanoi_info(self.v)["spots"]
    xy_err = {c: round(float(np.linalg.norm(pos[c][:2] - np.asarray(spots[expected_end[H.CUBES.index(c)]])))
                       * 1e3, 1) for c in self.cube_list}
    z_err = {c: round((float(pos[c][2]) - H.centre_height(expected_end, c)) * 1e3, 2)
             for c in self.cube_list}
    if any(e > XY_TOL * 1e3 for e in xy_err.values()):
      self._mark("cube_off_spot", t_end)
      if seen["state"] is not None and seen["state"] != expected_end:
        self._mark("wrong_arrangement", t_end)
    if any(abs(e) > Z_TOL * 1e3 for e in z_err.values()):
      self._mark("wrong_stack_height", t_end)
    if not self.at_rest():
      self._mark("not_resting", t_end)
    if self.last_jaw_contact_t >= t_end - END_WINDOW_S:
      self._mark("not_released", t_end)
    reasons = sorted(self.first, key=lambda r: (self.first[r], ALL_CLAUSES.index(r)))
    task_fail = [r for r in reasons if r in TASK_CLAUSES]
    style = [r for r in reasons if r in STYLE_CLAUSES]
    return {
      "passed": not reasons, "L_task": not task_fail, "L_full": not reasons,
      "reasons": reasons, "task_reasons": task_fail,
      "style_only": bool(reasons) and set(reasons) <= set(STYLE_CLAUSES),
      "style_reasons": style, "reason_times_s": dict(self.first),
      "observed_end_state": seen["state"], "observed_legal": seen["legal"],
      "expected_end_state": expected_end, "moved_cube": mover,
      "stats": {
        "sim_s": round(t_end, 3), "carries": carries,
        "carries_of_other_cubes": {c: len(self.carries[c]) for c in others},
        "end_xy_err_mm": xy_err, "end_z_err_mm": z_err, "read_state": seen,
        **self.physics_stats(),
      },
    }
