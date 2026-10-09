"""Phase-free verifier: judge ANY trajectory (scripted expert, policy, replay) from physics
alone, without the expert's phase labels (FINAL_DESIGN.md section 3).

The expert verifier in rollout.py reads the planner's phase labels ("carry", "lower",
"rest", ...) and only knows "holding" from targets that stay still to 1e-7. A policy has
no labels and never holds a target exactly still, so this module re-derives every event
from contacts, the object pose and the servo commands:

  carry      starts when BOTH jaws touch the object, the object centre is > 5 mm above its
             settled start height, and it touches neither the table nor the bin floor;
             lasts while both jaws touch it (the bin floor may be touched: that is placing).
  release    a carry that ends while the gripper is OPENING (jaw servo goal >= 0.1 rad
             above its lowest goal during that carry). A carry that ends any other way is
             a drop/slip.
  holding    (torque check) the 0.25 s saturation-and-sag window alone, no target test.

Clauses. L_task = the task is done; L_full = L_task and every physical-honesty / style
clause. ``passed`` is L_full.

  L_task  never_carried            no both-jaw carry ever happened (pushing, scooping,
                                   one-jaw tricks and "cube already in the bin" all fail)
          not_carried_into_bin     no carry ended with the object centre inside the bin
          not_in_place_region      final object not inside the walls (spec margin) and
                                   resting on the bin floor (+-4 mm), as in rollout.py
          not_resting              final object speed > 0.02 m/s
          not_released             a jaw touches the object in the last 0.5 s
          unstable                 non-finite state or absurd speeds
          wrong_object_in_bin      an extra free body (distractor, scene.extra_bodies) ends
                                   with its centre inside the bin's inner walls
  L_full  dropped_or_slipped       a carry ended without the gripper opening
          release_too_high         object's lowest point > 10 mm above the surface under
                                   it when a carry ended (bin floor inside the bin, table
                                   elsewhere)
          object_fling             object speed > 0.5 m/s at any physics step
          foreign_contact_while_carried   object touched table, bin wall or a non-jaw arm
                                   link while carried
          arm_body_touched_object  a non-jaw robot geom touched the object, any time
          object_disturbed_before_grasp   object moved > 10 mm (xy) by the first
                                   both-jaw contact. The expert's own close pushes the
                                   cube ~4 mm (fixed-jaw clearance), so the expert's 5 mm
                                   "before the close starts" test becomes 10 mm here.  [style]
          arm_table_collision / arm_bin_collision / self_collision
          joint_limit              a joint > 0.01 rad past its range
          torque_saturation        >= 98 % torque AND > 0.02 rad sag for > 0.25 s
          tracking_error           an arm joint lagged its goal by > 0.1 rad at a frame
                                   start                                    [style]
          speed_cap                commanded goal faster than 0.5 rad/s arm / 2 rad/s jaw
          penetration              a contact involving a ROBOT geom (jaw or link) deeper than
                                   2 mm for >= 3 consecutive physics steps (14 ms): the
                                   soft-contact abuse a controller can cause. Passive
                                   object-environment landings are only reported
                                   (``passive_min_contact_dist_mm``): with the scene's default
                                   soft contacts the expert's own 4-5 mm release drop sinks
                                   the cube 1.5-4.5 mm into the bin floor for up to 10 steps
                                   (measured, 60 expert episodes).
          not_lifted               object never rose >= 2 cm

Usage (one instance per episode, after the initial placement and settle):
  mon = PhaseFreeMonitor(sc)
  ... for every physics step: mujoco.mj_step(m, d); mon.step()
  ... at every 30 Hz frame start (before its substeps): mon.frame_start(target6)
  verdict = mon.verdict(spec, v)
Nothing here writes to the simulation.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import scene as S
from .paths import TaskSpec, mj_scene
from .planner import ARM_VMAX, JAW_VMAX
from .rollout import ROBOT_CATS, Monitor
from .variations import Variation

CAT_EXTRA = getattr(S, "CAT_EXTRA", 8)

TASK_CLAUSES = ("unstable", "never_carried", "not_carried_into_bin", "not_in_place_region",
                "not_resting", "not_released", "wrong_object_in_bin")
FULL_CLAUSES = ("dropped_or_slipped", "release_too_high", "object_fling",
                "foreign_contact_while_carried", "arm_body_touched_object",
                "object_disturbed_before_grasp", "arm_table_collision", "arm_bin_collision",
                "self_collision", "joint_limit", "torque_saturation", "tracking_error",
                "speed_cap", "penetration", "not_lifted")
STYLE_CLAUSES = ("object_disturbed_before_grasp", "tracking_error")
ALL_CLAUSES = TASK_CLAUSES + FULL_CLAUSES

LIFT_Z = 0.005           # carry starts above rest + 5 mm
OPENING_RAD = 0.10       # jaw goal this far above its carry minimum = "opening"
RELEASE_MAX_H = 0.010    # object bottom above the surface below at release
FLING_V = 0.5            # m/s
DISTURB_XY = 0.010       # m at the first both-jaw contact
PENETRATION = -0.002     # m (contact dist)
PENETRATION_STEPS = 3    # consecutive physics steps
NOT_LIFTED = 0.02        # m
END_WINDOW_S = 0.5       # "released" window at the end
TRACK_ERR = 0.1          # rad
SAT_WINDOW_S = 0.25


class PhaseFreeMonitor:
  def __init__(self, sc: S.EpisodeScene):
    self.sc = sc
    d, m = sc.data, sc.model
    # Reused from the expert verifier with a neutral phase: arm-table/bin/self collisions,
    # arm body touching the object, joint limits, instability, and (holding=None) the
    # saturation-and-sag window without any "target still" test.
    self.base = Monitor(sc)
    self.dt = m.opt.timestep
    self.rest = d.xpos[sc.obj_body].copy()
    self.rest_z = float(self.rest[2])
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    self.obj_geom = gid
    self.obj_half = np.asarray(m.geom_size[gid], float).copy()  # box half extents
    self.obj_is_box = int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    self.n_steps = 0
    self.carrying = False
    self.carry_min_goal = math.inf
    self.carries: list[dict] = []
    self.cur: dict | None = None
    self.first_both_jaw_t: float | None = None
    self.disturb_mm_at_grasp: float | None = None
    self.max_obj_z = self.rest_z
    self.max_obj_speed = 0.0
    self.max_obj_speed_t = 0.0
    self.min_contact_dist = 0.0
    self.min_contact_pair = ""
    self.pen_run = 0
    self.max_pen_run = 0
    self.passive_min_dist = 0.0
    self.arm_table_first: dict | None = None
    self.last_jaw_contact_t = -math.inf
    self.foreign: dict[str, int] = {}
    self.first: dict[str, float] = {}
    self.max_track_err = 0.0
    self.max_goal_speed = [0.0, 0.0]  # arm, jaw (rad/s, frame to frame)
    self.prev_target: np.ndarray | None = None
    self.n_frames = 0
    self.extra_rest = {int(b): d.xpos[b].copy() for b in getattr(sc, "extra_bodies", ())}
    self.extra_robot_steps = 0  # physics steps with a robot geom touching an extra body

  # ---- per 30 Hz frame, called BEFORE the frame's substeps -----------------------------
  def frame_start(self, target: np.ndarray) -> None:
    sc, d = self.sc, self.sc.data
    target = np.asarray(target, float)
    err = float(np.max(np.abs(d.qpos[sc.arm_qadr] - d.ctrl[sc.arm_act])))
    self.max_track_err = max(self.max_track_err, err)
    prev = self.prev_target
    if prev is None:  # the goal the servos held during the settle
      prev = np.r_[d.ctrl[sc.arm_act], d.ctrl[sc.g_act]]
    v = np.abs(target - prev) * S.FPS
    self.max_goal_speed = [max(self.max_goal_speed[0], float(v[:5].max())),
                           max(self.max_goal_speed[1], float(v[5]))]
    if v[:5].max() > ARM_VMAX + 1e-6 or v[5] > JAW_VMAX + 1e-6:
      self._mark("speed_cap")
    self.prev_target = target.copy()
    self.n_frames += 1

  def _mark(self, key: str, t: float | None = None) -> None:
    self.first.setdefault(key, round(float(self.sc.data.time if t is None else t), 3))

  def _robot_table_contact(self) -> dict:
    d, m, cat = self.sc.data, self.sc.model, self.sc.geom_cat
    best = None
    for i in range(d.ncon):
      g1, g2 = int(d.contact.geom[i, 0]), int(d.contact.geom[i, 1])
      for a, b in ((g1, g2), (g2, g1)):
        if cat[a] in ROBOT_CATS and cat[b] == S.CAT_TABLE:
          if best is None or d.contact.dist[i] < best[1]:
            best = (m.geom(a).name, float(d.contact.dist[i]))
    if best is None:
      return {"t": round(float(d.time), 3)}
    return {"t": round(float(d.time), 3), "geom": best[0], "dist_mm": round(best[1] * 1e3, 3)}

  def _obj_bottom_z(self) -> float:
    d, sc = self.sc.data, self.sc
    z = float(d.xpos[sc.obj_body, 2])
    if self.obj_is_box:
      R = d.geom_xmat[self.obj_geom].reshape(3, 3)
      return z - float(np.abs(R[2, :]) @ self.obj_half)
    return z - float(self.obj_half[0])

  def _surface_below(self, xy: np.ndarray, v: Variation | None) -> tuple[float, str]:
    if v is not None:
      bx, by = v.bin_xy
      outer = mj_scene.BIN_INNER_HALF + mj_scene.BIN_WALL_THICKNESS
      if abs(xy[0] - bx) <= outer and abs(xy[1] - by) <= outer:
        return mj_scene.bin_floor_top(), "bin_floor"
    return 0.0, "table"

  # ---- per physics step, called AFTER mj_step ------------------------------------------
  def step(self, v: Variation | None = None) -> None:
    sc, d, cat = self.sc, self.sc.data, self.sc.geom_cat
    n_tab = self.base.c["arm_table"]
    self.base.step("none", None)
    self.n_steps += 1
    if self.arm_table_first is None and self.base.c["arm_table"] > n_tab:
      self.arm_table_first = self._robot_table_contact()
    t = float(d.time)
    n = d.ncon
    fixed = moving = table = floor = wall = robot = False
    if n:
      g = d.contact.geom[:n]
      c1, c2 = cat[g[:, 0]], cat[g[:, 1]]
      other = np.where(c1 == S.CAT_OBJECT, c2, np.where(c2 == S.CAT_OBJECT, c1, -1))
      oo = other[other >= 0]
      if len(oo):
        fixed = bool(np.any(oo == S.CAT_FIXED_JAW))
        moving = bool(np.any(oo == S.CAT_MOVING_JAW))
        table = bool(np.any(oo == S.CAT_TABLE))
        floor = bool(np.any(oo == S.CAT_BIN_FLOOR))
        wall = bool(np.any(oo == S.CAT_BIN_WALL))
        robot = bool(np.any(oo == S.CAT_ROBOT))
      dist = d.contact.dist[:n]
      rob = np.isin(c1, ROBOT_CATS) | np.isin(c2, ROBOT_CATS)
      if self.extra_rest and np.any(rob & ((c1 == CAT_EXTRA) | (c2 == CAT_EXTRA))):
        self.extra_robot_steps += 1
      worst = 0.0
      if np.any(rob):
        i = int(np.flatnonzero(rob)[np.argmin(dist[rob])])
        worst = float(dist[i])
        if worst < self.min_contact_dist:
          self.min_contact_dist = worst
          self.min_contact_pair = f"{self.sc.model.geom(g[i, 0]).name}|{self.sc.model.geom(g[i, 1]).name}"
      if np.any(~rob):
        self.passive_min_dist = min(self.passive_min_dist, float(dist[~rob].min()))
      self.pen_run = self.pen_run + 1 if worst < PENETRATION else 0
    else:
      self.pen_run = 0
    self.max_pen_run = max(self.max_pen_run, self.pen_run)
    if self.pen_run >= PENETRATION_STEPS:
      self._mark("penetration", t)
    both = fixed and moving
    if fixed or moving:
      self.last_jaw_contact_t = t
    p = d.xpos[sc.obj_body]
    oz = float(p[2])
    self.max_obj_z = max(self.max_obj_z, oz)
    vel = float(np.linalg.norm(d.qvel[sc.obj_dadr:sc.obj_dadr + 3]))
    if vel > self.max_obj_speed:
      self.max_obj_speed, self.max_obj_speed_t = vel, t
    if vel > FLING_V:
      self._mark("object_fling", t)
    if both and self.first_both_jaw_t is None:
      self.first_both_jaw_t = t
      self.disturb_mm_at_grasp = float(np.linalg.norm(p[:2] - self.rest[:2])) * 1e3
      if self.disturb_mm_at_grasp > DISTURB_XY * 1e3:
        self._mark("object_disturbed_before_grasp", t)
    goal = float(d.ctrl[sc.g_act])
    if not self.carrying:
      if both and oz > self.rest_z + LIFT_Z and not table and not floor:
        self.carrying = True
        self.carry_min_goal = goal
        self.cur = {"t_start": round(t, 3), "foreign_steps": 0, "steps": 0}
    if self.carrying:
      if both:
        self.cur["steps"] += 1
        self.carry_min_goal = min(self.carry_min_goal, goal)
        if table or wall or robot:
          self.cur["foreign_steps"] += 1
          self._mark("foreign_contact_while_carried", t)
          for k, f in (("table", table), ("bin_wall", wall), ("arm", robot)):
            if f:
              self.foreign[k] = self.foreign.get(k, 0) + 1
      else:  # carry ended this step
        opening = goal >= self.carry_min_goal + OPENING_RAD
        surf, surf_name = self._surface_below(p[:2], v)
        h = self._obj_bottom_z() - surf
        in_bin = False
        if v is not None:
          bx, by = v.bin_xy
          in_bin = (abs(p[0] - bx) <= mj_scene.BIN_INNER_HALF
                    and abs(p[1] - by) <= mj_scene.BIN_INNER_HALF)
        self.cur.update({"t_end": round(t, 3), "opening": bool(opening),
                         "jaw_goal_rise_rad": round(goal - self.carry_min_goal, 4),
                         "bottom_above_surface_mm": round(h * 1e3, 2), "surface": surf_name,
                         "end_xy": [round(float(p[0]), 4), round(float(p[1]), 4)],
                         "in_bin": bool(in_bin)})
        if not opening:
          self._mark("dropped_or_slipped", t)
        if h > RELEASE_MAX_H:
          self._mark("release_too_high", t)
        self.carries.append(self.cur)
        self.cur = None
        self.carrying = False
        self.carry_min_goal = math.inf

  # ---- end of episode ------------------------------------------------------------------
  def verdict(self, spec: TaskSpec, v: Variation) -> dict:
    sc, d, m = self.sc, self.sc.data, self.sc.model
    b = self.base
    t_end = float(d.time)
    for key, reason in (("arm_body_object", "arm_body_touched_object"),
                        ("arm_table", "arm_table_collision"), ("arm_bin", "arm_bin_collision"),
                        ("self", "self_collision"), ("joint_limit", "joint_limit"),
                        ("unstable", "unstable")):
      if b.c[key]:
        self._mark(reason, b.first.get(key))
    if np.any(b.sat_max_run * m.opt.timestep > SAT_WINDOW_S):
      self._mark("torque_saturation", t_end)
    if self.max_track_err > TRACK_ERR:
      self._mark("tracking_error", t_end)
    if self.max_obj_z - self.rest_z < NOT_LIFTED:
      self._mark("not_lifted", t_end)
    carries = list(self.carries)
    if self.carrying and self.cur is not None:  # still carried at the end
      carries.append({**self.cur, "t_end": None, "still_held": True})
    if not carries:
      self._mark("never_carried", t_end)
    if not any(c.get("in_bin") for c in self.carries):
      self._mark("not_carried_into_bin", t_end)
    op = d.xpos[sc.obj_body]
    bx, by = v.bin_xy
    lim = mj_scene.BIN_INNER_HALF - spec.place.success_xy_margin
    rest_z = mj_scene.bin_floor_top() + v.obj_size / 2
    in_xy = abs(op[0] - bx) <= lim and abs(op[1] - by) <= lim
    on_floor = abs(op[2] - rest_z) < 0.004
    if not (in_xy and on_floor):
      self._mark("not_in_place_region", t_end)
    speed = float(np.linalg.norm(d.qvel[sc.obj_dadr:sc.obj_dadr + 3]))
    if speed > 0.02:
      self._mark("not_resting", t_end)
    if self.last_jaw_contact_t >= t_end - END_WINDOW_S:
      self._mark("not_released", t_end)
    extra = []
    for bid, p0 in self.extra_rest.items():
      p = d.xpos[bid]
      inside = (abs(p[0] - bx) <= mj_scene.BIN_INNER_HALF and abs(p[1] - by) <= mj_scene.BIN_INNER_HALF)
      if inside:
        self._mark("wrong_object_in_bin", t_end)
      extra.append({"name": m.body(bid).name, "in_bin": bool(inside),
                    "moved_mm": round(float(np.linalg.norm(p - p0)) * 1e3, 2),
                    "final_xyz": [round(float(x), 4) for x in p]})
    reasons = sorted(self.first, key=lambda r: (self.first[r], ALL_CLAUSES.index(r)))
    task_fail = [r for r in reasons if r in TASK_CLAUSES]
    style = [r for r in reasons if r in STYLE_CLAUSES]
    return {
      "passed": not reasons, "L_task": not task_fail, "L_full": not reasons,
      "reasons": reasons, "task_reasons": task_fail,
      "style_only": bool(reasons) and set(reasons) <= set(STYLE_CLAUSES),
      "style_reasons": style, "reason_times_s": dict(self.first),
      "stats": {
        "sim_s": round(t_end, 3), "carries": carries,
        "max_obj_lift_mm": round((self.max_obj_z - self.rest_z) * 1e3, 2),
        "max_obj_speed_m_s": round(self.max_obj_speed, 4),
        "max_obj_speed_t": round(self.max_obj_speed_t, 3),
        "min_contact_dist_mm": round(self.min_contact_dist * 1e3, 3),  # robot-involved
        "passive_min_contact_dist_mm": round(self.passive_min_dist * 1e3, 3),
        "min_contact_pair": self.min_contact_pair,
        "max_penetration_run_steps": self.max_pen_run,
        "disturb_mm_at_first_grasp": None if self.disturb_mm_at_grasp is None
        else round(self.disturb_mm_at_grasp, 2),
        "foreign_contact_kinds": self.foreign,
        "arm_table_first_contact": self.arm_table_first,
        "first_both_jaw_t": None if self.first_both_jaw_t is None else round(self.first_both_jaw_t, 3),
        "max_tracking_error_rad": round(self.max_track_err, 4),
        "max_goal_speed_rad_s": [round(x, 4) for x in self.max_goal_speed],
        "max_arm_torque_frac": np.round(b.max_torque_frac, 3).tolist(),
        "max_saturated_sag_s": np.round(b.sat_max_run * m.opt.timestep, 3).tolist(),
        "final_obj_offset_from_bin_mm": [round((op[0] - bx) * 1e3, 2), round((op[1] - by) * 1e3, 2)],
        "final_obj_z_err_mm": round((op[2] - rest_z) * 1e3, 2),
        "final_obj_speed_m_s": round(speed, 4),
        **({"extra_bodies": extra, "extra_robot_contact_steps": self.extra_robot_steps}
           if extra else {}),
      },
    }


def run_expert_judged(spec: TaskSpec, v: Variation, cfg=None, servo_foh: bool = True,
                      pinch_width: float | None = None, keep_arrays: bool = True):
  """Run the scripted expert exactly as datagen does (rollout.run_episode, unchanged) and
  judge the SAME simulation with both verifiers. The phase-free monitor is attached by
  wrapping rollout's Monitor/Recorder classes for the duration of the call (rollout.py is
  not edited); it sees every physics step and every frame's servo goal, never a phase.
  Returns (EpisodeResult with the expert verdict, phase-free verdict dict)."""
  from . import rollout as R

  holder: dict = {}

  class TapMonitor(R.Monitor):
    def __init__(self, sc):
      super().__init__(sc)
      holder["pf"] = PhaseFreeMonitor(sc)

    def step(self, phase, holding=None):
      super().step(phase, holding)
      holder["pf"].step(v)

  class TapRecorder(R.Recorder):
    def frame(self, target, phase):
      super().frame(target, phase)
      holder["pf"].frame_start(target)

  orig = R.Monitor, R.Recorder
  R.Monitor, R.Recorder = TapMonitor, TapRecorder
  try:
    res = R.run_episode(spec, v, cfg, pinch_width=pinch_width, keep_arrays=keep_arrays,
                        servo_foh=servo_foh)
  finally:
    R.Monitor, R.Recorder = orig
  pf = holder.get("pf")
  verdict = pf.verdict(spec, v) if pf is not None else None
  return res, verdict
