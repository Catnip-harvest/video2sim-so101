"""Verifier for scattered-cube stacking: physics only, no planner phase labels, so it scores
the expert, a 10 Hz replay and a policy alike.

It is the Hanoi monitor (hanoi_verify.HanoiMonitor: every physics step, all three cubes,
both-jaw carries, drops, release height, other cubes moved / tipped, robot collisions,
limits, torque, speed caps, penetration) with the Hanoi-only "larger on smaller" rule
switched off (wrong stacks are legal start states here) and its own arrangement check.

Reading the arrangement (``read_stack_state``): a cube must lie flat (tilted <= 10 deg);
cube c sits ON cube X if c's lowest point is within 6 mm of X's highest point and c's centre
lies over X's top face; it sits on the TABLE if its lowest point is within 6 mm of the
table. Anything else (leaning against something, in the air) is unreadable.

Verdict for one move (``StackMonitor.verdict``):
  L_task   the arrangement equals the expected next state, and
           - every stacked pair of the expected state: centres within 10 mm (xy), the upper
             cube's centre within 4 mm (z) of its nominal height
           - a cube the move put on the table: inside the spawn zone, touching no other cube
           - everything static (all cubes < 2 cm/s) for the last 0.5 s
           - gripper open (jaw angle >= 0.10 rad) and no jaw touching any cube in the last
             0.5 s
           reasons: unstable, wrong_arrangement, stack_xy, stack_height, outside_zone,
                    touching_cube, not_static, gripper_not_open, not_released
  L_full   L_task and the physics clauses of hanoi_verify (moved cube carried by both jaws,
           lifted >= 2 cm, not dropped, released <= 10 mm above its landing surface; other
           cubes moved < 1 cm and tipped < 15 deg; no arm-link contact with a cube, fling,
           arm-table / self collision, joint limit, torque saturation, speed cap,
           penetration; style: tracking error, cube pushed before the grasp)
  L_strict L_full and every stacked pair of the expected state within 5 mm (xy)
"""

from __future__ import annotations

import math

import numpy as np

from . import stack as K
from .hanoi import CUBES, SIZES
from .hanoi_verify import (END_WINDOW_S, FULL_CLAUSES as _HANOI_FULL, STYLE_CLAUSES, HanoiMonitor,
                           tilt_deg)

READ_Z = 0.006
READ_TILT_DEG = 10.0
XY_TASK = 0.010
XY_STRICT = 0.005
Z_TOL = 0.004
STATIC_S = 0.5
GRIPPER_OPEN = 0.10  # rad

TASK_CLAUSES = ("unstable", "wrong_arrangement", "stack_xy", "stack_height", "outside_zone",
                "touching_cube", "not_static", "gripper_not_open", "not_released")
FULL_CLAUSES = tuple(c for c in _HANOI_FULL if c != "illegal_stack")
ALL_CLAUSES = TASK_CLAUSES + FULL_CLAUSES + ("stack_xy_strict",)


def _extent(R: np.ndarray, half: float) -> float:
  """Half height of a box of half-edge ``half`` with rotation R (centre to lowest point)."""
  return half * float(np.abs(np.asarray(R).reshape(3, 3)[2, :]).sum())


def read_stack_state(poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
  """poses: cube -> (centre xyz, 3x3 rotation). Returns the symbolic state (None if any
  cube is unreadable) and per-cube details."""
  half = {c: SIZES[c] / 2 for c in CUBES}
  bot = {c: float(p[2]) - _extent(R, half[c]) for c, (p, R) in poses.items()}
  top = {c: float(p[2]) + _extent(R, half[c]) for c, (p, R) in poses.items()}
  sup, detail = {}, {}
  for c in CUBES:
    p = np.asarray(poses[c][0], float)
    best = None
    for o in CUBES:
      if o == c:
        continue
      po, Ro = np.asarray(poses[o][0], float), np.asarray(poses[o][1]).reshape(3, 3)
      local = Ro.T @ (p - po)  # c's centre in o's frame
      over = abs(local[0]) <= half[o] and abs(local[1]) <= half[o]
      if over and abs(bot[c] - top[o]) <= READ_Z:
        best = o
    if tilt_deg(poses[c][1]) > READ_TILT_DEG:
      sup[c] = "?"
    elif best is not None:
      sup[c] = best
    elif abs(bot[c]) <= READ_Z:
      sup[c] = K.TABLE
    else:
      sup[c] = "?"
    detail[c] = {"bottom_mm": round(bot[c] * 1e3, 2), "tilt_deg": round(tilt_deg(poses[c][1]), 2)}
  state = "".join(sup[c] for c in CUBES)
  ok = "?" not in state and K.is_valid(state)
  return {"state": state if ok else None, "raw": state, "cubes": detail}


class StackMonitor(HanoiMonitor):
  check_legality = False
  # The stacking expert opens its jaws slowly at the release; a pad can brush the cube
  # again for a step or two as it settles. Shorter "carries" than this are not grasps.
  min_carry_s = 0.1
  # ... and a cube put on the table touches down (1.5 mm) while both pads still touch it.
  # That is placing, not dragging, if the jaw goal has opened >= 0.05 rad and the cube is
  # not sliding (< 2 cm/s); any other table contact while carried still fails.
  release_touchdown_rad = 0.05

  def poses(self) -> dict:
    d = self.sc.data
    return {c: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for c, b in self.ids.items()}

  def touching(self, a: str, b: str) -> bool:
    d, m = self.sc.data, self.sc.model
    ga, gb = int(m.body_geomadr[self.ids[a]]), int(m.body_geomadr[self.ids[b]])
    for k in range(d.ncon):
      g1, g2 = int(d.contact.geom[k, 0]), int(d.contact.geom[k, 1])
      if {g1, g2} == {ga, gb}:
        return True
    return False

  def verdict(self, expected_end: str, mover: str, move_target: str | None = None) -> dict:
    d = self.sc.data
    t_end = float(d.time)
    carries, others = self.physics_clauses(mover)
    poses = self.poses()
    seen = read_stack_state(poses)
    if seen["state"] != expected_end:
      self._mark("wrong_arrangement", t_end)
    pairs = {}
    strict_bad = False
    for c, s in K.supports(expected_end).items():
      if s == K.TABLE:
        continue
      off = float(np.linalg.norm(poses[c][0][:2] - poses[s][0][:2]))
      dz = float(poses[c][0][2]) - K.centre_height(expected_end, c)
      pairs[f"{c}_on_{s}"] = {"xy_mm": round(off * 1e3, 2), "dz_mm": round(dz * 1e3, 2)}
      if off > XY_TASK:
        self._mark("stack_xy", t_end)
      if off > XY_STRICT:
        strict_bad = True
      if abs(dz) > Z_TOL:
        self._mark("stack_height", t_end)
    placed = {}
    if move_target == K.TABLE:  # the move put ``mover`` on free table space
      xy = poses[mover][0][:2]
      r, az = K.polar(xy)
      placed = {"r_mm": round(r * 1e3, 1), "az_deg": round(az, 1), "in_zone": K.in_zone(xy),
                "touching": [o for o in others if self.touching(mover, o)]}
      if not placed["in_zone"]:
        self._mark("outside_zone", t_end)
      if placed["touching"]:
        self._mark("touching_cube", t_end)
    static_for = t_end - self.last_motion_t
    if static_for < STATIC_S:
      self._mark("not_static", t_end)
    jaw = float(d.qpos[self.sc.g_qadr])
    if jaw < GRIPPER_OPEN:
      self._mark("gripper_not_open", t_end)
    if self.last_jaw_contact_t >= t_end - END_WINDOW_S:
      self._mark("not_released", t_end)
    order = {c: i for i, c in enumerate(ALL_CLAUSES)}
    reasons = sorted(self.first, key=lambda r: (self.first[r], order.get(r, 99)))
    task_fail = [r for r in reasons if r in TASK_CLAUSES]
    full = not reasons
    return {
      "passed": full, "L_task": not task_fail, "L_full": full,
      "L_strict": bool(full and not strict_bad),
      "reasons": reasons, "task_reasons": task_fail,
      "style_only": bool(reasons) and set(reasons) <= set(STYLE_CLAUSES),
      "reason_times_s": dict(self.first),
      "observed_end_state": seen["state"], "expected_end_state": expected_end,
      "moved_cube": mover,
      "stats": {"sim_s": round(t_end, 3), "carries": carries,
                "carries_of_other_cubes": {c: len(self.carries[c]) for c in others},
                "pairs": pairs, "table_placement": placed, "read_state": seen,
                "static_for_s": round(static_for, 3) if math.isfinite(static_for) else None,
                "jaw_rad": round(jaw, 3), **self.physics_stats()},
    }


def start_state_intact(poses: dict, start_state: str) -> dict:
  """After the settle: the drawn start state must still be what physics shows (a wrong
  stack that toppled is rejected)."""
  seen = read_stack_state(poses)
  ok = seen["state"] == start_state
  for c, s in K.supports(start_state).items():
    if s != K.TABLE:
      ok &= abs(float(poses[c][0][2]) - K.centre_height(start_state, c)) <= Z_TOL
  return {"ok": bool(ok), "seen": seen}

