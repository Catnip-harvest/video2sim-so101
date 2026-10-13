"""Scripted expert for one scattered-stacking move, and the episode / tower runners.

One move = pick a clear cube and put it on another cube (medium on large, small on medium)
or on free table space in the zone (taking a wrongly stacked cube off). It is the Hanoi
expert (hanoi_expert.HanoiExpert: straight-down approach, per-cube grasp height, vertical
lift, arc carry, slow final lowering, release 3 mm above the landing surface, retreat, ready
pose) with three changes that scattered cubes at ANY yaw need:

  finger tilt per move   A 5-joint arm tilts its fingers only in its own vertical plane, and
                         a cube is level in the jaws only if the jaws close perpendicular to
                         that plane OR the fingers are vertical. Cube yaw is now uniform, so
                         the expert uses vertical fingers (any face pair, pads exactly on the
                         faces) whenever the move's highest point is within the
                         vertical-finger reach, and otherwise the SMALLEST outward tilt that
                         reaches (measured table CEILING_MM below), grasping the face pair
                         closest to perpendicular to the arm. The leftover misalignment tilts
                         the cube by asin(sin(face angle) * sin(tilt)), at most 5.3 deg at
                         7.5 deg tilt and 10.6 deg at 15 deg; the place stage lowers by the
                         measured lowest corner, so the cube lands corner-first and settles flat.
  carry clearance 12 mm  (Hanoi: 20 mm) above the tallest other cube, to stay inside reach on
                         the final move (small cube onto a two-cube stack, top 80 mm).
  smaller jaw opening    mouth = cube + 25 mm (Hanoi: + 40 mm) so the open jaws fit between
                         cubes only 3-10 mm apart; the collision-checked planner still tries
                         every face pair and both jaw sides.
"""

from __future__ import annotations

import time

import numpy as np

from . import scene as S
from . import stack as K
from .hanoi import CUBES, SIZES
from .hanoi_expert import (TAIL_S, HanoiExpert, MoveContext, MoveSim, cube_top, expert_move_ctx,
                           place_and_settle)
from .paths import TaskSpec
from .planner import PlanConfig, mouth_to_jaw_angle, plan_config_for
from .rollout import EpisodeResult, _jsonable
from .stack_verify import StackMonitor, start_state_intact
from .variations import PAN_AXIS, Variation

# Highest grasp-site height (mm) the IK reaches with the fingers tilted outward by the given
# angle, by distance from the pan axis (m). Measured on this arm (soft limits 95 %, position
# <= 3 mm, finger axis <= 5 deg, no arm-self / arm-table contact), worst case over closing
# directions 0 / 30 / 45 deg from perpendicular, 2.5 mm height steps (README_stack.md).
CEILING_R = np.round(np.arange(0.11, 0.2801, 0.01), 2)
CEILING_MM = {
  0.0: [72, 78, 82, 85, 88, 88, 88, 88, 85, 82, 78, 72, 68, 60, 50, 40, 25, 15],
  10.0: [82, 90, 98, 102, 108, 110, 112, 112, 112, 112, 110, 108, 105, 100, 95, 88, 80, 70],
  15.0: [88, 98, 105, 112, 118, 120, 125, 125, 128, 128, 125, 125, 122, 118, 115, 110, 102, 95],
  20.0: [90, 102, 112, 120, 128, 132, 138, 140, 140, 142, 142, 140, 140, 138, 132, 130, 122, 118],
}
TILT_CANDIDATES = (0.0, 5.0, 7.5, 10.0, 12.5, 15.0)
REACH_MARGIN = 0.003
HOVER_MIN = 0.015


def ceiling_at(xy, tilt_deg: float) -> float:
  """Interpolated reach ceiling (m) at this xy for this finger tilt."""
  r = float(np.linalg.norm(np.asarray(xy, float)[:2] - PAN_AXIS))
  tilts = sorted(CEILING_MM)
  zs = [float(np.interp(r, CEILING_R, CEILING_MM[t])) for t in tilts]
  return float(np.interp(tilt_deg, tilts, zs)) / 1e3


class StackExpert(HanoiExpert):
  carry_clearance = 0.012
  open_margin = 0.025  # jaw mouth = cube + 25 mm

  def __init__(self, *a, **k):
    super().__init__(*a, **k)
    # the base Expert never opens below 0.6 rad (mouth ~62 mm); scattered cubes can stand
    # 3-10 mm apart, so open only to cube + open_margin
    self.g_open = float(np.clip(mouth_to_jaw_angle(self.v.obj_size + self.open_margin), 0.3, 1.0))
    self.g_release = float(min(self.g_release, self.g_open))

  def _choose_tilts(self) -> None:
    ctx = self.ctx
    above_centre = self.grasp_z - float(ctx.obj_pos[2])
    carry_site = self.carry_z() + above_centre
    need_pick = max(self.grasp_z + HOVER_MIN, carry_site) + REACH_MARGIN
    need_place = carry_site + REACH_MARGIN
    mid = (np.asarray(ctx.obj_pos[:2]) + np.asarray(ctx.dest_xy[:2])) / 2
    ok = [t for t in TILT_CANDIDATES
          if ceiling_at(ctx.obj_pos, t) >= need_pick and ceiling_at(ctx.dest_xy, t) >= need_place
          and ceiling_at(mid, t) >= need_place]
    first = ok[0] if ok else TILT_CANDIDATES[-1]
    self.tilt_choice = first
    self.need_site = {"pick_mm": round(need_pick * 1e3, 1), "place_mm": round(need_place * 1e3, 1)}
    self.tilts_deg = tuple([first] + [t for t in TILT_CANDIDATES if t > first] + [20.0])

  def ceiling(self, xy) -> float:
    return ceiling_at(xy, self.tilt_choice)

  def _closings(self, normals: list, tilt_deg: float) -> list:
    # normals arrive sorted: the face pair closest to perpendicular-to-the-arm first
    return list(normals) if tilt_deg == 0.0 else list(normals[:2]) + list(normals[2:])

  def move_report(self) -> dict:
    return {"tilt_choice_deg": self.tilt_choice, "need_site": self.need_site,
            "carry_z": round(self.carry_z(), 4)}


# ---- context of a move, read from the simulation --------------------------------------------
def poses_of(sc: S.EpisodeScene, ids: dict[str, int]) -> dict:
  d = sc.data
  return {c: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for c, b in ids.items()}


def stack_move_context(sc: S.EpisodeScene, ids: dict[str, int], state: str,
                       move: tuple[str, str], spot) -> MoveContext:
  d = sc.data
  cube, target = move
  half = {c: SIZES[c] / 2 for c in CUBES}
  sup = K.supports(state)[cube]
  surface_below = 0.0 if sup == K.TABLE else cube_top(d, ids[sup], half[sup])
  if target == K.TABLE:
    dest_xy, dest_surface = np.asarray(spot, float), 0.0
  else:
    dest_xy = d.xpos[ids[target], :2].copy()
    dest_surface = cube_top(d, ids[target], half[target])
  obstacle_top = max(cube_top(d, ids[c], half[c]) for c in CUBES if c != cube)
  q = d.xquat[ids[cube]]
  return MoveContext(state=state, move=tuple(move), cube=cube, obj_pos=d.xpos[ids[cube]].copy(),
                     obj_yaw=2 * float(np.arctan2(q[3], q[0])), surface_below=surface_below,
                     dest_xy=dest_xy, dest_surface=dest_surface, obstacle_top=obstacle_top)


def free_spot_now(rng: np.random.Generator, sc: S.EpisodeScene, ids: dict[str, int],
                  state: str, mover: str) -> np.ndarray:
  """A free table spot for ``mover`` given where the other cubes are NOW (towers of the
  state after the move, each with its measured base position and footprint)."""
  d = sc.data
  after = K.apply_move(state, (mover, K.TABLE))
  others = [(d.xpos[ids[t[0]], :2].copy(), max(K.FOOTPRINT[c] for c in t))
            for t in K.towers(after) if mover not in t]
  return K.free_spot(rng, mover, others)


# ---- runners -----------------------------------------------------------------------------------
def _reject(v: Variation, reason: str, wall0: float, stats: dict) -> EpisodeResult:
  return EpisodeResult(attempt=v.attempt, accepted=False, reasons=[reason], primary_reason=reason,
                       stats=_jsonable(stats), variation=v.to_dict(), plan_info={},
                       sim_seconds=0.0, wall_seconds=time.perf_counter() - wall0)


def run_stack_episode(spec: TaskSpec, v: Variation, cfg: PlanConfig | None = None,
                      pinch_width: float | None = None, render_size=(256, 256),
                      keep_arrays: bool = True, servo_foh: bool = True) -> EpisodeResult:
  """One single-move stacking episode (datagen): settle the drawn start state (rejected as
  start_unstable if a stack does not survive the settle), let the expert make the rule's
  next move, record a settled tail, judge. accepted = L_full and a plan was found."""
  wall0 = time.perf_counter()
  info = K.stack_info(v)
  cfg = cfg or plan_config_for(v)
  sc = S.build_episode_scene(v, object_kind="box", render_size=render_size)
  place_and_settle(sc, v)
  sim = MoveSim(sc, v, servo_foh, monitor_cls=StackMonitor, noise=info.get("action_noise"))
  intact = start_state_intact(sim.mon.poses(), info["start_state"])
  if not intact["ok"]:
    return _reject(v, "start_unstable", wall0, {"start_check": intact})
  state, move = info["start_state"], tuple(info["move"])
  ctx = stack_move_context(sc, sim.ids, state, move, info.get("free_spot"))
  vm = K.variation_for_move(v, state, move, info.get("free_spot"))
  plan_info, plan_fails = expert_move_ctx(sim, spec, v, ctx, vm, cfg, pinch_width,
                                          expert_cls=StackExpert)
  if sim.rec.action and not sim.unstable():
    sim.hold(TAIL_S)
  vd = sim.mon.verdict(info["end_state"], info["moved_cube"], move[1])
  if not sim.speed_ok and "speed_cap" not in vd["reasons"]:
    vd["reasons"].append("speed_cap")
    vd["L_full"] = vd["passed"] = vd["L_strict"] = False
  reasons = list(plan_fails) + [r for r in vd["reasons"] if r not in plan_fails]
  accepted = bool(vd["L_full"] and not plan_fails)
  stats = {"stack": {k: vd[k] for k in ("L_task", "L_full", "L_strict", "reasons", "task_reasons",
                                        "style_only", "observed_end_state", "expected_end_state",
                                        "moved_cube")},
           "bucket": info["bucket"], "stratum": info["stratum"],
           **vd["stats"], "planner_failures": plan_fails, "reason_times_s": vd["reason_times_s"],
           "max_target_speed_arm_rad_s": round(sim.speed_max[0], 4),
           "max_target_speed_jaw_rad_s": round(sim.speed_max[1], 4)}
  if sim.noise:
    stats["noise_abs_max_rad"] = round(sim.noise_abs_max, 4)
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


def advancement(start: str, end: str | None) -> float:
  """Fraction of the rule's moves from ``start`` that are done at ``end`` (0..1; an
  unreadable end counts as no progress)."""
  n0 = len(K.rule_moves_to_goal(start))
  if end is None or n0 == 0:
    return 1.0 if n0 == 0 else 0.0
  return max(0.0, (n0 - len(K.rule_moves_to_goal(end))) / n0)


def run_stack_tower_expert(spec: TaskSpec, v: Variation, max_moves: int = 8,
                           pinch_width: float | None = None, servo_foh: bool = True) -> dict:
  """The scripted expert builds the whole tower in ONE continuous scene: read the state from
  the simulation, make the rule's next move (free spots drawn from the live positions),
  repeat until the goal or ``max_moves``."""
  wall0 = time.perf_counter()
  info = K.stack_info(v)
  cfg = plan_config_for(v)
  sc = S.build_episode_scene(v, object_kind="box")
  place_and_settle(sc, v)
  sim = MoveSim(sc, v, servo_foh, monitor_cls=StackMonitor)
  rng = np.random.default_rng([v.seed, 19])
  start = info["start_state"]
  intact = start_state_intact(sim.mon.poses(), start)
  if not intact["ok"]:
    return {"solved": False, "moves_used": 0, "start_state": start, "final_state": None,
            "moves": [], "stop_reason": "start_unstable", "advancement": 0.0}
  state, moves, stop = start, [], "max_moves"
  while state != K.GOAL_STATE and len(moves) < max_moves:
    move = K.next_move(state)
    spot = free_spot_now(rng, sc, sim.ids, state, move[0]) if move[1] == K.TABLE else None
    expected = K.apply_move(state, move)
    sim.new_move()
    ctx = stack_move_context(sc, sim.ids, state, move, spot)
    vm = K.variation_for_move(v, state, move, spot)
    pinfo, fails = expert_move_ctx(sim, spec, v, ctx, vm, cfg, pinch_width, expert_cls=StackExpert)
    if sim.rec.action and not sim.unstable():
      sim.hold(TAIL_S)
    vd = sim.mon.verdict(expected, move[0], move[1])
    moves.append({"from_state": state, "move": list(move), "expected": expected,
                  "observed": vd["observed_end_state"], "L_task": vd["L_task"],
                  "L_full": vd["L_full"], "L_strict": vd["L_strict"],
                  "reasons": fails + vd["reasons"], "tilt_deg": pinfo.get("tilt_choice_deg")})
    if fails or sim.unstable():
      stop = "planner_failure" if fails else "unstable"
      state = vd["observed_end_state"]
      break
    state = vd["observed_end_state"]
    if state is None:
      stop = "unreadable_state"
      break
  if state == K.GOAL_STATE:
    stop = "solved"
  return {"solved": state == K.GOAL_STATE, "moves_used": len(moves), "start_state": start,
          "final_state": state, "rule_moves": len(K.rule_moves_to_goal(start)),
          "advancement": round(advancement(start, state), 3), "moves": moves, "stop_reason": stop,
          "sim_s": round(float(sc.data.time), 2), "wall_s": round(time.perf_counter() - wall0, 2)}
