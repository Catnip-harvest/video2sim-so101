"""Scattered-cube stacking: rules + solver, spawn zone and draw, reachability, stacking
physics, scripted expert and verifier.

  python -m pytest -q -s -p no:cacheprovider datagen/tests/test_stack.py               (~3 min)
  STACK_LONG=1 python -m pytest -q -s -p no:cacheprovider datagen/tests/test_stack.py -k long
      (the measured numbers in README_stack.md: 120 single moves + 20 whole towers, ~6 min)

Physics only (CPU MuJoCo, no rendering). The expert data comes from a SCRIPTED EXPERT, not
a human video. Use at most 6 worker processes on the 31 GB laptop (12 ran out of memory).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

DATAGEN = Path(__file__).resolve().parents[1]
if str(DATAGEN) not in sys.path:
  sys.path.insert(0, str(DATAGEN))

from generate import pinch_width_from_spec  # noqa: E402
from r2s import scene as S  # noqa: E402
from r2s import stack as K  # noqa: E402
from r2s import stack_reach as R  # noqa: E402
from r2s.hanoi import CUBES, SIZES  # noqa: E402
from r2s.hanoi_expert import MoveSim, expert_move_ctx, hanoi_start_pose, place_and_settle  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.planner import plan_config_for  # noqa: E402
from r2s.stack_expert import (StackExpert, ceiling_at, run_stack_episode,  # noqa: E402
                              run_stack_tower_expert, stack_move_context)
from r2s.stack_verify import StackMonitor, read_stack_state, start_state_intact  # noqa: E402

SUGAR = DATAGEN.parent / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SPEC = load_task_spec(SUGAR)
PINCH = pinch_width_from_spec(SUGAR)
SEED = 1  # the training seed of the datagen commands
WORKERS = min(6, mp.cpu_count())
LONG = os.environ.get("STACK_LONG") == "1"


# ---- rules and solver --------------------------------------------------------------------------
def test_states_and_moves():
  states = K.all_states()
  assert len(states) == 13 and K.GOAL_STATE in states and K.ALL_ON_TABLE in states
  assert K.towers("TLM") == [["L", "M", "S"]]
  assert sorted(K.legal_moves("TTT")) == sorted([(a, b) for a in CUBES for b in CUBES if a != b])
  assert ("L", "T") not in K.legal_moves("TTT")  # already on the table
  assert ("M", "L") not in K.legal_moves("TSL")  # L is covered by S
  for s in states:
    for mv in K.legal_moves(s):
      assert K.is_valid(K.apply_move(s, mv))
  with pytest.raises(ValueError):
    K.apply_move("TLM", ("M", "T"))  # S sits on M


def test_rule_reaches_the_goal_from_every_state():
  """The dataset's next-move rule reaches the tower from all 12 non-goal states; it equals
  the BFS optimum except in some states with the medium cube on the small one, where it
  spends one extra move (medium to the table first instead of straight onto the large)."""
  assert K.next_move(K.GOAL_STATE) is None
  extras = []
  for s in K.all_states():
    moves = K.rule_moves_to_goal(s)
    t = s
    for mv in moves:
      assert mv in K.legal_moves(t), (s, t, mv)
      t = K.apply_move(t, mv)
    assert t == K.GOAL_STATE and len(moves) <= 4
    extra = len(moves) - K.OPTIMAL_MOVES[s]
    assert extra in (0, 1), (s, moves, K.OPTIMAL_MOVES[s])
    if extra:
      extras.append(s)
      assert K.supports(s)["M"] == "S", s
  assert "TST" in extras and len(extras) <= 3, extras
  assert K.rule_moves_to_goal("TTT") == [("M", "L"), ("S", "M")]
  assert K.next_move("TLT") == ("S", "M")
  assert K.next_move("TTL") == ("S", "T")  # small on large: take it off first
  assert K.next_move("TSL") == ("M", "T")  # medium on small on large: the top cube first


def test_start_buckets():
  assert K.PARTIAL_STATE == "TLT"
  assert abs(sum(K.BUCKETS.values()) - 1) < 1e-9 and abs(sum(K.SCATTER_STRATA.values()) - 1) < 1e-9
  for s in K.WRONG_STATES:  # every wrong start has a wrongly placed cube to take off
    assert K.is_valid(s) and s != K.GOAL_STATE and K.wrongly_placed(s)
    assert K.next_move(s)[1] == K.TABLE


# ---- spawn zone and the per-episode draw ---------------------------------------------------------
def _draws(n=400):
  return [K.sample_stack_variation(SPEC, SEED, i) for i in range(n)]


def test_draw_layout_separations_and_mix():
  vs = _draws()
  buckets, strata = Counter(), Counter()
  for v in vs:
    h = v.extras["stack"]
    buckets[h["bucket"]] += 1
    poses = {h["bodies"]["object"]: v.obj_xy}
    for b in v.extras["free_bodies"]:
      poses[h["bodies"][b["name"]]] = b["xy"]
    for c, xy in poses.items():  # every cube on the table inside the zone
      if K.supports(h["start_state"])[c] == K.TABLE:
        assert K.in_zone(np.asarray(xy)), (c, K.polar(np.asarray(xy)))
    gaps = h["gaps_mm"]
    if h["start_state"] == K.ALL_ON_TABLE:
      strata[h["stratum"]] += 1
      g = sorted(gaps.values())
      if h["stratum"] == "default":
        assert g[0] >= K.GAP_DEFAULT * 1e3 - 0.1, gaps
      elif h["stratum"] == "near":
        assert g[0] >= 2.9 and g[1] <= 10.1, gaps  # a tight cluster
      else:  # far_near: one cube far out, the other two 3-10 mm apart
        assert 2.9 <= g[0] <= 10.1, gaps
        rs = sorted(K.polar(np.asarray(xy))[0] for xy in poses.values())
        assert rs[-1] >= K.FAR_FRACTION * K.ZONE_R[1] - 1e-9
    else:
      assert all(x >= K.GAP_DEFAULT * 1e3 - 0.1 for x in gaps.values()), gaps
    if h["free_spot"] is not None:
      assert K.in_zone(np.asarray(h["free_spot"]), K.SPOT_MARGIN)
    assert h["end_state"] == K.apply_move(h["start_state"], tuple(h["move"]))
    assert h["instruction"] == K.INSTRUCTION
  print(f"\n[stack draw] buckets {dict(buckets)} scatter strata {dict(strata)}")
  for b, w in K.BUCKETS.items():
    assert abs(buckets[b] / len(vs) - w) < 0.07, buckets
  yaws = [v.obj_yaw for v in vs]
  assert min(yaws) < -2.5 and max(yaws) > 2.5  # full-circle yaw


def test_sampler_raises_instead_of_accepting_an_invalid_layout(monkeypatch):
  monkeypatch.setattr(K, "MAX_TRIES", 1)
  monkeypatch.setattr(K, "GAP_DEFAULT", 0.5)  # impossible separation
  with pytest.raises(RuntimeError):
    K.sample_stack_variation(SPEC, SEED, 0, bucket="scatter", stratum="default")


# ---- reachability (the zone comes from this) -----------------------------------------------------
def test_zone_corners_reachable():
  """Zone corners and middle: a table-height grasp with vertical fingers and the worst
  closing direction, the carry height over a two-cube stack with the 15-20 deg tilt, and the
  retreat height over a finished tower (~124 mm) with the 20 deg tilt."""
  R.init(str(SUGAR))
  rng = np.random.default_rng(0)
  rmin, rmax = K.ZONE_R
  for r in (rmin, (rmin + rmax) / 2, rmax):
    for az in (-K.ZONE_AZ_DEG, 0.0, K.ZONE_AZ_DEG):
      xy = R.zone_xy(r, az)
      assert R.feasible(xy, 0.020, 0.0, 45.0, rng), ("table grasp", r, az)
      assert R.feasible(xy, 0.113, 20.0, 45.0, rng), ("carry over two cubes", r, az)
      assert R.feasible(xy, 0.124, 20.0, 0.0, rng), ("retreat over a tower", r, az)
  assert not R.feasible(R.zone_xy(0.12, 0.0), 0.124, 20.0, 0.0, rng)  # why the zone starts at 0.15
  assert ceiling_at(R.zone_xy(rmin, 0.0), 20.0) >= 0.124 and ceiling_at(R.zone_xy(0.14, 0.0), 20.0) < 0.124


# ---- stacking physics and reading the arrangement --------------------------------------------------
@pytest.mark.parametrize("state", ["TLT", "TTL", "TTM", "TST", "MTT", "STT", "TSL"])
def test_start_stacks_settle_and_are_read_back(state):
  for i in range(2):
    bucket = "partial" if state == "TLT" else "wrong"
    v = K.sample_stack_variation(SPEC, SEED, i, bucket=bucket, start_state=state)
    sc = S.build_episode_scene(v, object_kind="box")
    place_and_settle(sc, v)
    mon = StackMonitor(sc, v)
    ok = start_state_intact(mon.poses(), state)
    assert ok["ok"], ok
    assert read_stack_state(mon.poses())["state"] == state


# ---- the scripted expert ---------------------------------------------------------------------------
def _episode(args) -> dict:
  seed, i, kw = args
  v = K.sample_stack_variation(SPEC, seed, i, **kw)
  r = run_stack_episode(SPEC, v, pinch_width=PINCH, keep_arrays=False)
  h = v.extras["stack"]
  st = r.stats.get("stack", {})
  return {"i": i, "bucket": h["bucket"], "stratum": h["stratum"], "start": h["start_state"],
          "move": h["move"], "L_full": bool(r.accepted), "L_task": bool(st.get("L_task")),
          "L_strict": bool(st.get("L_strict")), "reasons": r.reasons,
          "observed": st.get("observed_end_state"), "expected": h["end_state"],
          "tilt": r.plan_info.get("tilt_choice_deg")}


def _report(tag: str, rows: list[dict]) -> None:
  by = defaultdict(lambda: [0, 0, 0, 0])
  for r in rows:
    k = r["bucket"] + (f"/{r['stratum']}" if r["stratum"] else f" {r['start']}")
    by[k][0] += r["L_full"]
    by[k][1] += r["L_task"]
    by[k][2] += r["L_strict"]
    by[k][3] += 1
  n = len(rows)
  print(f"\n[{tag}] L_full {sum(r['L_full'] for r in rows)}/{n}, L_task "
        f"{sum(r['L_task'] for r in rows)}/{n}, L_strict {sum(r['L_strict'] for r in rows)}/{n}")
  for k in sorted(by):
    f, t, s, m = by[k]
    print(f"  {k:24s} L_full {f}/{m}  L_task {t}/{m}  strict {s}/{m}")
  print("  reasons:", dict(Counter(x for r in rows for x in r["reasons"]).most_common()))
  for r in rows:
    if not r["L_full"]:
      print("  failed:", json.dumps(r))


FORCED = ([{"bucket": "scatter", "stratum": s} for s in ("default", "near", "far_near") for _ in (0, 1)]
          + [{"bucket": "partial"}] * 2
          + [{"bucket": "wrong", "start_state": s} for s in ("TTL", "TTM", "TST", "MTT", "STT", "TSL")])


def test_expert_moves_from_every_bucket():
  """14 single moves, two per scatter stratum, two partial towers and every wrong start."""
  jobs = [(SEED, 1000 + i, kw) for i, kw in enumerate(FORCED)]
  with mp.get_context("spawn").Pool(WORKERS) as pool:
    rows = pool.map(_episode, jobs)
  _report("stack expert, forced buckets", rows)
  assert all(r["observed"] == r["expected"] for r in rows if r["L_task"])
  assert sum(r["L_full"] for r in rows) >= 12, rows


@pytest.mark.skipif(not LONG, reason="set STACK_LONG=1 (120 episodes, ~2 min with 6 workers)")
def test_long_expert_single_moves_120():
  """The datagen draw itself (seed 1, attempts 0-119, all buckets in their mix)."""
  with mp.get_context("spawn").Pool(WORKERS) as pool:
    rows = pool.map(_episode, [(SEED, i, {}) for i in range(120)])
  _report("stack expert, datagen mix", rows)
  assert sum(r["L_full"] for r in rows) >= 0.85 * len(rows)


def _tower(args) -> dict:
  seed, i, bucket = args
  v = K.sample_stack_variation(SPEC, seed, i, bucket=bucket)
  out = run_stack_tower_expert(SPEC, v, pinch_width=PINCH)
  return {"i": i, "bucket": bucket, "start": out["start_state"], "solved": out["solved"],
          "moves_used": out["moves_used"], "rule_moves": out["rule_moves"],
          "advancement": out["advancement"], "stop": out["stop_reason"],
          "all_L_full": all(m["L_full"] for m in out["moves"]),
          "reasons": sorted({x for m in out["moves"] for x in m["reasons"]})}


@pytest.mark.skipif(not LONG, reason="set STACK_LONG=1 (20 whole towers, ~3 min with 6 workers)")
def test_long_expert_whole_towers():
  """The expert builds the whole tower from 10 full scatters and 10 wrong stacks."""
  jobs = [(SEED, 2000 + i, "scatter") for i in range(10)] + [(SEED, 3000 + i, "wrong") for i in range(10)]
  with mp.get_context("spawn").Pool(WORKERS) as pool:
    rows = pool.map(_tower, jobs)
  for b in ("scatter", "wrong"):
    sub = [r for r in rows if r["bucket"] == b]
    print(f"\n[stack tower, {b}] solved {sum(r['solved'] for r in sub)}/{len(sub)}, "
          f"in rule moves {sum(r['solved'] and r['moves_used'] == r['rule_moves'] for r in sub)}, "
          f"every move L_full {sum(r['all_L_full'] for r in sub)}")
    for r in sub:
      print("  ", json.dumps(r))
  assert sum(r["solved"] for r in rows) >= 18


def test_tower_runner_from_full_scatter():
  v = K.sample_stack_variation(SPEC, SEED, 2000, bucket="scatter")
  out = run_stack_tower_expert(SPEC, v, pinch_width=PINCH)
  assert out["solved"] and out["moves_used"] == 2 and out["advancement"] == 1.0, out
  assert [tuple(m["move"]) for m in out["moves"]] == [("M", "L"), ("S", "M")]


# ---- the verifier rejects wrong moves and no moves ----------------------------------------------------
def _forced_move(v, state, move):
  sc = S.build_episode_scene(v, object_kind="box")
  place_and_settle(sc, v)
  sim = MoveSim(sc, v, monitor_cls=StackMonitor)
  ctx = stack_move_context(sc, sim.ids, state, move, None)
  vm = K.variation_for_move(v, state, move)
  info, fails = expert_move_ctx(sim, SPEC, v, ctx, vm, plan_config_for(v), PINCH,
                                expert_cls=StackExpert)
  sim.hold(1.0)
  return sim, fails


def test_verifier_rejects_a_wrong_but_physical_move():
  # From "TTT" the rule says medium onto large; the expert puts the SMALL cube on the large
  # one instead. The physics is clean, so only L_task fails (wrong arrangement).
  v = K.sample_stack_variation(SPEC, SEED, 4000, bucket="scatter", stratum="default")
  sim, fails = _forced_move(v, "TTT", ("S", "L"))
  assert not fails
  vd = sim.mon.verdict(K.apply_move("TTT", ("M", "L")), "M", "L")
  assert not vd["L_task"] and not vd["L_full"] and "wrong_arrangement" in vd["reasons"]
  assert vd["observed_end_state"] == "TTL"


def test_verifier_needs_a_move():
  v = K.sample_stack_variation(SPEC, SEED, 4001, bucket="scatter")
  h = v.extras["stack"]
  sc = S.build_episode_scene(v, object_kind="box")
  place_and_settle(sc, v)
  sim = MoveSim(sc, v, monitor_cls=StackMonitor)
  q = np.r_[hanoi_start_pose(v), v.gripper_start]
  sim.run(np.repeat(q[None], 90, 0), ["idle"] * 90, None)
  vd = sim.mon.verdict(h["end_state"], h["moved_cube"], h["move"][1])
  assert not vd["L_task"] and "never_carried" in vd["reasons"]
  assert vd["observed_end_state"] == h["start_state"]


def test_read_stack_state_geometry():
  eye = np.eye(3)
  half = {c: SIZES[c] / 2 for c in CUBES}
  poses = {"L": (np.array([0.2, 0.0, half["L"]]), eye),
           "M": (np.array([0.2, 0.004, 2 * half["L"] + half["M"]]), eye),
           "S": (np.array([0.25, 0.05, half["S"]]), eye)}
  assert read_stack_state(poses)["state"] == "TLT"
  poses["M"] = (np.array([0.2 + half["L"] + 0.002, 0.0, 2 * half["L"] + half["M"]]), eye)
  assert read_stack_state(poses)["state"] is None  # centre beyond L's top face: not "on"
  tilt = np.array([[1, 0, 0], [0, math.cos(0.5), -math.sin(0.5)], [0, math.sin(0.5), math.cos(0.5)]])
  poses["M"] = (np.array([0.3, 0.1, half["M"] + 0.01]), tilt)  # leaning on something
  assert read_stack_state(poses)["state"] is None


def test_noise_bucket_perturbs_the_expert():
  v = K.sample_stack_variation(SPEC, SEED, 4002, bucket="noise")
  assert v.extras["stack"]["action_noise"]["sigma_rad"] == K.NOISE["sigma_rad"]
  r = run_stack_episode(SPEC, v, pinch_width=PINCH, keep_arrays=False)
  assert r.stats["noise_abs_max_rad"] > 0.005
