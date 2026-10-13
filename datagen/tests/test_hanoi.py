"""Tower of Hanoi: rules, solver, scene, stacking physics, scripted expert and verifier.

  python -m pytest -q -s -p no:cacheprovider datagen/tests/test_hanoi.py      (~2-3 min)

Physics only (CPU MuJoCo); the one render test is skipped when no OpenGL backend exists
(the Windows laptop). The expert data comes from a SCRIPTED EXPERT, not a human video.
"""

from __future__ import annotations

import colorsys
import itertools
import json
import math
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest

DATAGEN = Path(__file__).resolve().parents[1]
if str(DATAGEN) not in sys.path:
  sys.path.insert(0, str(DATAGEN))

from generate import pinch_width_from_spec  # noqa: E402
from r2s import hanoi as H  # noqa: E402
from r2s import scene as S  # noqa: E402
from r2s.hanoi_expert import (MoveSim, expert_move, hanoi_start_pose, place_and_settle,  # noqa: E402
                              run_hanoi_episode, run_hanoi_puzzle_expert)
from r2s.hanoi_verify import cube_body_ids, read_state  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.planner import plan_config_for  # noqa: E402
from r2s.render import gl_available  # noqa: E402
from r2s.variations import _in_reach, _in_view  # noqa: E402

SUGAR = DATAGEN.parent / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SPEC = load_task_spec(SUGAR)
PINCH = pinch_width_from_spec(SUGAR)
GL_OK, GL_MSG = gl_available()
EXPERT_SEED = 1  # the training seed the datagen commands use
N_EXPERT = 20


# ---- rules and solver ------------------------------------------------------------------------
def test_states_and_legal_moves():
  states = H.all_states()
  assert len(states) == 27 and len(set(states)) == 27
  assert len(H.NON_GOAL_STATES) == 26 and H.GOAL_STATE not in H.NON_GOAL_STATES
  assert H.stacks("AAA") == {"A": ["L", "M", "S"], "B": [], "C": []}
  assert H.stacks("ABA") == {"A": ["L", "S"], "B": ["M"], "C": []}
  assert sorted(H.legal_moves("AAA")) == [("A", "B"), ("A", "C")]  # only the small cube can move
  assert ("B", "A") not in H.legal_moves("ABA")  # medium onto small: illegal
  with pytest.raises(ValueError):
    H.apply_move("ABA", ("B", "A"))
  for s in states:
    for mv in H.legal_moves(s):
      t = H.apply_move(s, mv)
      assert (mv[1], mv[0]) in H.legal_moves(t) and H.apply_move(t, (mv[1], mv[0])) == s  # reversible


def test_solver_seven_moves_from_all_on_a():
  moves = H.optimal_moves_between("AAA")
  assert len(moves) == 7 == H.DIST_TO_GOAL["AAA"]
  assert moves == [("A", "C"), ("A", "B"), ("C", "B"), ("A", "C"), ("B", "A"), ("B", "C"), ("A", "C")]
  assert H.next_move(H.GOAL_STATE) is None
  assert max(H.DIST_TO_GOAL.values()) == 7 and len(H.DIST_TO_GOAL) == 27


def test_every_non_goal_state_has_an_optimal_next_move_of_the_top_cube():
  for s in H.NON_GOAL_STATES:
    mv = H.next_move(s)
    assert mv is not None and mv in H.legal_moves(s), s
    cube = H.moved_cube(s, mv)
    assert cube == H.top_cube(s, mv[0]) and cube is not None  # always the top cube
    below = H.top_cube(s, mv[1])
    assert below is None or H.is_larger(below, cube)  # never onto a smaller cube
    t = H.apply_move(s, mv)
    assert H.DIST_TO_GOAL[t] == H.DIST_TO_GOAL[s] - 1
    # optimal and unique: no other legal move gets one step closer
    closer = [m for m in H.legal_moves(s) if H.DIST_TO_GOAL[H.apply_move(s, m)] == H.DIST_TO_GOAL[s] - 1]
    assert closer == [mv], s
    # following next_move from any state reaches the goal in DIST_TO_GOAL moves
    assert len(H.optimal_moves_between(s)) == H.DIST_TO_GOAL[s]


def test_stack_heights():
  assert H.centre_height("AAA", "L") == pytest.approx(0.0225)
  assert H.centre_height("AAA", "M") == pytest.approx(0.0625)
  assert H.centre_height("AAA", "S") == pytest.approx(0.0925)
  assert H.centre_height("ABA", "S") == pytest.approx(0.0575)
  assert H.centre_height("CAB", "S") == pytest.approx(0.0125)
  assert H.stack_top("AAA", "A") == pytest.approx(0.105)


# ---- scene layout -------------------------------------------------------------------------------
def test_variation_layout_and_bookkeeping():
  starts, movers = Counter(), Counter()
  for i in range(300):
    v = H.sample_hanoi_variation(SPEC, 3, i)
    assert H.sample_hanoi_variation(SPEC, 3, i).to_dict() == v.to_dict()  # deterministic
    d = json.loads(json.dumps(v.to_dict()))
    assert d == v.to_dict()  # JSON round trip (episodes.jsonl)
    h = v.extras["hanoi"]
    spots = {s: np.asarray(xy) for s, xy in h["spots"].items()}
    for s, xy in spots.items():
      assert np.linalg.norm(xy - H.nominal_spot_xy(s)) <= H.SPOT_JITTER + 1e-12
      assert _in_reach(xy)
    for a, b in itertools.combinations(H.SPOTS, 2):
      assert np.linalg.norm(spots[a] - spots[b]) >= 0.09
    assert spots["A"][1] < spots["B"][1] < spots["C"][1]  # A left ... C right in cam_head
    assert h["start_state"] in H.NON_GOAL_STATES
    assert tuple(h["move"]) == H.next_move(h["start_state"])
    assert h["end_state"] == H.apply_move(h["start_state"], tuple(h["move"]))
    assert h["moved_cube"] == H.top_cube(h["start_state"], h["move"][0])
    assert h["instruction"] == H.INSTRUCTION
    assert v.extras["no_bin"] and len(v.extras["spot_marks"]) == 3
    bodies = h["bodies"]
    assert bodies["object"] == h["moved_cube"] and sorted(bodies.values()) == sorted(H.CUBES)
    assert v.obj_size == H.SIZES[h["moved_cube"]]
    sizes = {b["name"]: b["size"] for b in v.extras["free_bodies"]}
    for name, cube in bodies.items():
      if name != "object":
        assert sizes[name] == H.SIZES[cube]
    # colours: large blue-ish, medium green-ish, small red-ish
    rgb = {bodies["object"]: v.object_rgb, **{bodies[b["name"]]: b["rgb"] for b in v.extras["free_bodies"]}}
    hue = {c: colorsys.rgb_to_hsv(*rgb[c])[0] for c in H.CUBES}
    assert 0.55 < hue["L"] < 0.70 and 0.25 < hue["M"] < 0.42 and (hue["S"] < 0.05 or hue["S"] > 0.95)
    # the front camera (cam_head) sees all three spots and the top of a 3-cube stack on any
    for s, xy in spots.items():
      for z in (0.0, H.stack_top("AAA", "A")):
        for dx, dy in itertools.product((-0.0225, 0.0225), repeat=2):  # large cube's corners
          assert _in_view(v, np.array([xy[0] + dx, xy[1] + dy, z]), margin=0.95), (i, s, z)
    starts[h["start_state"]] += 1
    movers[h["moved_cube"]] += 1
  assert set(starts) == set(H.NON_GOAL_STATES)  # all 26 start states drawn
  assert set(movers) == {"L", "M", "S"}


def test_scene_has_no_bin_no_welds_and_visual_only_spot_marks():
  v = H.sample_hanoi_variation(SPEC, 0, 0, start_state="AAA")
  sc = S.build_episode_scene(v)
  m = sc.model
  assert m.neq == 0 and m.nmocap == 0
  names = [m.geom(g).name for g in range(m.ngeom)]
  assert not any(n.startswith("bin") for n in names)
  marks = [g for g in range(m.ngeom) if m.geom(g).name.startswith("spot_")]
  assert len(marks) == 6 and all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in marks)
  ids = cube_body_ids(sc, v)
  for c, b in ids.items():
    j = m.body_jntadr[b]
    assert m.body_jntnum[b] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    g = m.body_geomadr[b]
    assert m.geom_size[g][0] * 2 == pytest.approx(H.SIZES[c])


@pytest.mark.parametrize("state", ["AAA", "BBB", "CCB", "ABA"])
def test_stacks_settle_and_stay_put(state):
  """Three cubes stacked (or two) rest on their stack heights and drift < 2 mm over 2 s."""
  v = H.sample_hanoi_variation(SPEC, 0, 1, start_state=state)
  sc = S.build_episode_scene(v)
  place_and_settle(sc, v)
  ids = cube_body_ids(sc, v)
  d, m = sc.data, sc.model
  p0 = {c: d.xpos[b].copy() for c, b in ids.items()}
  for c in H.CUBES:
    assert abs(p0[c][2] - H.centre_height(state, c)) < 0.002, (c, p0[c][2])
  seen = read_state(v, p0)
  assert seen["state"] == state and seen["legal"] and seen["heights_ok"]
  worst = 0.0
  for _ in range(int(round(2.0 / m.opt.timestep))):
    mujoco.mj_step(m, d)
    worst = max(worst, max(float(np.linalg.norm(d.xpos[b] - p0[c])) for c, b in ids.items()))
  assert worst < 0.002, worst


def test_ready_pose_clears_a_three_cube_stack():
  """The Hanoi start pose (with the start jitter) keeps every robot geom off a full stack."""
  v = H.sample_hanoi_variation(SPEC, 0, 2, start_state="BBB")
  sc = S.build_episode_scene(v)
  place_and_settle(sc, v)
  m, d = sc.model, sc.data
  ids = cube_body_ids(sc, v)
  robot = np.isin(sc.geom_cat, (S.CAT_FIXED_JAW, S.CAT_MOVING_JAW, S.CAT_ROBOT))
  cube_geoms = {int(m.body_geomadr[b]) for b in ids.values()}
  for k in range(d.ncon):
    g1, g2 = (int(x) for x in d.contact.geom[k])
    assert not ((robot[g1] and g2 in cube_geoms) or (robot[g2] and g1 in cube_geoms))
    assert not ((robot[g1] and sc.geom_cat[g2] == S.CAT_TABLE) or
                (robot[g2] and sc.geom_cat[g1] == S.CAT_TABLE))
  assert np.allclose(d.qpos[sc.arm_qadr], hanoi_start_pose(v), atol=0.05)
  assert all(abs(d.xpos[b][2] - H.centre_height("BBB", c)) < 0.002 for c, b in ids.items())


# ---- read_state ----------------------------------------------------------------------------------
def test_read_state_from_positions():
  v = H.sample_hanoi_variation(SPEC, 0, 0)
  spots = {s: np.asarray(xy) for s, xy in v.extras["hanoi"]["spots"].items()}

  def at(state, nudge=None):
    out = {}
    for c in H.CUBES:
      xy = spots[state[H.CUBES.index(c)]]
      out[c] = np.r_[xy, H.centre_height(state, c)]
    if nudge:
      out[nudge[0]] = out[nudge[0]] + np.asarray(nudge[1])
    return out

  for s in ("AAA", "CAB", "BBC"):
    r = read_state(v, at(s))
    assert r["state"] == s and r["legal"] and r["heights_ok"]
  r = read_state(v, at("ABC", ("S", [0.05, 0.0, 0.0])))  # small cube between spots
  assert r["state"] is None and "S" in r["off_spot_mm"]
  pos = at("AAA")
  pos["L"][2], pos["S"][2] = H.SIZES["S"] + H.SIZES["M"] + H.SIZES["L"] / 2, H.SIZES["S"] / 2
  pos["M"][2] = H.SIZES["S"] + H.SIZES["M"] / 2  # upside-down tower
  r = read_state(v, pos)
  assert r["state"] == "AAA" and not r["legal"] and not r["heights_ok"]


# ---- the scripted expert, physics only ---------------------------------------------------------
def _expert_episode(i: int) -> dict:
  v = H.sample_hanoi_variation(SPEC, EXPERT_SEED, i)
  r = run_hanoi_episode(SPEC, v, pinch_width=PINCH, keep_arrays=False)
  h = v.extras["hanoi"]
  return {"i": i, "start": h["start_state"], "move": h["move"], "cube": h["moved_cube"],
          "L_task": bool(r.stats["hanoi"]["L_task"]), "L_full": bool(r.accepted),
          "reasons": r.reasons, "observed": r.stats["hanoi"]["observed_end_state"],
          "expected": h["end_state"], "sim_s": round(r.sim_seconds, 1)}


def test_expert_single_moves_physics_only():
  """20 single-move expert episodes (seed 1, ids 0-19), no rendering: target >= 17/20 L_full."""
  with mp.get_context("spawn").Pool(min(8, mp.cpu_count())) as pool:
    rows = pool.map(_expert_episode, range(N_EXPERT))
  n_task = sum(r["L_task"] for r in rows)
  n_full = sum(r["L_full"] for r in rows)
  print(f"\n[hanoi expert] single moves, seed {EXPERT_SEED}: L_task {n_task}/{N_EXPERT}, "
        f"L_full {n_full}/{N_EXPERT}; moved cube {dict(Counter(r['cube'] for r in rows))}")
  for r in rows:
    if not r["L_full"]:
      print("  failed:", json.dumps(r))
  assert all(r["L_task"] == (r["observed"] == r["expected"]) or not r["L_task"] for r in rows)
  assert n_full >= 17, rows


def test_expert_full_puzzle_solves_in_seven_moves():
  v = H.sample_hanoi_variation(SPEC, EXPERT_SEED, 0, start_state="AAA")
  out = run_hanoi_puzzle_expert(SPEC, v, pinch_width=PINCH)
  print(f"\n[hanoi expert] full puzzle: solved={out['solved']} moves={out['moves_used']} "
        f"illegal={out['illegal_states']} sim={out['sim_s']} s")
  assert out["solved"] and out["moves_used"] == 7 and out["illegal_states"] == 0
  assert [tuple(m["move"]) for m in out["moves"]] == H.optimal_moves_between("AAA")
  assert all(m["L_full"] for m in out["moves"]), out["moves"]


# ---- the verifier rejects wrong and illegal moves -------------------------------------------------
def _run_move(v, state, move):
  sc = S.build_episode_scene(v)
  place_and_settle(sc, v)
  sim = MoveSim(sc, v)
  info, fails = expert_move(sim, SPEC, v, state, move, plan_config_for(v), PINCH)
  sim.hold(1.0)
  return sim, fails


def test_verifier_rejects_a_legal_but_wrong_move():
  # "AAC": the solver says A->B (medium cube); the expert makes another legal move instead,
  # C->B (small cube onto the empty B). L_task must fail: not the expected arrangement.
  v = H.sample_hanoi_variation(SPEC, 0, 3, start_state="AAC")
  assert H.next_move("AAC") == ("A", "B")
  sim, fails = _run_move(v, "AAC", ("C", "B"))
  assert not fails
  vd = sim.mon.verdict(H.apply_move("AAC", ("A", "B")), "M")
  assert not vd["L_task"] and vd["observed_end_state"] == "AAB"
  assert "cube_off_spot" in vd["reasons"] and "wrong_arrangement" in vd["reasons"]


def test_verifier_flags_a_larger_cube_on_a_smaller_one(monkeypatch):
  # "ABC" (large A, medium B, small C): force the expert to put the medium cube on the small one.
  def unchecked(state, move):
    cube = H.top_cube(state, move[0])
    i = H.CUBES.index(cube)
    return state[:i] + move[1] + state[i + 1:]
  monkeypatch.setattr(H, "apply_move", unchecked)
  v = H.sample_hanoi_variation(SPEC, 0, 4, start_state="ABC")
  sim, fails = _run_move(v, "ABC", ("B", "C"))
  vd = sim.mon.verdict("ABB", "S")  # whatever was expected, the episode is illegal
  assert "illegal_stack" in vd["reasons"] and not vd["L_full"], vd["reasons"]
  assert "M_on_S" in vd["stats"]["illegal_pairs"]


def test_verifier_needs_a_move():
  # The arm never moves: the start arrangement stays, so L_task fails and nothing was carried.
  v = H.sample_hanoi_variation(SPEC, 0, 5)
  h = v.extras["hanoi"]
  sc = S.build_episode_scene(v)
  place_and_settle(sc, v)
  sim = MoveSim(sc, v)
  q = np.r_[hanoi_start_pose(v), v.gripper_start]
  sim.run(np.repeat(q[None], 90, 0), ["idle"] * 90, None)
  vd = sim.mon.verdict(h["end_state"], h["moved_cube"])
  assert not vd["L_task"] and not vd["L_full"]
  assert "never_carried" in vd["reasons"] and vd["observed_end_state"] == h["start_state"]


# ---- rendering (skipped without OpenGL) ------------------------------------------------------------
@pytest.mark.skipif(not GL_OK, reason=f"needs an OpenGL backend (MUJOCO_GL=egl/osmesa): {GL_MSG}")
def test_front_camera_shows_spots_and_cube_colours():
  """Render cam_head ("front") of an AAA scene: every cube's projected centre has its hue
  family, and every spot's mark is visible where its spot is empty (BBB scene)."""
  for state in ("AAA", "BBB"):
    v = H.sample_hanoi_variation(SPEC, 0, 6, start_state=state)
    sc = S.build_episode_scene(v)
    place_and_settle(sc, v)
    m, d = sc.model, sc.data
    r = mujoco.Renderer(m, 256, 256)
    r.update_scene(d, camera="front")
    img = r.render()
    wrist = (r.update_scene(d, camera="wrist_cam"), r.render())[1]
    r.close()
    assert img.std() > 5 and wrist.std() > 5
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    pos, R = d.cam_xpos[cam], d.cam_xmat[cam].reshape(3, 3)
    f = 0.5 * 256 / math.tan(math.radians(m.cam_fovy[cam]) / 2)

    def pix(p):
      c = R.T @ (np.asarray(p) - pos)  # camera frame: x right, y up, -z forward
      return int(round(128 + f * c[0] / -c[2])), int(round(128 - f * c[1] / -c[2]))

    ids = cube_body_ids(sc, v)
    want = {"L": 2, "M": 1, "S": 0}  # dominant channel: blue, green, red
    for c, b in ids.items():
      # a point inside the cube, 80 % of the way to its camera-facing side at mid height:
      # the camera ray to it enters through this cube's own visible side
      to_cam = (pos - d.xpos[b])[:2]
      p = d.xpos[b] + np.r_[to_cam / np.linalg.norm(to_cam) * 0.8 * H.SIZES[c] / 2, 0.0]
      u, w = pix(p)
      assert 1 <= u < 255 and 1 <= w < 255, (state, c, u, w)
      patch = img[w - 1:w + 2, u - 1:u + 2].reshape(-1, 3).mean(0)
      assert int(np.argmax(patch)) == want[c], (state, c, patch)
    if state == "BBB":  # A and C are empty: their marks show a light centre in a dark ring
      for s in ("A", "C"):
        xy = np.asarray(v.extras["hanoi"]["spots"][s])
        uc, wc = pix(np.r_[xy, 0.0])
        ur, wr = pix(np.r_[xy + np.array([0.030, 0.0]), 0.0])
        assert img[wc, uc].mean() > img[wr, ur].mean() + 40, (s, img[wc, uc], img[wr, ur])
