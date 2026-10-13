"""Scattered-stacking eval harness (CPU only, no GPU; ~3 min).

  python -m pytest -q -s -p no:cacheprovider eval/tests/test_stack_eval.py

Policy mode is exercised with a stand-in runner that plays the scripted expert's 10 Hz
actions (no checkpoint, no GPU); without OpenGL its camera renders are replaced by blank
frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval" / "tests"))

import sim_eval as E  # noqa: E402
import stack_eval as SV  # noqa: E402
from generate import pinch_width_from_spec  # noqa: E402
from r2s import stack as K  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.stack_expert import run_stack_episode  # noqa: E402
from test_hanoi_eval import BlankRenderer, ExpertPolicy  # noqa: E402

SPEC = load_task_spec(E.SPEC)
PINCH = pinch_width_from_spec(Path(E.SPEC))


def _cfg(tmp_path, mode: str, stack: str, **kw) -> dict:
  cfg = {"mode": mode, "stack": stack, "seed": 1000, "out": str(tmp_path), "quality": "fast",
         "obs_codec": False, "interp": "lag", "max_sim_s": 45.0, "move_max_s": 45.0,
         "max_moves": 8, "tail_s": 0.3, "pinch_width": PINCH, "smooth": "none",
         "ema_alpha": 0.5, "replan_every": 5, "video_ids": [], "workers": 1, "checkpoint": None}
  cfg.update(kw)
  return cfg


def test_harness_start_matches_datagen(tmp_path):
  v, info = SV._eval_variation(SPEC, _cfg(tmp_path, "expert", "move"), 1)
  res = run_stack_episode(SPEC, v, pinch_width=PINCH)
  h = SV.StackHarness(SPEC, v)
  assert np.array_equal(h.sc.data.qpos, res.qpos_full[0])  # same build, placement, settle


def test_eval_scenes_have_no_action_noise(tmp_path):
  cfg = _cfg(tmp_path, "expert", "move")
  for i in range(60):
    v, info = SV._eval_variation(SPEC, cfg, i)
    assert info["bucket"] != "noise" and info["action_noise"] is None


def test_move_expert_mode(tmp_path):
  cfg = _cfg(tmp_path, "expert", "move")
  rows = [SV.episode(i, cfg, SPEC, {}) for i in (1, 2, 5)]
  for r in rows:
    assert r["valid_scene"] and r["expert_native"]["expert_verifier_pass"]
    assert r["L_task"] and r["L_full"], r["reasons"]
    assert r["observed_end_state"] == r["expected_end_state"]
    assert Path(r["rollout_file"]).exists()
  s = SV.summarize(rows, cfg)
  assert s["end_state_matches_expected"] == 3 and s["valid_scenes"] == 3
  assert sum(x["n"] for x in s["by_bucket"].values()) == 3
  full = E.summarize(rows, cfg, 1.0, None)  # sim_eval's summary accepts stacking rows
  assert full["success_L_full"] == 3 and full["stack"]["task"] == "move"


def test_move_policy_mode_stops_when_the_move_is_done(tmp_path, monkeypatch):
  monkeypatch.setattr(E, "ObsRenderer", BlankRenderer)
  cfg = _cfg(tmp_path, "policy", "move")
  v, info = SV._eval_variation(SPEC, cfg, 2)
  res = run_stack_episode(SPEC, v, pinch_width=PINCH)
  runner = ExpertPolicy(E.sim_to_mm2(res.action[::E.STEP_FRAMES]))
  r = SV.episode(2, cfg, SPEC, {"runner": runner})
  assert r["stop_reason"] == "move_done_1s", r["stop_reason"]
  assert r["L_task"] and r["L_full"], r["reasons"]
  assert runner.images_seen > 0 and r["inference_ms"]["chunks"] > 0
  assert r["sim_s"] <= len(res.action) / 30 + 0.2


def test_move_policy_that_does_nothing_fails(tmp_path, monkeypatch):
  monkeypatch.setattr(E, "ObsRenderer", BlankRenderer)
  cfg = _cfg(tmp_path, "policy", "move", max_sim_s=3.0)
  v, info = SV._eval_variation(SPEC, cfg, 3)
  h = SV.StackHarness(SPEC, v)
  still = h.state_mm2()
  r = SV.episode(3, cfg, SPEC, {"runner": ExpertPolicy(still[None])})
  assert r["stop_reason"] == "time_limit" and not r["L_task"] and not r["L_full"]
  assert "never_carried" in r["reasons"]


def test_tower_expert_mode_from_scatter_and_wrong_stack(tmp_path):
  cfg = _cfg(tmp_path, "expert", "tower")
  rows = [SV.episode(i, cfg, SPEC, {}) for i in (2, 3)]
  assert [r["start_kind"] for r in rows] == ["scatter", "wrong"]
  for r in rows:
    assert r["solved"] and r["advancement"] == 1.0 and r["final_state"] == K.GOAL_STATE, r["reasons"]
    assert r["moves_used"] == r["rule_moves"]
    assert [tuple(s["expected_move"]) for s in r["segments"]] == K.rule_moves_to_goal(r["start_state"])
    assert r["expert_native"]["expert_verifier_pass"]
  s = SV.summarize(rows, cfg)
  assert s["all"]["solved"] == 2 and set(s["by_start_kind"]) == {"scatter", "wrong"}


def test_tower_policy_that_does_nothing(tmp_path, monkeypatch):
  monkeypatch.setattr(E, "ObsRenderer", BlankRenderer)
  cfg = _cfg(tmp_path, "policy", "tower", move_max_s=2.0, max_moves=2)
  v, info = SV._eval_variation(SPEC, cfg, 0, bucket="scatter")
  h = SV.StackHarness(SPEC, v)
  r = SV.episode(0, cfg, SPEC, {"runner": ExpertPolicy(h.state_mm2()[None])})
  assert not r["solved"] and r["moves_used"] == 2 and r["advancement"] == 0.0
  assert r["stop_reason"] == "max_moves" and "not_solved" in r["reasons"]
