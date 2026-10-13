"""Tower-of-Hanoi eval harness (CPU only, no GPU; ~2-3 min).

  python -m pytest -q -s -p no:cacheprovider eval/tests/test_hanoi_eval.py

Policy mode is exercised with a stand-in runner that plays the scripted expert's 10 Hz
actions (no checkpoint, no GPU); without OpenGL its camera renders are replaced by blank
frames. The render test at the bottom uses the real ObsRenderer and is skipped when no
OpenGL backend exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import hanoi_eval as HV  # noqa: E402
import sim_eval as E  # noqa: E402
from generate import pinch_width_from_spec  # noqa: E402
from r2s import hanoi as H  # noqa: E402
from r2s.hanoi_expert import run_hanoi_episode  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.render import gl_available  # noqa: E402

SPEC = load_task_spec(E.SPEC)
PINCH = pinch_width_from_spec(Path(E.SPEC))
GL_OK, GL_MSG = gl_available()


def _cfg(tmp_path, mode: str, hanoi: str, **kw) -> dict:
  cfg = {"mode": mode, "hanoi": hanoi, "seed": 1000, "out": str(tmp_path), "quality": "fast",
         "obs_codec": False, "interp": "lag", "max_sim_s": 40.0, "move_max_s": 40.0,
         "max_moves": 12, "tail_s": 0.3, "pinch_width": PINCH, "smooth": "none",
         "ema_alpha": 0.5, "replan_every": 5, "video_ids": [], "workers": 1, "checkpoint": None}
  cfg.update(kw)
  return cfg


class ExpertPolicy:
  """Stand-in for policy_io.PolicyRunner: plays fixed 10 Hz actions, one per call, and
  repeats the last one when they run out (a policy never "runs out")."""

  def __init__(self, a10: np.ndarray):
    self.a, self.k = np.asarray(a10), 0
    self.chunk_ms, self.call_ms = [], []
    self.task = ""
    self.images_seen = 0

  def set_task(self, task):
    self.task = task

  def reset(self):
    self.k = 0

  def needs_obs(self):
    return self.k % 10 == 0

  def act(self, state, images):
    if images is not None:
      assert set(images) == set(E.CAMS)
      self.images_seen += 1
    a = self.a[min(self.k, len(self.a) - 1)]
    self.k += 1
    self.chunk_ms.append(1.0)
    return a


class BlankRenderer:
  def __init__(self, *a, **k):
    pass

  def render(self, qpos):
    return {k: np.zeros((256, 256, 3), np.uint8) for k in E.CAMS}

  def close(self):
    pass


def test_harness_start_matches_datagen():
  v = H.sample_hanoi_variation(SPEC, 1000, 0)
  res = run_hanoi_episode(SPEC, v, pinch_width=PINCH)
  h = HV.HanoiHarness(SPEC, v)
  assert np.array_equal(h.sc.data.qpos, res.qpos_full[0])  # same build, placement, settle


def test_move_expert_mode(tmp_path):
  cfg = _cfg(tmp_path, "expert", "move")
  rows = [HV.episode(i, cfg, SPEC, {}) for i in range(3)]
  for r in rows:
    assert r["expert_native"]["expert_verifier_pass"]
    assert r["L_task"] and r["L_full"], r["reasons"]
    assert r["observed_end_state"] == r["expected_end_state"]
    assert Path(r["rollout_file"]).exists()
  s = HV.summarize(rows, cfg)
  assert s["end_state_matches_expected"] == 3
  assert sum(x["n"] for x in s["by_moved_cube"].values()) == 3
  full = E.summarize(rows, cfg, 1.0, None)  # sim_eval's summary accepts Hanoi rows
  assert full["success_L_full"] == 3 and full["hanoi"]["task"] == "move"


def test_move_policy_mode_stops_when_the_move_is_done(tmp_path, monkeypatch):
  monkeypatch.setattr(E, "ObsRenderer", BlankRenderer)
  cfg = _cfg(tmp_path, "policy", "move")
  v = H.sample_hanoi_variation(SPEC, 1000, 1)
  res = run_hanoi_episode(SPEC, v, pinch_width=PINCH)
  runner = ExpertPolicy(E.sim_to_mm2(res.action[::E.STEP_FRAMES]))
  r = HV.episode(1, cfg, SPEC, {"runner": runner})
  assert r["stop_reason"] == "move_done_1s", r["stop_reason"]
  assert r["L_task"] and r["L_full"], r["reasons"]
  assert runner.images_seen > 0 and r["inference_ms"]["chunks"] > 0
  assert r["sim_s"] < len(res.action) / 30  # stopped before the expert's actions ran out


def test_move_policy_that_does_nothing_fails(tmp_path, monkeypatch):
  monkeypatch.setattr(E, "ObsRenderer", BlankRenderer)
  cfg = _cfg(tmp_path, "policy", "move", max_sim_s=3.0)
  v = H.sample_hanoi_variation(SPEC, 1000, 2)
  h = HV.HanoiHarness(SPEC, v)
  still = h.state_mm2()
  r = HV.episode(2, cfg, SPEC, {"runner": ExpertPolicy(still[None])})
  assert r["stop_reason"] == "time_limit" and not r["L_task"] and not r["L_full"]
  assert "never_carried" in r["reasons"]


def test_puzzle_expert_mode_solves_in_seven(tmp_path):
  cfg = _cfg(tmp_path, "expert", "puzzle")
  r = HV.episode(0, cfg, SPEC, {})
  print(f"\n[hanoi eval] puzzle expert: solved={r['solved']} moves={r['moves_used']} "
        f"illegal={r['illegal_states']} sim={r['sim_s']} s wall={r['wall_s']} s")
  assert r["solved"] and r["moves_used"] == 7 and r["illegal_states"] == 0 and r["solved_optimally"]
  assert [tuple(s["expected_move"]) for s in r["segments"]] == H.optimal_moves_between("AAA")
  assert all(s["L_full"] for s in r["segments"]), [s["reasons"] for s in r["segments"]]
  assert r["expert_native"]["expert_verifier_pass"] and r["expert_native"]["moves_used"] == 7
  s = HV.summarize([r], cfg)
  assert s["solved"] == 1 and s["solved_in_7"] == 1 and s["illegal_states_total"] == 0


def test_puzzle_policy_mode_bookkeeping(tmp_path, monkeypatch):
  """A "policy" that only ever makes the first optimal move, then holds still: the puzzle
  is not solved, each later move-episode ends at the time limit, nothing is illegal."""
  monkeypatch.setattr(E, "ObsRenderer", BlankRenderer)
  cfg = _cfg(tmp_path, "policy", "puzzle", max_moves=3, move_max_s=30.0)
  v = H.sample_hanoi_variation(SPEC, 1000, 0, start_state="AAA")
  res = run_hanoi_episode(SPEC, v, pinch_width=PINCH)
  a10 = E.sim_to_mm2(res.action[::E.STEP_FRAMES])

  class OneMove(ExpertPolicy):
    def __init__(self, a):
      super().__init__(a)
      self.episodes = 0

    def reset(self):
      super().reset()
      self.episodes += 1
      if self.episodes > 1:  # later move-episodes: hold the last action
        self.a = self.a[-1:]

  r = HV.episode(0, cfg, SPEC, {"runner": OneMove(a10)})
  seg = r["segments"]
  assert not r["solved"] and r["moves_used"] == 3 and r["illegal_states"] == 0
  assert seg[0]["stop_reason"] == "move_done_1s" and seg[0]["observed_state"] == "AAC"
  assert seg[0]["L_task"] and seg[0]["L_full"]
  assert all(s["stop_reason"] == "time_limit" and s["observed_state"] == "AAC" for s in seg[1:])
  assert r["reasons"][0] == "not_solved"


@pytest.mark.skipif(not GL_OK, reason=f"needs an OpenGL backend (MUJOCO_GL=egl/osmesa): {GL_MSG}")
def test_policy_mode_with_real_renders(tmp_path):
  cfg = _cfg(tmp_path, "policy", "move", obs_codec=True, video_ids=["3"])
  v = H.sample_hanoi_variation(SPEC, 1000, 3)
  res = run_hanoi_episode(SPEC, v, pinch_width=PINCH)
  runner = ExpertPolicy(E.sim_to_mm2(res.action[::E.STEP_FRAMES]))
  r = HV.episode(3, cfg, SPEC, {"runner": runner})
  assert runner.images_seen > 0 and r["timing_s"]["render"] > 0
  assert r["L_task"] and Path(r["video"]).exists()
