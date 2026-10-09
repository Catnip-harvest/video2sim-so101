"""Harness tests (CPU only, no GL, no GPU; ~20 s).

  python -m pytest -q -p no:cacheprovider eval/tests/test_sim_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import sim_eval as E  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.variations import sample_variation  # noqa: E402
from r2s.verify_policy import run_expert_judged  # noqa: E402

SPEC = load_task_spec(E.SPEC)


def test_units_round_trip():
  rng = np.random.default_rng(0)
  q = rng.uniform(-1.5, 1.5, (100, 6))
  back = E.mm2_to_sim(E.sim_to_mm2(q))
  assert np.abs(back - q).max() < 1e-5  # float32 storage, like the dataset


def test_wilson():
  lo, hi = E.wilson(19, 20)
  assert abs(lo - 0.7639) < 1e-3 and abs(hi - 0.9911) < 1e-3
  assert E.wilson(0, 20)[0] == 0.0


def test_initial_state_matches_expert_rollout():
  # The harness scene after the settle is bit-identical to rollout.run_episode's.
  v = sample_variation(SPEC, 1000, 0, variety=True)
  res, _ = run_expert_judged(SPEC, v, keep_arrays=True)
  sc = E.setup_scene(SPEC, v)
  assert np.array_equal(sc.data.qpos, res.qpos_full[0])


def test_expert_through_harness_passes():
  # Expert's 30 Hz goals -> every 3rd (the dataset's 10 Hz) -> MolmoAct2 frame -> harness.
  v = sample_variation(SPEC, 1000, 0, variety=True)
  res, pf = run_expert_judged(SPEC, v, keep_arrays=True)
  assert res.accepted and pf["passed"]
  a10 = E.sim_to_mm2(res.action[::E.STEP_FRAMES])
  r = E.run_harness_episode(SPEC, v, E.ReplaySource(a10), tail_s=0.3)
  assert r["verdict"]["passed"], r["verdict"]["reasons"]
  assert r["clip"]["frames_clipped"] == 0 and r["clip"]["steps_range_clipped"] == 0


def test_hold_still_fails():
  v = sample_variation(SPEC, 1000, 1, variety=True)
  still = np.repeat(E.sim_to_mm2(np.r_[v.q_start, v.gripper_start])[None], 20, 0)
  r = E.run_harness_episode(SPEC, v, E.ReplaySource(still))
  assert not r["verdict"]["L_task"] and "never_carried" in r["verdict"]["reasons"]
