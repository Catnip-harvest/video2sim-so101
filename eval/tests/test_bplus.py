"""Unit tests for the filtered-BC (B+) pieces (CPU, no GL, ~15 s).

  python -m pytest -q -p no:cacheprovider eval/tests/test_bplus.py
The end-to-end chain (rollouts -> dataset -> control A -> merge -> round-2 config) is
exercised by `TEST=1 bash eval/bplus_run.sh` (expert stands in for the policy, ~2 min).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import compare as C  # noqa: E402
import sim_eval as E  # noqa: E402
from r2s.variations import sample_variation  # noqa: E402
from r2s.verify_policy import run_expert_judged  # noqa: E402


class _Seq:
  needs_images = False

  def __init__(self, a):
    self.a = np.asarray(a, float)

  def reset(self):
    pass

  def needs_obs(self):
    return False

  def act(self, k, s, i):
    return None if k >= len(self.a) else self.a[k]


def test_ema():
  a = np.zeros((5, 6))
  a[1:] = 10.0
  src = E.EMASource(_Seq(a), 0.5)
  src.reset()
  out = [src.act(k, None, None) for k in range(5)]
  assert np.allclose([o[0] for o in out], [0, 5, 7.5, 8.75, 9.375])
  assert src.act(5, None, None) is None


def test_blend_crossfade():
  class R:  # chunk = constant c_k at every call k
    def chunk(self, s, i, k=0):
      return np.full((10, 6), float(k))
  b = E.BlendSource(R(), every=5)
  b.reset()
  out = [b.act(k, None, None)[0] for k in range(12)]
  assert out[:5] == [0, 0, 0, 0, 0]
  # at k=5 the new chunk (value 5) fades in over the 5 overlapping steps: w = 1/6 .. 5/6
  assert np.allclose(out[5:10], [5 * w for w in np.arange(1, 6) / 6])
  assert b.wants_obs(15) and not b.wants_obs(11)  # last re-plan was at k=10


def test_blend_on_consistent_chunks_is_identity():
  spec = E.load_task_spec(E.SPEC)
  v = sample_variation(spec, 1000, 0, variety=True)
  res, _ = run_expert_judged(spec, v, keep_arrays=True)
  a10 = E.sim_to_mm2(res.action[::3])
  b = E.BlendSource(E.ExpertChunkRunner(a10), every=5)
  b.reset()
  out = np.array([b.act(k, None, None) for k in range(len(a10))])
  assert np.abs(out - a10).max() < 1e-4


def test_mcnemar_and_compare():
  assert abs(C.mcnemar(10, 2)["p_exact"] - 0.03857421875) < 1e-12  # = scipy.stats.binomtest(2, 12)
  a = {(1000, i): {"L_full": i < 8, "L_task": True} for i in range(10)}
  b = {(1000, i): {"L_full": i >= 2, "L_task": True} for i in range(10)}
  r = C.compare(a, b, "L_full")
  assert (r["n_paired"], r["a_success"], r["b_success"], r["a_pass_b_fail"], r["a_fail_b_pass"]) == (10, 8, 8, 2, 2)
  assert r["p_exact"] == 1.0
