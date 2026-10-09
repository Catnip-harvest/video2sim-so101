"""Phase-free verifier tests (CPU MuJoCo only, no GL; ~1 min).

  python -m pytest -q -p no:cacheprovider datagen/tests/test_verify_policy.py

The 60-episode agreement study is datagen/tests/verify_agreement.py (results in
/root/work/gates/gate_eval_harness.md); these are the fast versions of the same checks.
All seeds are eval seeds (--seed 1000, variety on); the training set used --seed 1.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

DATAGEN = Path(__file__).resolve().parents[1]
ROOT = DATAGEN.parent
for p in (DATAGEN, ROOT / "eval", ROOT / "calib"):
  if str(p) not in sys.path:
    sys.path.insert(0, str(p))

from generate import pinch_width_from_spec  # noqa: E402
from r2s.paths import load_task_spec, mj_scene  # noqa: E402
from r2s.planner import plan_config_for  # noqa: E402
from r2s.variations import sample_variation  # noqa: E402
from r2s.verify_policy import TASK_CLAUSES, run_expert_judged  # noqa: E402

SPEC_PATH = ROOT / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SPEC = load_task_spec(SPEC_PATH)
PW = pinch_width_from_spec(SPEC_PATH)
SEED = 1000


def _run(i, servo_foh=True, **cfg_over):
  v = sample_variation(SPEC, SEED, i, variety=True)
  cfg = replace(plan_config_for(v), **cfg_over) if cfg_over else None
  return run_expert_judged(SPEC, v, cfg, servo_foh=servo_foh, pinch_width=PW, keep_arrays=False)


def test_expert_verdicts_agree():
  # (a) small version: same simulation, both verifiers, labels unused by the phase-free one
  for i in range(4):
    res, pf = _run(i)
    assert res.accepted == pf["passed"], (i, res.reasons, pf["reasons"])
    if res.accepted:
      assert pf["L_task"] and len(pf["stats"]["carries"]) == 1
      c = pf["stats"]["carries"][0]
      assert c["opening"] and c["in_bin"] and c["bottom_above_surface_mm"] <= 10.0


def test_jaw_never_closes_fails():
  # (b) negative control: PlanConfig(close_jaw=False)
  for i in range(2):
    res, pf = _run(i, close_jaw=False)
    assert not res.accepted and not pf["passed"] and not pf["L_task"]
    assert "never_carried" in pf["reasons"]


def test_cube_in_bin_without_carry_fails_on_both_jaw_clause():
  # (c) the "push/scoop it in" family: the cube ends resting in the bin, released, but no
  # both-jaw carry ever happened. Only the both-jaw clauses may fail it. (A literal push
  # cannot reach this bin: its walls are 25 mm high, taller than the 20-26 mm cube.)
  from sim_eval import ReplaySource, run_harness_episode, sim_to_mm2
  v = sample_variation(SPEC, SEED, 0, variety=True)
  v2 = replace(v, obj_xy=list(v.bin_xy))
  z0 = mj_scene.bin_floor_top() + v.obj_size / 2 + 0.0005
  still = np.repeat(sim_to_mm2(np.r_[v.q_start, v.gripper_start])[None], 30, 0)  # 3 s, arm still
  r = run_harness_episode(SPEC, v2, ReplaySource(still), tail_s=0.3, init_obj_z=z0)
  vd = r["verdict"]
  assert not vd["L_task"] and not vd["passed"]
  assert set(vd["task_reasons"]) == {"never_carried", "not_carried_into_bin"}, vd["reasons"]
  assert "not_in_place_region" not in vd["reasons"]  # it IS in the bin, resting
  assert "not_resting" not in vd["reasons"] and "not_released" not in vd["reasons"]


def test_zoh_drop_fails_like_expert():
  # (d) zero-order-hold servos shake the cube out of the jaws: attempts 10 and 14 of seed
  # 1000 drop it (expert verifier: dropped_or_slipped), measured in verify_agreement.py.
  for i in (10, 14):
    res, pf = _run(i, servo_foh=False)
    assert not res.accepted and "dropped_or_slipped" in res.reasons, res.reasons
    assert not pf["passed"] and "dropped_or_slipped" in pf["reasons"], pf["reasons"]


def test_high_release_fails_new_clause():
  # new clause: opening 6 cm above the floor passes the expert verifier but not this one
  res, pf = _run(0, release_gap=0.06)
  assert res.accepted
  assert not pf["passed"] and "release_too_high" in pf["reasons"]
  assert pf["L_task"]  # the task itself (cube released, resting in the bin) is done


def test_task_clauses_are_subset():
  assert set(TASK_CLAUSES) >= {"never_carried", "not_in_place_region", "not_released"}
