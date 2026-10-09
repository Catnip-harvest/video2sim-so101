"""P3-A-lite: named cube among three (CPU MuJoCo only, no GL; ~1-2 min).

  python -m pytest -q -p no:cacheprovider datagen/tests/test_multitask.py

The expert data for P3-A comes from a SCRIPTED EXPERT (our planner), not a human video.
These tests check the scene, the instruction bookkeeping and the verifier clauses:
distractors stay put (< 1 mm), a wrong cube in the tray or a wrong cube moved is
rejected, the model has no welds / mocap / equality constraints, and old single-cube
Variations still round-trip.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

DATAGEN = Path(__file__).resolve().parents[1]
if str(DATAGEN) not in sys.path:
  sys.path.insert(0, str(DATAGEN))

from generate import episode_task, pinch_width_from_spec, select_accepted  # noqa: E402
from r2s import scene as S  # noqa: E402
from r2s.free_bodies import _dist_to_square  # noqa: E402
from r2s.paths import load_task_spec, mj_scene  # noqa: E402
from r2s.rollout import run_episode  # noqa: E402
from r2s.variations import (HELDOUT_TEMPLATES, NAMED_COLOURS, PAN_AXIS, TRAIN_TARGETS,  # noqa: E402
                            TRAIN_TEMPLATES, Variation, named_body, sample_multitask_variation,
                            sample_variation)

SUGAR = DATAGEN.parent / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SPEC = load_task_spec(SUGAR)
PINCH = pinch_width_from_spec(SUGAR)
OLD_SIDECAR = Path("/work/outputs_large/datagen/sugar_v10_n1000/lerobot/meta/datagen_episodes.jsonl")


def _cubes(v: Variation) -> dict[str, np.ndarray]:
  out = {"object": np.asarray(v.obj_xy, float)}
  for b in v.extras["free_bodies"]:
    out[b["name"]] = np.asarray(b["xy"], float)
  return out


def test_old_single_cube_variations_round_trip():
  for i in range(10):
    v = sample_variation(SPEC, 1, i, variety=True)
    d = json.loads(json.dumps(v.to_dict()))
    assert Variation.from_dict(d).to_dict() == v.to_dict()
    assert "multitask" not in v.extras and "free_bodies" not in v.extras
    assert named_body(v) == "object"
  if OLD_SIDECAR.exists():  # a variation written by the real n=1000 cube run
    row = json.loads(OLD_SIDECAR.read_text().splitlines()[0])
    v = Variation.from_dict(row["variation"])
    assert v.to_dict() == sample_variation(SPEC, 1, v.attempt, variety=True).to_dict()
    sc = S.build_episode_scene(v, object_kind=SPEC.object.kind)
    assert sc.extra_bodies == ()


def test_multitask_variation_layout_colours_and_instruction():
  seen_t, seen_slot, seen_tpl = set(), set(), set()
  for i in range(150):
    v = sample_multitask_variation(SPEC, 2, i)
    assert sample_multitask_variation(SPEC, 2, i).to_dict() == v.to_dict()  # deterministic
    assert Variation.from_dict(json.loads(json.dumps(v.to_dict()))).to_dict() == v.to_dict()
    mt = v.extras["multitask"]
    cols = mt["colours"]
    assert len(set(cols.values())) == 3 and set(cols.values()) <= set(NAMED_COLOURS)
    assert mt["target_colour"] in TRAIN_TARGETS and cols["object"] == mt["target_colour"]
    assert named_body(v) == "object"
    assert mt["template"] in TRAIN_TEMPLATES
    assert mt["instruction"] == mt["template"].format(c=mt["target_colour"])
    assert len(v.extras["free_bodies"]) == 2
    for b in v.extras["free_bodies"]:
      assert b["shape"] == "box" and 0.020 <= b["size"] <= 0.050
    xy = list(_cubes(v).values())
    bin_half = mj_scene.BIN_INNER_HALF + mj_scene.BIN_WALL_THICKNESS
    for a in range(3):
      r = np.linalg.norm(xy[a] - PAN_AXIS)
      assert 0.12 - 1e-9 <= r <= 0.26 + 1e-9
      assert _dist_to_square(xy[a], np.asarray(v.bin_xy), bin_half) >= 0.06 - 1e-9
      for b in range(a + 1, 3):
        assert np.linalg.norm(xy[a] - xy[b]) >= 0.06 - 1e-9
    seen_t.add(mt["target_colour"])
    seen_slot.add(mt["target_slot"])
    seen_tpl.add(mt["template"])
  assert seen_t == set(TRAIN_TARGETS) and seen_slot == {0, 1, 2} and seen_tpl == set(TRAIN_TEMPLATES)
  # balanced targets by attempt id; yellow only as a distractor unless explicitly allowed
  assert [sample_multitask_variation(SPEC, 2, i).extras["multitask"]["target_colour"]
          for i in range(3)] == list(TRAIN_TARGETS)
  with pytest.raises(ValueError):
    sample_multitask_variation(SPEC, 2, 0, target_colour="yellow")
  v = sample_multitask_variation(SPEC, 5000, 100, target_colour="yellow",
                                 template=HELDOUT_TEMPLATES[0], allow_yellow_target=True)
  assert v.extras["multitask"]["instruction"] == "move the yellow one to the tray"
  # same physical layout for every named colour of an eval scene (instruction swap)
  a = sample_multitask_variation(SPEC, 5000, 3, target_colour="red")
  b = sample_multitask_variation(SPEC, 5000, 3, target_colour="blue")
  assert a.bin_xy == b.bin_xy


def test_model_has_no_welds_and_cubes_are_free_bodies():
  v = sample_multitask_variation(SPEC, 2, 0)
  sc = S.build_episode_scene(v, object_kind=SPEC.object.kind)
  m = sc.model
  assert m.neq == 0 and m.nmocap == 0
  assert len(sc.extra_bodies) == 2
  for b in sc.extra_bodies:
    j = m.body_jntadr[b]
    assert m.body_jntnum[b] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    assert m.body_mocapid[b] == -1
  assert int(np.sum(sc.geom_cat == S.CAT_EXTRA)) == 2


def test_episode_task_and_quota_selection():
  rows = {}
  for i in range(12):
    v = sample_multitask_variation(SPEC, 2, i)
    rows[i] = {"accepted": i not in (1, 4), "variation": v.to_dict()}
  assert episode_task(rows[0], "x") == rows[0]["variation"]["extras"]["multitask"]["instruction"]
  assert episode_task({"variation": sample_variation(SPEC, 1, 0).to_dict()}, "x") == "x"
  sel = select_accepted(rows, 6, quota=2)
  tg = [rows[i]["variation"]["extras"]["multitask"]["target_colour"] for i in sel]
  assert len(sel) == 6 and all(tg.count(t) == 2 for t in TRAIN_TARGETS)
  assert sel == sorted(sel)


@pytest.fixture(scope="module")
def accepted_episode():
  for i in range(6):
    v = sample_multitask_variation(SPEC, 2, i)
    r = run_episode(SPEC, v, pinch_width=PINCH, keep_arrays=False)
    if r.accepted:
      return v, r
  pytest.fail("no accepted multitask episode in 6 attempts")


def test_expert_accepts_with_distractors_still(accepted_episode):
  v, r = accepted_episode
  assert max(r.stats["distractor_max_move_mm"]) < 1.0
  assert max(r.stats["distractor_max_move_pick_mm"]) < 1.0
  assert r.stats["carry_both_jaw_steps"] == r.stats["carry_steps"] > 0  # both jaws carried it


def test_wrong_object_negative_control_is_rejected(accepted_episode):
  """Same scene and the same expert motion, but the instruction names another cube
  (colour labels swapped between the moved cube and extra_0): must be rejected."""
  v, _ = accepted_episode
  w = copy.deepcopy(v)
  mt = w.extras["multitask"]
  other = w.extras["free_bodies"][0]
  mt["colours"]["object"], mt["colours"]["extra_0"] = mt["colours"]["extra_0"], mt["colours"]["object"]
  w.object_rgb, other["rgb"] = other["rgb"], w.object_rgb
  assert named_body(w) == "extra_0"
  r = run_episode(SPEC, w, pinch_width=PINCH, keep_arrays=False)
  assert not r.accepted and "wrong_object_moved" in r.reasons, r.reasons


def test_distractor_in_tray_is_rejected(accepted_episode):
  v, _ = accepted_episode
  w = copy.deepcopy(v)
  # extra_0 starts in a corner of the tray, away from where the expert releases.
  w.extras["free_bodies"][0]["xy"] = [w.bin_xy[0] + 0.035, w.bin_xy[1] + 0.035]
  w.extras["free_bodies"][0]["xy"][1] += 0.0  # (on the bin floor after settling)
  r = run_episode(SPEC, w, pinch_width=PINCH, keep_arrays=False)
  assert not r.accepted and "wrong_object_in_tray" in r.reasons, r.reasons


def test_bumped_distractor_is_rejected(accepted_episode):
  """A distractor 28 mm from the named cube against the closing axis (where a jaw comes
  down in the accepted run) gets hit: the verifier must see it move and reject."""
  v, r0 = accepted_episode
  w = copy.deepcopy(v)
  closing = np.asarray(r0.plan_info["pick"]["closing"])[:2]
  w.extras["free_bodies"][0]["xy"] = (np.asarray(w.obj_xy) - closing * 0.028).tolist()
  r = run_episode(SPEC, w, pinch_width=PINCH, keep_arrays=False)
  assert not r.accepted and "distractor_disturbed" in r.reasons, r.reasons
  assert max(r.stats["distractor_max_move_mm"]) >= 1.0
  assert r.stats["distractor_contacts"]["arm"] > 0
