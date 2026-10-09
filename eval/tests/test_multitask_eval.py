"""P3-A-lite eval helpers (CPU only; ~1 min).

  python -m pytest -q -p no:cacheprovider eval/tests/test_multitask_eval.py

P3-A demos come from a SCRIPTED EXPERT, not a human video.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import multitask as MT  # noqa: E402
import sim_eval as E  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.variations import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, sample_multitask_variation  # noqa: E402

SPEC = load_task_spec(E.SPEC)


def test_named_outcome():
  assert MT.named_outcome({"red": True, "green": False, "blue": False}, "red") == "named"
  assert MT.named_outcome({"red": False, "green": True, "blue": False}, "red") == "other"
  assert MT.named_outcome({"red": False, "green": False, "blue": False}, "red") == "none"
  assert MT.named_outcome({"red": True, "green": True, "blue": False}, "red") == "multiple"


def test_eval_rows_scenes():
  items = MT.eval_items("swap", [0, 1])
  assert items == [(0, "red"), (0, "green"), (0, "blue"), (1, "red"), (1, "green"), (1, "blue")]
  for i in range(6):
    tr = MT.eval_variation(SPEC, 5000, i, "train")
    pa = MT.eval_variation(SPEC, 5000, i, "paraphrase")
    assert MT.scene_signature(tr) == MT.scene_signature(pa)  # same physical scene
    assert tr.extras["multitask"]["template"] in TRAIN_TEMPLATES
    assert pa.extras["multitask"]["template"] in HELDOUT_TEMPLATES
    assert tr.extras["multitask"]["target_colour"] == pa.extras["multitask"]["target_colour"]
    sw = [MT.eval_variation(SPEC, 5000, i, "swap", c) for c in MT.SWAP_COLOURS]
    assert all(MT.scene_signature(s) == MT.scene_signature(sw[0]) for s in sw)
    assert set(MT.scene_signature(sw[0])) == {"red", "green", "blue"}
    assert [s.extras["multitask"]["target_colour"] for s in sw] == list(MT.SWAP_COLOURS)
    assert all(s.extras["multitask"]["target_colour"] in s.extras["multitask"]["instruction"] for s in sw)
    y = MT.eval_variation(SPEC, 5000, 100 + i, "yellow")
    assert y.extras["multitask"]["target_colour"] == "yellow"


def test_leakage_helper(tmp_path):
  train = [sample_multitask_variation(SPEC, 2, i) for i in range(30)]
  side = tmp_path / "datagen_episodes.jsonl"
  side.write_text("\n".join(json.dumps({"episode_index": k, "variation": v.to_dict()})
                            for k, v in enumerate(train)) + "\n")
  sigs = MT.training_signatures([side])
  assert len(sigs) == 30
  evs = [MT.eval_variation(SPEC, 5000, i, "train") for i in range(30)]
  assert MT.leakage(evs, sigs) == []
  # a planted duplicate (0.5 mm away) is caught; 2 mm away is not a match
  dup = train[7].to_dict()
  d2 = json.loads(json.dumps(dup))
  d2["obj_xy"][0] += 0.0005
  from r2s.variations import Variation
  assert MT.leakage([Variation.from_dict(d2)], sigs) == [(0, 7)]
  d2["obj_xy"][0] += 0.0015
  assert MT.leakage([Variation.from_dict(d2)], sigs) == []


def test_swap_scoring_with_the_expert(tmp_path):
  """The expert always moves the red cube (the planner's object): named for red, other
  for green and blue - the scorer reads colours, not body names."""
  cfg = {"mode": "expert", "seed": 5000, "spec": str(E.SPEC), "max_sim_s": 40.0,
         "interp": "lag", "quality": "fast", "tail_s": 0.3, "video_ids": [], "out": str(tmp_path),
         "pinch_width": None, "device": "cpu", "workers": 1, "obs_codec": False,
         "smooth": "none", "multitask": "swap"}
  from generate import pinch_width_from_spec
  cfg["pinch_width"] = pinch_width_from_spec(Path(E.SPEC))
  E._init(cfg)
  rows = []
  for i in range(3):  # the first scene whose expert run is accepted
    rows = [E._episode((i, c)) for c in MT.SWAP_COLOURS]
    if rows[0]["expert_native"]["expert_verifier_pass"]:
      break
  out = {r["named_cube"]["named"]: r["named_cube"]["outcome"] for r in rows}
  assert out == {"red": "named", "green": "other", "blue": "other"}, out
  assert [r["L_task_named"] for r in rows] == [True, False, False]
  # the expert-side verifier also rejects the expert's motion when another cube is named
  assert [r["expert_native"]["expert_verifier_pass"] for r in rows] == [True, False, False]
  assert all("wrong_object_moved" in r["expert_native"]["expert_reasons"] for r in rows[1:])
  s = MT.summarize_named(rows)
  assert s["outcomes"] == {"named": 1, "other": 2, "none": 0, "multiple": 0}


def test_policy_runner_set_task_drops_cached_batch():
  from policy_io import PolicyRunner

  class FakePolicy:
    def reset(self):
      self.was_reset = True

  r = PolicyRunner.__new__(PolicyRunner)  # no checkpoint load (CPU unit test)
  r.policy, r.pre, r.post, r.task = FakePolicy(), object(), object(), "old"
  r._last_batch = {"tokens": "for the old task"}
  r.set_task("put the blue cube in the tray")
  assert r.task == "put the blue cube in the tray" and r._last_batch is None
  r._last_batch = {"tokens": "stale"}
  r.reset()
  assert r._last_batch is None and r.policy.was_reset
