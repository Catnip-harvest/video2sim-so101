"""Tests for eval/scoreboard.py (CPU only, < 2 s).

  python -m pytest -q -p no:cacheprovider eval/tests/test_scoreboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import scoreboard as SB  # noqa: E402


def test_wilson_hand_value():
  # 8/10 by hand: p=0.8, z^2=3.8416 -> (0.99208 -+ 0.313624) / 1.38416
  lo, hi = SB.wilson(8, 10)
  assert lo == pytest.approx(0.49016, abs=1e-4) and hi == pytest.approx(0.94332, abs=1e-4)


def test_wilson_matches_harness_summary():
  # policy_final_n200_seed3000/summary.json: 137/200 -> [0.6176, 0.7454]
  lo, hi = SB.wilson(137, 200)
  assert round(lo, 4) == 0.6176 and round(hi, 4) == 0.7454
  assert SB.cell(137, 200) == "137/200 (61.8-74.5%)"


def test_mcnemar_hand_value():
  # b=10, c=2: 2 * (1 + 12 + 66) / 4096 = 0.03857
  assert SB.mcnemar_exact(10, 2) == pytest.approx(0.0386, abs=1e-4)
  assert SB.mcnemar_exact(2, 10) == SB.mcnemar_exact(10, 2)
  assert SB.mcnemar_exact(0, 0) == 1.0
  assert SB.mcnemar_exact(5, 5) == 1.0


def test_mcnemar_matches_scipy():
  st = pytest.importorskip("scipy.stats")
  for b in range(0, 25, 3):
    for c in range(0, 25, 4):
      if b + c == 0:
        continue
      ref = st.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
      assert SB.mcnemar_exact(b, c) == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_holm_toy():
  # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
  assert SB.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
  assert SB.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
  assert SB.holm([]) == []


def _run(tmp: Path, name: str, seed: int, ood: str, full: list[int], task: list[int]) -> str:
  d = tmp / name
  d.mkdir()
  (d / "config.json").write_text(json.dumps({"mode": "policy", "ood": ood, "replan_every": 5}))
  with open(d / "episodes.jsonl", "w") as f:
    for i, (lf, lt) in enumerate(zip(full, task)):
      f.write(json.dumps({"seed": seed, "attempt": i, "episode_seed": seed * 10**6 + i, "L_full": bool(lf),
                          "L_task": bool(lt), "reasons": [], "inference_ms": {"chunk_median": 100 + i}}) + "\n")
    f.write('{"seed": 4000, "attem')  # half-written line of a live run must be skipped
  return str(d)


def test_end_to_end(tmp_path):
  n = 30
  a_none = _run(tmp_path, "a_none", 4000, "none", [1] * n, [1] * n)
  a_cam = _run(tmp_path, "a_cam", 4000, "camera", [1] * 5 + [0] * 25, [1] * 10 + [0] * 20)
  b_none = _run(tmp_path, "b_none", 4000, "none", [1] * 28 + [0] * 2, [1] * n)
  row = dict(params="-", hardware="MI300X", exec_horizon="auto", ms_chunk="auto", expected_n=n)
  man = {"tables": [{"title": "T", "rows": [
    dict(row, model="A", test="none", rung="none", runs=[a_none]),
    dict(row, model="A", test="camera", rung="camera", runs=[a_cam]),
    dict(row, model="B", test="none", rung="none", runs=[b_none]),
    dict(row, model="C", test="none", rung="none", runs=[str(tmp_path / "missing")]),
    dict(row, model="D", test="none", rung="none", runs=[str(tmp_path / "missing")], hide_if_missing=True),
  ]}]}
  text = SB.render(man, "m.json")
  assert "| A | - | MI300X | none | 30/30 (" in text
  assert "| C | - | MI300X | none | - | - | - |" in text
  assert "| D |" not in text
  assert "| 5 |" in text  # exec horizon auto from replan_every
  # 1 model pair (A none vs B none) + 1 drop (A camera) = 2 pairs x 2 metrics = 4 Holm tests
  assert "Holm across all 4 tests" in text
  assert "not distinguishable at n=30" in text  # A vs B: b=2, c=0
  assert "significant drop" in text  # A none vs camera on L_full: b=25, c=0
  assert "skipped 1 unparsable line(s)" in text


def test_manifest_is_valid():
  man = json.loads((ROOT / "eval" / "scoreboard_manifest.json").read_text())
  for t in man["tables"]:
    for r in t["rows"]:
      assert {"model", "test"} <= set(r) and isinstance(r.get("runs", []), list)
