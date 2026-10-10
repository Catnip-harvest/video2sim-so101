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


# ----------------------------------------------------------------------------- single-task snapshot

# sha256 of the "## Single-task cube (P1 / P2)" section of /work/SCOREBOARD.md as rendered by the code
# before the P3 named-cube path was added (2026-10-10). Update only when single-task data changes on purpose.
SINGLE_TASK_SHA256 = "392992ff52fabdd1c6cebcd4edc91a3029d3a21d8c34dde12ffa522f2389ae5e"


def _single_task_section(text: str) -> str:
  out, on = [], False
  for line in text.splitlines():
    if line.startswith("## "):
      on = line.startswith("## Single-task")
    if on:
      out.append(line)
  return "\n".join(out) + "\n"


def test_single_task_table_unchanged():
  import hashlib
  man = json.loads((ROOT / "eval" / "scoreboard_manifest.json").read_text())
  first = man["tables"][0]["rows"][0]["runs"][0]
  if not (SB.resolve(first) / "episodes.jsonl").is_file():
    pytest.skip("real eval runs not on this machine")
  sec = _single_task_section(SB.render(man, "scoreboard_manifest.json"))
  assert "Holm across all 204 tests in this table" in sec
  assert hashlib.sha256(sec.encode()).hexdigest() == SINGLE_TASK_SHA256


# ----------------------------------------------------------------------------- P3 named cube

def _p3_run(tmp: Path, name: str, rows: list[tuple], multitask: str) -> str:
  """rows: (attempt, run_key, named colour, outcome, named_in_tray, wrong_in_tray)."""
  d = tmp / name
  d.mkdir()
  (d / "config.json").write_text(json.dumps({"mode": "policy", "ood": "none", "replan_every": 5,
                                             "multitask": multitask}))
  with open(d / "episodes.jsonl", "w") as f:
    for att, key, col, out, intray, wrong in rows:
      other = "green" if col == "red" else "red"
      f.write(json.dumps({
        "seed": 5000, "attempt": att, "run_key": key, "episode_seed": 5000 * 10**6 + att,
        # pass / L_task / L_full follow the planner's red cube: the P3 path must ignore them
        "pass": True, "L_task": True, "L_full": True, "L_task_named": intray, "L_full_named": None,
        "named_cube": {"named": col, "outcome": out, "named_in_tray": intray,
                       "in_tray": {col: intray, other: wrong}}}) + "\n")
  return str(d)


def _swap_rows(named_for: set) -> list[tuple]:
  """5 scene ids x 3 colours; outcome 'named' when (idx, colour) is in named_for, else 'none'."""
  return [(i, f"{i}:{c}", c, "named" if (i, c) in named_for else "none", (i, c) in named_for, False)
          for i in range(5) for c in ("red", "green", "blue")]


def test_swap_join_on_run_key(tmp_path):
  all_keys = {(i, c) for i in range(5) for c in ("red", "green", "blue")}
  a = _p3_run(tmp_path, "a_swap", _swap_rows(all_keys), "swap")  # A: named on all 15
  b = _p3_run(tmp_path, "b_swap", _swap_rows({(i, "red") for i in range(5)}), "swap")  # B: red only
  # Without the run_key join, (seed, attempt) repeats 3x and 15 runs collapse to 5.
  assert len(SB.load_run([a])["eps"]) == 5
  assert len(SB.load_run([a], "run_key")["eps"]) == 15
  row = dict(params="-", hardware="-", rung="A-swap", multitask="swap", join="run_key",
             paired_metrics=["named"], expected_n=15)
  man = {"tables": [{"title": "P3", "kind": "named_cube", "rows": [
    dict(row, model="A", test="swap", runs=[a]), dict(row, model="B", test="swap", runs=[b])]}]}
  text = SB.render(man, "m.json")
  # L_task column = named-in-tray, not the planner's L_task (which is True on every row)
  assert "| B | - | - | swap | - | 5/15 (" in text
  assert "5/0/10/0; named 5/15 (" in text
  # 15 joined runs: b = 10 (A named, B not), c = 0 -> p = 2/1024
  assert "| A [swap] | B [swap] | named cube moved (outcome = named) | (seed, run_key) | 15 | 15 | 5 | 10 | 0 |" in text
  assert "Holm across all 1 tests in this table" in text
  assert "A better (significant)" in text


def test_named_holm_is_its_own_family(tmp_path):
  # single-task table: 1 model pair x 2 metrics = 2 tests; must not leak into the P3 family
  a_none = _run(tmp_path, "a_none", 4000, "none", [1] * 30, [1] * 30)
  b_none = _run(tmp_path, "b_none", 4000, "none", [1] * 28 + [0] * 2, [1] * 30)
  st = {"title": "Single", "rows": [dict(model="A", test="none", rung="none", runs=[a_none]),
                                     dict(model="B", test="none", rung="none", runs=[b_none])]}
  # P3 held-out: 12 ids; A named + in tray on all, B on none -> b = 12, c = 0, p = 2/4096 = 0.00049
  ho_a = [(i, str(i), "red", "named", True, False) for i in range(12)]
  ho_b = [(i, str(i), "red", "other", False, True) for i in range(12)]
  pa = _p3_run(tmp_path, "pa", ho_a, "train")
  pb = _p3_run(tmp_path, "pb", ho_b, "train")
  row = dict(rung="A-held-out", multitask="train", paired_metrics=["named", "named_in_tray"])
  p3 = {"title": "P3", "kind": "named_cube", "rows": [
    dict(row, model="A", test="held-out", runs=[pa]), dict(row, model="B", test="held-out", runs=[pb]),
    dict(model="A", test="yellow", rung="A-yellow", runs=[str(tmp_path / "missing")])]}
  text = SB.render({"tables": [st, p3]}, "m.json")
  assert text.count("Holm across all 2 tests in this table") == 2  # one in each table, each m=2
  p3_text = text[text.index("## P3"):]
  assert "| A [held-out] | B [held-out] | named cube in tray | (seed, attempt) | 12 | 12 | 0 | 12 | 0 | 0.00049 | 0.00098 |" in p3_text
  assert "| B | - | - | held-out | - | 0/12 (" in p3_text  # L_task column ignores row L_task=True
  assert "| 12/12 (" in p3_text  # wrong-object for B: other cube in the tray every run
  assert "| A | - | - | yellow | - | - | - | - | - |" in p3_text  # missing run -> '-'
  assert "- none" in p3_text  # multitask matches config, no data problems
