"""Task-spec parser checks (pure Python + numpy). The demo-video worker can run this
against its own output:  python -m pytest -q manip/task/tests/test_task_spec.py
or validate one file:     python -I manip/task/tests/test_task_spec.py path/to/spec.json
"""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
  sys.path.insert(0, str(TASK_ROOT))

from so101_pickplace import task_spec as ts  # noqa: E402

SPECS = [TASK_ROOT / "specs" / "default_task_spec.json",
         TASK_ROOT / "specs" / "example_from_video_task_spec.json"]


def _load(p: Path) -> dict:
  return json.loads(p.read_text(encoding="utf-8"))


def _raises(d: dict, needle: str) -> None:
  try:
    ts.parse_task_spec(d)
  except ValueError as e:
    assert needle in str(e), str(e)
    return
  raise AssertionError(f"expected ValueError containing {needle!r}")


def test_shipped_specs_parse_and_reference_is_sane():
  for p in SPECS:
    s = ts.load_task_spec(p)
    t, pos, g, r = s.trajectory.reference()
    assert t[0] == 0.0 and np.all(np.diff(t) > 0)
    assert pos.shape == (len(t), 3) and np.isfinite(pos).all()
    assert 0.0 <= g <= r <= t[-1] + 1e-9
    assert pos[:, 2].min() >= -1e-6


def test_rejects_out_of_reach_and_bad_times():
  base = _load(SPECS[0])
  d = copy.deepcopy(base)
  d["object"]["start_pose_range"]["x"] = [0.30, 0.40]
  _raises(d, "outside the measured workspace")
  d = copy.deepcopy(base)
  d["object_trajectory"]["grasp_time"] = 9.9
  d["object_trajectory"]["release_time"] = 1.0
  _raises(d, "grasp_time")
  d = copy.deepcopy(base)
  d["object"]["size"] = 0.08
  _raises(d, "object.size")
  d = copy.deepcopy(base)
  d["format"] = "something_else"
  _raises(d, "format")


def test_hand_path_is_ignored_with_warning():
  d = _load(SPECS[0])
  d["hand_path"] = {"weight": 1.0, "waypoints": [{"anchor": "object", "offset": [0, 0, 0.1]}]}
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ts.parse_task_spec(d)
  assert any("hand_path" in str(x.message) for x in w)


if __name__ == "__main__":
  if len(sys.argv) > 1:
    s = ts.load_task_spec(sys.argv[1])
    t, pos, g, r = s.trajectory.reference() if s.trajectory else (None, None, None, None)
    print(f"OK: {sys.argv[1]}: object {s.object.kind} {s.object.size * 1e3:.0f} mm, "
          f"reference {0 if t is None else t[-1]:.1f} s robot time, grasp {g}, release {r}")
  else:
    test_shipped_specs_parse_and_reference_is_sane()
    test_rejects_out_of_reach_and_bad_times()
    test_hand_path_is_ignored_with_warning()
    print("PASS task spec tests")
