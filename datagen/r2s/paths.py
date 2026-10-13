"""Make manip/task importable (read-only reuse of the SO-101 model, IK and spec loader)."""

from __future__ import annotations

import sys
from pathlib import Path

DATAGEN_ROOT = Path(__file__).resolve().parents[1]  # manip/datagen
MANIP_ROOT = DATAGEN_ROOT.parent  # manip
TASK_ROOT = MANIP_ROOT / "task"
DEMO_ROOT = MANIP_ROOT / "demo"

if str(TASK_ROOT) not in sys.path:
  sys.path.insert(0, str(TASK_ROOT))
if str(MANIP_ROOT) not in sys.path:  # repo root: friendrig (scene profiles)
  sys.path.append(str(MANIP_ROOT))

from so101_pickplace import ik, mj_scene  # noqa: E402,F401
from so101_pickplace import robot_model as rm  # noqa: E402,F401
from so101_pickplace.task_spec import TaskSpec, load_task_spec  # noqa: E402,F401
