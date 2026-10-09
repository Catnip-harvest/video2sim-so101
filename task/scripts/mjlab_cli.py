"""Run mjlab's stock train / play CLIs with SO101-PickPlace-State registered, without
pip-installing this package (handy on the laptop; on the MI300X `pip install -e
manip/task` registers the task through the mjlab.tasks entry point instead).

  python manip/task/scripts/mjlab_cli.py train SO101-PickPlace-State [mjlab train args]
      (runs tests/test_grasp_physics.py first and refuses to train if it fails;
       --skip-grasp-gate overrides)
  python manip/task/scripts/mjlab_cli.py play  SO101-PickPlace-State [mjlab play args]
  python manip/task/scripts/mjlab_cli.py list
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
  sys.path.insert(0, str(TASK_ROOT))


def grasp_gate() -> None:
  """Hard gate: the physics-honesty grasp test must pass before any training run."""
  import subprocess

  test = TASK_ROOT / "tests" / "test_grasp_physics.py"
  print(f"[so101_pickplace] grasp-physics gate: {test}")
  rc = subprocess.call([sys.executable, "-I", str(test)])
  if rc != 0:
    print("[so101_pickplace] grasp-physics gate FAILED; refusing to train "
          "(pass --skip-grasp-gate only if you know why).")
    sys.exit(rc)


def main() -> None:
  if len(sys.argv) < 2 or sys.argv[1] not in ("train", "play", "list"):
    print(__doc__)
    sys.exit(2)
  cmd = sys.argv.pop(1)
  skip_gate = "--skip-grasp-gate" in sys.argv
  if skip_gate:
    sys.argv.remove("--skip-grasp-gate")
  if cmd == "train" and not skip_gate:
    grasp_gate()
  import so101_pickplace.register  # noqa: F401  (registers the task)

  if cmd == "list":
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    print("\n".join(list_tasks()))
  elif cmd == "train":
    from mjlab.scripts.train import main as train_main

    train_main()
  else:
    from mjlab.scripts.play import main as play_main

    play_main()


if __name__ == "__main__":
  main()
