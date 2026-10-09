"""Registers SO101-PickPlace-State with mjlab (loaded by the ``mjlab.tasks`` entry
point, or imported by scripts/mjlab_cli.py when the package is not pip-installed).

The task spec comes from $SO101_TASK_SPEC if set, else specs/default_task_spec.json.
"""

from __future__ import annotations

from . import PLAY_TASK_ID, TASK_ID, platform_patches

platform_patches.apply()

from mjlab.tasks.registry import list_tasks, register_mjlab_task  # noqa: E402

from .env_cfg import so101_pickplace_env_cfg  # noqa: E402
from .rl_cfg import so101_pickplace_ppo_runner_cfg  # noqa: E402
from .runner import SO101PickPlaceRunner  # noqa: E402
from .task_spec import load_task_spec  # noqa: E402

if TASK_ID not in list_tasks():
  _spec = load_task_spec()
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=so101_pickplace_env_cfg(_spec),
    play_env_cfg=so101_pickplace_env_cfg(_spec, play=True),
    rl_cfg=so101_pickplace_ppo_runner_cfg(),
    runner_cls=SO101PickPlaceRunner,
  )
