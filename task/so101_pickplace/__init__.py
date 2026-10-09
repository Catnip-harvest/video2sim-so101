"""SO-101 pick-and-place task for mjlab.

This package ``__init__`` is deliberately light (no mjlab / torch / warp import) so the
pure-MuJoCo pieces (``robot_model``, ``task_spec``) can be used by the CPU grasp test
and by the demo-video worker without loading the RL stack.

Task registration lives in :mod:`so101_pickplace.register`, which is what mjlab's
``mjlab.tasks`` entry point loads (see pyproject.toml).
"""

TASK_ID = "SO101-PickPlace-State"
PLAY_TASK_ID = TASK_ID  # play uses the registered play_env_cfg of the same id
