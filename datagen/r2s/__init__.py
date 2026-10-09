"""Real-to-sim data engine: one human demo (task spec) -> N physically verified SO-101
pick-and-place demonstrations, written as a LeRobot v3 dataset.

Modules
  paths       import paths into manip/task (SO-101 model, IK, spec loader); never edits them
  variations  per-episode random draws (object, bin, servos, look, cameras), seeded
  scene       builds the MuJoCo model for one episode (robot + table + cube + bin + cameras)
  planner     scripted expert: IK waypoints and speed-capped joint-target trajectories
  rollout     full-physics simulation of one episode, contact bookkeeping and the verifier
  render      offline re-rendering of an accepted episode's recorded states to video
  lerobot_io  LeRobot v3 dataset writer (pyarrow if importable, else DuckDB)
"""
