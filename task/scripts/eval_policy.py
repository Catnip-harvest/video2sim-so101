"""Headless evaluation of a trained SO101-PickPlace-State checkpoint (no viewer needed;
for the MI300X box where `mjlab.scripts.play` has no display).

  python manip/task/scripts/eval_policy.py --checkpoint logs/rsl_rl/so101_pickplace_state/<run>/model_4000.pt \
      --num-envs 256 --device cuda:0 [--spec path/to/spec.json] [--train-cfg-dr]

Runs one full episode per env with the play config (no observation noise / dropouts)
unless --train-cfg-dr is given (training config: full DR and noise), and reports per-
episode rates: ever grasped, ever lifted, placed, released at the end, mean distance to
the object reference. Writes <checkpoint dir>/eval_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
  sys.path.insert(0, str(TASK_ROOT))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--spec", default=None)
  ap.add_argument("--train-cfg-dr", action="store_true",
                  help="evaluate under the training config (DR + noise + dropouts)")
  args = ap.parse_args()
  if args.spec:
    os.environ["SO101_TASK_SPEC"] = args.spec

  import torch

  import so101_pickplace.register  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from so101_pickplace import TASK_ID
  from so101_pickplace.env_cfg import COMMAND
  from so101_pickplace.mdp import state
  from so101_pickplace.runner import SO101PickPlaceRunner

  cfg = load_env_cfg(TASK_ID, play=not args.train_cfg_dr)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  agent = load_rl_cfg(TASK_ID)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = SO101PickPlaceRunner(wrapped, asdict(agent), None, args.device)
  runner.load(args.checkpoint, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  obs = wrapped.get_observations()
  env.reset()
  obs = wrapped.get_observations()
  n = env.num_envs
  T = env.max_episode_length
  ever_grasp = torch.zeros(n, dtype=torch.bool, device=env.device)
  ever_lift = torch.zeros_like(ever_grasp)
  ever_placed = torch.zeros_like(ever_grasp)
  ref_err = torch.zeros(n, device=env.device)
  last = None
  with torch.inference_mode():
    for _ in range(T - 1):
      obs, _, _, _ = wrapped.step(policy(obs))
      s = state.compute(env, COMMAND, phase="obs")
      ever_grasp |= s.grasped
      ever_lift |= s.lifted
      ever_placed |= s.placed
      ref_err += s.ref_error
      last = s
  assert last is not None
  res = {
    "checkpoint": args.checkpoint,
    "num_episodes": n,
    "config": "train (DR on)" if args.train_cfg_dr else "play (noise off)",
    "ever_grasped": float(ever_grasp.float().mean()),
    "ever_lifted": float(ever_lift.float().mean()),
    "ever_placed": float(ever_placed.float().mean()),
    "released_in_bin_at_end": float(last.released.float().mean()),
    "mean_object_reference_error_m": float((ref_err / (T - 1)).mean()),
  }
  print(json.dumps(res, indent=1))
  out = Path(args.checkpoint).with_name(
    f"eval_{Path(args.checkpoint).stem}_{'dr' if args.train_cfg_dr else 'play'}.json")
  out.write_text(json.dumps(res, indent=1))
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
