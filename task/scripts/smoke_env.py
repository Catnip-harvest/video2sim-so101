"""Smoke test for SO101-PickPlace-State (laptop: tiny env count, short, no training run).

1. Builds the env and prints observation / action layout.
2. Steps random actions; asserts observations and rewards are finite.
3. Scripted pick-and-place THROUGH THE ENV (MuJoCo Warp physics, the action term with
   its speed cap, the contact sensors): IK goals are solved on the CPU model and turned
   into capped target-delta actions. Reports whether the jaw sensors, lifted, placed and
   released flags light up. No state is written; the object moves by contact only.
4. Builds the PPO runner, runs a couple of tiny iterations, saves a checkpoint, and
   checks the exported ONNX against the torch policy with onnxruntime.

  python -I manip/task/scripts/smoke_env.py --device cpu --num-envs 2
  python -I manip/task/scripts/smoke_env.py --device cuda:0 --num-envs 4   (if CPU warp is too slow)

Writes measurements/smoke_<device>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
  sys.path.insert(0, str(TASK_ROOT))


def _finite(x) -> bool:
  import torch

  if isinstance(x, dict):
    return all(_finite(v) for v in x.values())
  return bool(torch.isfinite(x).all())


def scripted_pick_place(env, report: dict, max_steps: int = 500) -> None:
  import mujoco
  import numpy as np
  import torch

  from so101_pickplace import ik
  from so101_pickplace import robot_model as rm
  from so101_pickplace.env_cfg import COMMAND
  from so101_pickplace.mdp import state

  u = env.unwrapped if hasattr(env, "unwrapped") else env
  # The scripted controller is slower than a policy (it finishes each waypoint before
  # starting the next), so the 12 s time-out is lifted for this section only; the
  # step at which it would have fired is reported. Any other termination is counted.
  saved_len = u.cfg.episode_length_s
  u.cfg.episode_length_s = 1.0e5
  u.reset()
  act = u.action_manager.get_term("joint_pos")
  act.latency[:] = False  # scripted controller: no latency
  cmd = u.command_manager.get_term(COMMAND)
  mj_model = u.sim.mj_model
  n = u.num_envs
  dev = u.device
  names = list(act.target_names)
  arm_idx = [names.index(j) for j in rm.ARM_JOINTS]
  g_idx = names.index(rm.GRIPPER_JOINT)

  # IK in the base frame: the CPU model's robot base sits at (0, 0, mount).
  base = cmd.base_pos_w.cpu().numpy()
  origin = u.scene.env_origins.cpu().numpy()
  obj = cmd.object_start_w.cpu().numpy() - base
  tgt = cmd.target_pos_w.cpu().numpy() - base
  obj_quat = cmd.object.data.root_link_quat_w.cpu().numpy()
  half_h = cmd.object_half_height.cpu().numpy()
  half_w = cmd.object_half_width.cpu().numpy()
  floor = cmd.cfg.bin_floor_top
  mount = cmd.cfg.mount_height

  plans = []
  q_hint = act.target[:, arm_idx].cpu().numpy()
  for e in range(n):
    radial = obj[e, :2] - np.array([0.0388, 0.0])
    radial /= np.linalg.norm(radial)
    c = np.array([-radial[1], radial[0], 0.0])  # jaws close tangentially ...
    if cmd.cfg.object_kind == "box":  # ... snapped to the nearest cube face normal
      qw, qz = obj_quat[e, 0], obj_quat[e, 3]
      yaw = 2.0 * np.arctan2(qz, qw)
      normals = [np.array([np.cos(yaw + j * np.pi / 2), np.sin(yaw + j * np.pi / 2), 0.0])
                 for j in range(4)]
      c = max(normals, key=lambda v: float(v @ c))
    gz = max(half_h[e], rm.JAW_TIP_BELOW_SITE + 0.003) + mount
    off = c * (half_w[e] + 0.004 - rm.NOMINAL_OBJECT_WIDTH / 2)
    o = np.array([obj[e, 0], obj[e, 1], 0.0]) + off
    t = np.array([tgt[e, 0], tgt[e, 1], 0.0]) + off
    carry_z = mount + floor + half_h[e] + 0.075
    steps = [  # (site target, gripper target, extra hold steps)
      (o + [0, 0, gz + 0.06], 0.75, 2),
      (o + [0, 0, gz + 0.02], 0.75, 1),
      (o + [0, 0, gz], 0.75, 3),
      (o + [0, 0, gz], rm.GRIPPER_CLOSED, 8),
      (o + [0, 0, gz + 0.08], rm.GRIPPER_CLOSED, 2),
      (t + [0, 0, carry_z], rm.GRIPPER_CLOSED, 2),
      (t + [0, 0, mount + floor + half_h[e] + 0.015], rm.GRIPPER_CLOSED, 3),
      (t + [0, 0, mount + floor + half_h[e] + 0.015], rm.GRIPPER_OPEN, 6),
      (t + [0, 0, carry_z + 0.03], rm.GRIPPER_OPEN, 4),
    ]
    goals = []
    hint = q_hint[e]
    for pos, grip, hold in steps:
      r = ik.multi_start(mj_model, np.asarray(pos, float), closing=c, q_hint=hint,
                         seeds=4, prefix="robot/")
      hint = r.q
      goals.append((r.q, grip, hold, r.pos_err))
    plans.append(goals)
  report["scripted_ik_max_err_mm"] = round(
    1e3 * max(g[3] for p in plans for g in p), 2)

  stage = [0] * n
  hold = [0] * n
  log = {k: np.zeros((max_steps, n)) for k in
         ("bottom", "grasped", "lifted", "placed", "released", "fixed", "moving")}
  rew = np.zeros((max_steps, n))
  max_delta = act.max_delta.cpu().numpy()
  # Goals are clipped to the action term's soft joint limits (the term clamps targets
  # there), otherwise e.g. a fully-closed jaw goal (-0.17) is never "reached".
  lo = act._lo.expand(n, -1).cpu().numpy()
  hi = act._hi.expand(n, -1).cpu().numpy()
  settle = 0
  resets = np.zeros(n, dtype=int)
  stage_steps: list[list[int]] = [[] for _ in range(n)]
  for k in range(max_steps):
    a = np.zeros((n, act.action_dim), dtype=np.float32)
    cur = act.target.cpu().numpy()
    for e in range(n):
      if stage[e] >= len(plans[e]):
        continue
      q_goal, grip, hold_n, _ = plans[e][stage[e]]
      goal = cur[e].copy()
      goal[arm_idx] = q_goal
      goal[g_idx] = grip
      goal = np.clip(goal, lo[e], hi[e])
      a[e] = np.clip((goal - cur[e]) / max_delta, -1.0, 1.0)
      if np.max(np.abs(goal - cur[e])) < 1e-4:
        hold[e] += 1
        if hold[e] >= hold_n:
          stage[e] += 1
          hold[e] = 0
          stage_steps[e].append(k)
    _, r, term, trunc, _ = u.step(torch.as_tensor(a, device=dev))
    resets += (term | trunc).cpu().numpy().astype(int)
    s = state.compute(u, COMMAND, phase="obs")
    log["bottom"][k] = s.object_bottom.cpu().numpy()
    for key, v in (("grasped", s.grasped), ("lifted", s.lifted), ("placed", s.placed),
                   ("released", s.released), ("fixed", s.fixed_contact),
                   ("moving", s.moving_contact)):
      log[key][k] = v.float().cpu().numpy()
    rew[k] = r.cpu().numpy()
    if all(st >= len(p) for st, p in zip(stage, plans)):
      settle += 1
      if settle >= 10:  # let the dropped object come to rest before judging
        log = {kk: vv[: k + 1] for kk, vv in log.items()}
        rew = rew[: k + 1]
        break
  u.cfg.episode_length_s = saved_len
  report["scripted"] = {
    "steps": int(len(rew)),
    "episode_length_steps_in_training": int(math.ceil(saved_len / u.step_dt)),
    "resets_during_script": [int(v) for v in resets],
    "stage_names": ["pre-grasp", "descend", "at-grasp", "close", "lift", "carry",
                    "lower", "open", "retreat"],
    "step_each_stage_completed": stage_steps,
    "stages_completed": [int(s) for s in stage],
    "stages_total": len(plans[0]),
    "object_max_bottom_height_m": [round(float(v), 4) for v in log["bottom"].max(0)],
    "steps_grasped": [int(v) for v in log["grasped"].sum(0)],
    "steps_lifted": [int(v) for v in log["lifted"].sum(0)],
    "steps_fixed_jaw_contact": [int(v) for v in log["fixed"].sum(0)],
    "steps_moving_jaw_contact": [int(v) for v in log["moving"].sum(0)],
    "ever_placed": [bool(v) for v in log["placed"].max(0)],
    "released_at_end": [bool(v) for v in log["released"][-1]],
    "episode_return": [round(float(v), 3) for v in rew.sum(0)],
    "rewards_finite": bool(np.isfinite(rew).all()),
  }
  s_end = state.compute(u, COMMAND, phase="obs")
  rel = (s_end.obj_pos_w - cmd.target_pos_w).cpu().numpy()
  report["scripted"]["end_object_minus_bin_centre_m"] = rel.round(4).tolist()
  report["scripted"]["end_object_speed_mps"] = (
    s_end.obj_linvel_w.norm(dim=-1).cpu().numpy().round(4).tolist())
  report["scripted"]["end_jaw_contacts"] = [
    [bool(a), bool(b)] for a, b in zip(s_end.fixed_contact.cpu(), s_end.moving_contact.cpu())]
  # The startup / reset DR draws of each env, to explain differences between envs.
  m = u.sim.model
  robot = u.scene["robot"]
  gid = int(cmd.object.indexing.geom_ids[0])
  finger_gids = [int(g) for g, nme in zip(robot.indexing.geom_ids.cpu(), robot.geom_names)
                 if nme.startswith(("fixed_jaw", "moving_jaw"))]
  g_ctrl = [int(i) for a in robot.actuators if "gripper" in a.target_names
            for i in a.global_ctrl_ids.cpu()]
  def per_env(field, idx):
    t = getattr(m, field)
    return [t[min(e, t.shape[0] - 1)][idx] for e in range(n)]
  report["scripted"]["dr_draws"] = {
    "object_size_m": [v.cpu().numpy().round(4).tolist() for v in per_env("geom_size", gid)],
    "object_mass_kg": [round(float(v), 4) for v in per_env("body_mass", int(cmd.object.indexing.body_ids[0]))],
    "finger_slide_friction_mean": [round(float(v[:, 0].mean()), 3) for v in per_env("geom_friction", finger_gids)],
    "gripper_kp": [round(float(v[0, 0]), 1) for v in per_env("actuator_gainprm", g_ctrl)],
    "gripper_force_limit_nm": [round(float(v[0, 1]), 3) for v in per_env("actuator_forcerange", g_ctrl)],
    "action_latency": [bool(v) for v in act.latency.cpu()],
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--num-envs", type=int, default=2)
  ap.add_argument("--random-steps", type=int, default=20)
  ap.add_argument("--train-iters", type=int, default=2)
  ap.add_argument("--skip-scripted", action="store_true")
  ap.add_argument("--skip-runner", action="store_true")
  ap.add_argument("--spec", default=None, help="task spec JSON (default: SO101_TASK_SPEC "
                  "or specs/default_task_spec.json)")
  args = ap.parse_args()
  if args.spec:
    os.environ["SO101_TASK_SPEC"] = args.spec
  if args.device == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

  report: dict = {"device": args.device, "num_envs": args.num_envs}
  t0 = time.time()
  import torch

  import so101_pickplace.register  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from so101_pickplace import TASK_ID
  from so101_pickplace.runner import SO101PickPlaceRunner, deploy_contract

  report["import_s"] = round(time.time() - t0, 1)
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = args.num_envs
  t1 = time.time()
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  report["env_build_s"] = round(time.time() - t1, 1)
  u = env
  report["step_dt"] = u.step_dt
  report["decimation"] = cfg.decimation
  report["episode_length_steps"] = u.max_episode_length
  report["obs_terms"] = {g: list(t) for g, t in u.observation_manager.active_terms.items()}
  report["obs_dims"] = {g: [list(d) for d in v]
                        for g, v in u.observation_manager.group_obs_term_dim.items()}
  report["action_dim"] = u.action_manager.total_action_dim
  report["reward_terms"] = list(u.reward_manager.active_terms)
  print(json.dumps({k: report[k] for k in ("import_s", "env_build_s", "action_dim",
                                           "obs_dims")}, indent=1))

  obs, _ = u.reset()
  assert _finite(obs), "non-finite observation after reset"
  t2 = time.time()
  ok = True
  for _ in range(args.random_steps):
    a = 0.7 * torch.randn(u.num_envs, report["action_dim"], device=u.device)
    obs, r, term, trunc, extras = u.step(a)
    ok &= _finite(obs) and _finite(r)
  dt = time.time() - t2
  report["random_steps"] = args.random_steps
  report["random_steps_finite"] = bool(ok)
  report["sim_steps_per_s"] = round(args.random_steps * cfg.decimation * u.num_envs / dt, 1)
  print(f"[smoke] random steps finite={ok}  {report['sim_steps_per_s']} physics steps/s")
  assert ok, "non-finite obs/reward under random actions"

  if not args.skip_scripted:
    t3 = time.time()
    scripted_pick_place(u, report)
    report["scripted_s"] = round(time.time() - t3, 1)
    print("[smoke] scripted:", json.dumps(report["scripted"]))

  if not args.skip_runner:
    agent = load_rl_cfg(TASK_ID)
    agent.num_steps_per_env = 8
    agent.max_iterations = args.train_iters
    agent.save_interval = 1
    agent.logger = "tensorboard"
    log_dir = TASK_ROOT / "logs" / f"smoke_{args.device.replace(':', '')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    wrapped = RslRlVecEnvWrapper(u, clip_actions=agent.clip_actions)
    runner = SO101PickPlaceRunner(wrapped, asdict(agent), str(log_dir), args.device)
    t4 = time.time()
    runner.learn(num_learning_iterations=args.train_iters, init_at_random_ep_len=True)
    report["train_iters"] = args.train_iters
    report["train_s"] = round(time.time() - t4, 1)
    ckpts = sorted(log_dir.glob("model_*.pt"))
    onnx_files = sorted(log_dir.glob("*.onnx"))
    report["checkpoints"] = [p.name for p in ckpts]
    report["onnx_files"] = [p.name for p in onnx_files]
    report["deploy_contract_written"] = (log_dir / "deploy_contract.json").exists()
    if onnx_files:
      import numpy as np
      import onnxruntime as ort

      sess = ort.InferenceSession(str(onnx_files[-1]), providers=["CPUExecutionProvider"])
      inp = sess.get_inputs()
      obs = wrapped.get_observations()
      policy = runner.alg.get_policy()
      policy.eval()
      with torch.no_grad():
        ref = policy(obs)  # deterministic (mean) output by default in rsl-rl 5
      ref = ref.detach().cpu().numpy() if hasattr(ref, "detach") else np.asarray(ref)
      actor_obs = obs["actor"].detach().cpu().numpy().astype(np.float32)
      out = sess.run(None, {inp[0].name: actor_obs[:1]})[0]
      report["onnx_inputs"] = [(i.name, list(i.shape)) for i in inp]
      report["onnx_output_shape"] = list(out.shape)
      report["onnx_vs_torch_max_abs_diff"] = float(np.abs(out - ref[:1]).max())
    report["deploy_contract"] = deploy_contract(wrapped)
    print("[smoke] runner:", {k: report.get(k) for k in (
      "train_s", "checkpoints", "onnx_files", "onnx_vs_torch_max_abs_diff")})

  out = TASK_ROOT / "measurements" / f"smoke_{args.device.replace(':', '')}.json"
  out.write_text(json.dumps(report, indent=1, default=str))
  print(f"[smoke] wrote {out}  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
  main()
