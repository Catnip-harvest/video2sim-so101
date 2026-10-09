"""Small damped-least-squares IK for the SO-101 grasp site (pure MuJoCo, CPU).

Only used by tests and tools (grasp-physics test, home-pose derivation). The RL
policy never sees it. It solves on a scratch MjData so the simulated state is never
touched: the result is a vector of joint *targets* that the position servos then have
to reach through the dynamics.

The arm has 5 joints, so a full 6-D pose is generally unreachable. The target used
here is position (3) + approach axis (2) + closing-axis yaw (1); for a near-vertical
approach the last one is realised by ``wrist_roll`` almost independently, so the
target is consistent and the residual is reported, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import robot_model as rm


@dataclass
class IkResult:
  q: np.ndarray  # arm joint targets (5,), order rm.ARM_JOINTS
  pos_err: float  # metres
  approach_err_deg: float
  closing_err_deg: float
  converged: bool


def gripper_rotation(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
  """World rotation of the ``gripper`` body for a given finger (approach) direction and
  jaw-closing direction. Fingers point along body -z, the moving jaw closes along -x."""
  z = -approach / np.linalg.norm(approach)
  x = -closing - np.dot(-closing, z) * z
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  return np.stack([x, y, z], axis=1)


def solve(
  model: mujoco.MjModel,
  q_seed: np.ndarray,
  target_pos: np.ndarray,
  approach: np.ndarray = np.array([0.0, 0.0, -1.0]),
  closing: np.ndarray | None = None,
  site: str = "grasp_site",
  iters: int = 300,
  damping: float = 1e-3,
  w_rot: float = 0.05,
  tol_pos: float = 5e-4,
  qpos_full: np.ndarray | None = None,
  prefix: str = "",
) -> IkResult:
  data = mujoco.MjData(model)
  if qpos_full is not None:
    data.qpos[:] = qpos_full
  jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
          for j in rm.ARM_JOINTS]
  qadr = np.array([model.jnt_qposadr[j] for j in jids])
  dadr = np.array([model.jnt_dofadr[j] for j in jids])
  lo = model.jnt_range[jids, 0]
  hi = model.jnt_range[jids, 1]
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + site)
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "gripper")
  q = np.clip(np.asarray(q_seed, float).copy(), lo, hi)
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))
  a_des = approach / np.linalg.norm(approach)
  for _ in range(iters):
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    R = data.xmat[bid].reshape(3, 3)
    a = -R[:, 2]
    e_pos = target_pos - data.site_xpos[sid]
    e_rot = np.cross(a, a_des)  # rotate a onto a_des
    if closing is not None:
      c = -R[:, 0]
      c_des = closing - np.dot(closing, a) * a
      c_des /= np.linalg.norm(c_des)
      e_rot = e_rot + np.dot(np.cross(c, c_des), a) * a
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    J = np.vstack([jacp[:, dadr], w_rot * jacr[:, dadr]])
    e = np.concatenate([e_pos, w_rot * e_rot])
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
    q = np.clip(q + dq, lo, hi)
    if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_rot) < 1e-3:
      break
  data.qpos[qadr] = q
  mujoco.mj_kinematics(model, data)
  R = data.xmat[bid].reshape(3, 3)
  a = -R[:, 2]
  pos_err = float(np.linalg.norm(target_pos - data.site_xpos[sid]))
  app_err = float(np.degrees(np.arccos(np.clip(np.dot(a, a_des), -1, 1))))
  clo_err = 0.0
  if closing is not None:
    c = -R[:, 0]
    c_des = closing - np.dot(closing, a) * a
    c_des /= np.linalg.norm(c_des)
    clo_err = float(np.degrees(np.arccos(np.clip(abs(np.dot(c, c_des)), -1, 1))))
  return IkResult(q=q, pos_err=pos_err, approach_err_deg=app_err, closing_err_deg=clo_err,
                  converged=pos_err < 2e-3 and app_err < 3.0)


def multi_start(model, target_pos, approach=np.array([0.0, 0.0, -1.0]), closing=None,
                seeds: int = 12, rng_seed: int = 0, q_hint: np.ndarray | None = None,
                **kw) -> IkResult:
  rng = np.random.default_rng(rng_seed)
  prefix = kw.get("prefix", "")
  jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
          for j in rm.ARM_JOINTS]
  lo = model.jnt_range[jids, 0]
  hi = model.jnt_range[jids, 1]
  best = None
  starts = [q_hint] if q_hint is not None else []
  starts += [rng.uniform(lo * 0.6, hi * 0.6) for _ in range(seeds)]
  for q0 in starts:
    r = solve(model, q0, target_pos, approach, closing, **kw)
    score = r.pos_err + 0.002 * r.approach_err_deg + 0.0005 * r.closing_err_deg
    if best is None or score < best[0]:
      best = (score, r)
  return best[1]
