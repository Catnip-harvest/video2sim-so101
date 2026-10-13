"""Stage 2: does the Menagerie `wrist_cam` (fovy 61) see the jaws where the real cam_right does?

cam_right is the WRIST camera (verified from frames, 13 Oct: both jaw tips are visible at the
bottom edge and the view moves with the gripper). In the wrist camera's own frame only the
gripper joint changes what is seen of the robot (the fixed jaw is rigid with the camera; the
moving jaw rotates with `gripper`). So the check is: for frames with many different gripper
openings, compare the real dark-jaw pixels in the bottom part of the image with the sim jaws.

Real mask: dark & not green (masks.dark_mask) in the bottom BAND fraction of the image.
Sim mask: robot surface points of the gripper bodies, projected from the wrist camera
(PointArm, for the optimiser); reported numbers use MuJoCo's renderer with the full meshes.

Fit (only if the default is clearly off): camera offset in the camera_mount body (dx, dy, dz in
metres), small rotation (rx, ry, rz in radians about the camera's own axes) and fovy; optionally
the gripper map (sim angle at lerobot gripper 0 and 100). Fit on TRAIN frames, report on HELD-OUT.

  python -m friendrig.wristcam frames --out OUT --workers 8
  python -m friendrig.wristcam fit    --out OUT --workers 8 [--gripper-map]
  python -m friendrig.wristcam eval   --out OUT --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from calib.so101_units import ArmCalib, lerobot_to_sim
from friendrig.masks import dark_mask, iou
from friendrig.simcam import SO101_XML, PointArm, _rx, _ry, _rz

BAND = 0.45          # bottom 45 % of the image
PER_EP = 4
GRIP_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount")
SIGMA = np.array([0.004, 0.004, 0.004, 0.05, 0.05, 0.05, 4.0])
SIGMA_G = np.array([8.0, 8.0])
_G: dict = {}


def band_mask(m: np.ndarray) -> np.ndarray:
  out = m.copy()
  out[: int(m.shape[0] * (1 - BAND))] = False
  return out


def _frames_job(ep):
  from friendrig.realdata import FriendData
  D = _G.setdefault("D", FriendData())
  st = D.states(ep)
  rng = np.random.default_rng(ep)
  g = st[:, 5]
  # spread over gripper values: sort frames by gripper and take quantiles
  order = np.argsort(g)
  ks = sorted(set(int(order[int(q)]) for q in np.linspace(0, len(order) - 1, PER_EP + 2)[1:-1]))
  ks = [k for k in ks if 0 <= k < len(st)]
  imgs = D.frames(ep, "cam_right", ks)
  rows = []
  for k, img in zip(ks, imgs):
    m = band_mask(dark_mask(img))
    rows.append((k, st[k], np.packbits(m)))
  return ep, rows


def cmd_frames(a):
  import multiprocessing as mp
  from friendrig.camfit import split_of
  from friendrig.realdata import FriendData
  out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
  sp = split_of()
  eps = FriendData().episodes
  with mp.get_context("fork").Pool(a.workers) as pool:
    res = pool.map(_frames_job, eps, chunksize=4)
  E, F, S, M, SP = [], [], [], [], []
  for ep, rows in res:
    for k, s, m in rows:
      E.append(ep); F.append(k); S.append(s); M.append(m); SP.append(sp[ep])
  np.savez_compressed(out / "wrist_frames.npz", episode=np.array(E), frame_index=np.array(F), state=np.stack(S),
                      mask=np.stack(M), split=np.array(SP))
  print(f"wrist frames: {len(E)} (train {SP.count('train')}, held_out {SP.count('held_out')}), "
        f"gripper range {np.min(np.stack(S)[:, 5]):.1f}..{np.max(np.stack(S)[:, 5]):.1f}", flush=True)


def load(out):
  z = dict(np.load(Path(out) / "wrist_frames.npz"))
  n = len(z["episode"])
  z["mask"] = np.unpackbits(z["mask"], axis=1)[:, :640 * 480].reshape(n, 480, 640).astype(bool)
  return z


class WristModel:
  """Gripper points + the wrist camera, with a parametric correction of the camera."""

  def __init__(self, n_points=80000):
    import mujoco
    self.pa = PointArm(n_points=n_points, bodies=GRIP_BODIES)
    m = self.pa.m
    self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    self.body = m.cam_bodyid[self.cam]
    self.pos0 = m.cam_pos[self.cam].copy()
    q = m.cam_quat[self.cam]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q)
    self.R0 = R.reshape(3, 3)
    self.fovy0 = float(m.cam_fovy[self.cam])

  def local_cam(self, c):
    """c = (dx, dy, dz, rx, ry, rz, fovy) -> camera pos/R in the camera_mount body frame."""
    c = np.asarray(c, float)
    return self.pos0 + c[:3], self.R0 @ _rx(c[3]) @ _ry(c[4]) @ _rz(c[5]), c[6]

  def mask(self, c, q, w=160, h=120):
    pts = self.pa.world_points(q)
    d = self.pa.d
    Rb = d.xmat[self.body].reshape(3, 3)
    pl, Rl, fovy = self.local_cam(c)
    pose = (d.xpos[self.body] + Rb @ pl, Rb @ Rl)
    return band_mask(self.pa.mask([0, 0, 0, 0, 0, 0, fovy], pts=pts, w=w, h=h, pose=pose, close_px=3))


def default_c(wm):
  return np.r_[np.zeros(6), wm.fovy0]


def _init():
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  _G["wm"] = WristModel()


def calib_from(gmap):
  if gmap is None:
    return ArmCalib()
  return ArmCalib(gripper_deg_at_0=float(gmap[0]), gripper_deg_at_100=float(gmap[1]))


def _score_job(args):
  import cv2
  from friendrig.camfit import soft_iou
  c, gmap, states, reals = args
  wm = _G["wm"]
  qs = lerobot_to_sim(states, calib_from(gmap))
  s = []
  for q, (r, k) in zip(qs, reals):
    s.append(soft_iou(wm.mask(c, q), r, k))
  return s


def lowres(m):
  import cv2
  r = cv2.resize(m.astype(np.float32), (160, 120), interpolation=cv2.INTER_AREA)
  keep = np.ones((120, 160), bool)
  keep[: int(120 * (1 - BAND))] = False
  return cv2.GaussianBlur(r, (0, 0), 0.8), keep


def cmd_fit(a):
  import cma
  import multiprocessing as mp
  from friendrig.camfit import trimmed
  out = Path(a.out)
  z = load(out)
  rng = np.random.default_rng(a.seed)
  tr = np.where(z["split"] == "train")[0]
  sub = rng.choice(tr, min(a.n_frames, len(tr)), replace=False)
  reals = [lowres(z["mask"][i]) for i in sub]
  chunks = np.array_split(np.arange(len(sub)), a.workers)
  t0 = time.time()
  with mp.get_context("fork").Pool(a.workers, initializer=_init) as pool:
    wm = WristModel(n_points=1000)
    c0 = default_c(wm)
    g0 = np.array([ArmCalib().gripper_deg_at_0, ArmCalib().gripper_deg_at_100])

    def score(x):
      c = c0 + x[:7] * SIGMA
      gm = g0 + x[7:] * SIGMA_G if a.gripper_map else None
      if not (30 <= c[6] <= 110):
        return 0.0
      r = pool.map(_score_job, [(c, gm, z["state"][sub[ch]], [reals[i] for i in ch]) for ch in chunks])
      return trimmed(np.concatenate(r))

    nx = 9 if a.gripper_map else 7
    s_default = score(np.zeros(nx))
    print(f"default wrist_cam: train soft IoU (trimmed) {s_default:.3f}", flush=True)
    es = cma.CMAEvolutionStrategy(np.zeros(nx), 1.0, {"popsize": 12, "seed": a.seed + 1, "verbose": -9,
                                                       "maxiter": a.iters})
    best = (np.zeros(nx), s_default)
    while not es.stop():
      xs = es.ask()
      fs = [-score(x) for x in xs]
      es.tell(xs, fs)
      i = int(np.argmin(fs))
      if -fs[i] > best[1]:
        best = (np.asarray(xs[i]), -fs[i])
    x = best[0]
  c = c0 + x[:7] * SIGMA
  gm = (g0 + x[7:] * SIGMA_G).tolist() if a.gripper_map else None
  rec = {"default": c0.tolist(), "fitted": c.tolist(), "names": ["dx_m", "dy_m", "dz_m", "rx_rad", "ry_rad", "rz_rad", "fovy_deg"],
         "gripper_map_deg_at_0_100": gm, "train_soft_default": s_default, "train_soft_fitted": best[1],
         "n_frames": int(len(sub))}
  tag = "_gmap" if a.gripper_map else ""
  (out / f"wrist_fit{tag}.json").write_text(json.dumps(rec, indent=1))
  print(f"fitted wrist_cam {np.round(c, 4).tolist()} gripper map {gm}: train soft IoU {best[1]:.3f} "
        f"(default {s_default:.3f}) ({time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------------------------- eval (OSMesa)

def _eval_init():
  import mujoco
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  spec = mujoco.MjSpec.from_file(str(SO101_XML))
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
  m = spec.compile()
  _G["m"], _G["d"] = m, mujoco.MjData(m)
  _G["r"] = mujoco.Renderer(m, 480, 640)
  _G["r"].enable_segmentation_rendering()
  _G["wm"] = WristModel(n_points=1000)


def _eval_job(args):
  import mujoco
  i, state, real, c, gmap = args
  m, d, r, wm = _G["m"], _G["d"], _G["r"], _G["wm"]
  cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
  pl, Rl, fovy = wm.local_cam(c)
  q = np.zeros(4); mujoco.mju_mat2Quat(q, Rl.flatten())
  m.cam_pos[cam], m.cam_quat[cam], m.cam_fovy[cam] = pl, q, fovy
  from friendrig.simcam import JOINTS
  qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
  d.qpos[qadr] = lerobot_to_sim(state, calib_from(gmap))
  mujoco.mj_forward(m, d)
  r.update_scene(d, camera=cam)
  seg = r.render()
  sim = band_mask(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
  return i, iou(sim, real), np.packbits(sim)


def cmd_eval(a):
  import multiprocessing as mp
  out = Path(a.out)
  z = load(out)
  idx = np.where(z["split"] == "held_out")[0]
  wm = WristModel(n_points=1000)
  variants = {"default": (default_c(wm), None)}
  for tag in ("", "_gmap"):
    p = out / f"wrist_fit{tag}.json"
    if p.exists():
      f = json.loads(p.read_text())
      variants["fitted" + tag] = (np.array(f["fitted"]), f["gripper_map_deg_at_0_100"])
  rec = {}
  with mp.get_context("fork").Pool(a.workers, initializer=_eval_init) as pool:
    for name, (c, gm) in variants.items():
      res = pool.map(_eval_job, [(i, z["state"][i], z["mask"][i], c, gm) for i in idx], chunksize=4)
      v = np.array([u for _, u, _ in res])
      rec[name] = {"camera": np.asarray(c).tolist(), "gripper_map": gm, "n_frames": int(len(v)),
                   "mean_iou": float(v.mean()), "median_iou": float(np.median(v))}
      print(f"held-out wrist band IoU [{name}]: mean {v.mean():.3f} median {np.median(v):.3f} n={len(v)}", flush=True)
      if a.overlays:
        _overlays(out, z, res, name, a.overlays)
  (out / "wrist_eval.json").write_text(json.dumps(rec, indent=1))


def _overlays(out, z, res, name, n):
  import cv2
  from friendrig.realdata import FriendData
  D = FriendData()
  od = out / f"wrist_overlays_{name}"; od.mkdir(exist_ok=True)
  for j in np.linspace(0, len(res) - 1, n).round().astype(int):
    i, u, pm = res[j]
    ep, k = int(z["episode"][i]), int(z["frame_index"][i])
    img = D.frames(ep, "cam_right", [k])[0].copy()
    sim = np.unpackbits(pm)[:640 * 480].reshape(480, 640)
    for m, col in ((z["mask"][i], (0, 255, 0)), (sim, (255, 0, 255))):
      cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
      cv2.drawContours(img, cs, -1, col, 2)
    cv2.putText(img, f"{name} ep{ep} f{k} grip {z['state'][i][5]:.0f} IoU {u:.2f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imwrite(str(od / f"ep{ep:03d}_f{k:04d}.jpg"), img[..., ::-1])


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  for name in ("frames", "fit", "eval"):
    p = sub.add_parser(name)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
  sub.choices["fit"].add_argument("--n-frames", type=int, default=400)
  sub.choices["fit"].add_argument("--iters", type=int, default=60)
  sub.choices["fit"].add_argument("--gripper-map", action="store_true")
  sub.choices["eval"].add_argument("--overlays", type=int, default=8)
  a = ap.parse_args()
  {"frames": cmd_frames, "fit": cmd_fit, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
  main()
