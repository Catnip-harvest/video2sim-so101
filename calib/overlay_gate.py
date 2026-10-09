"""G0 overlay gate: does the friend's joint data, converted with calib/so101_units.py, put the
simulated SO-101 where the real arm is in the overhead camera (cam_head)?

Method (one paragraph per step, so it can be explained out loud):

1. Frames (`frames` sub-command). For a few hand-picked table-to-bowl episodes (picked only for
   "arm clearly visible", before any IoU was computed) we decode every STEP-th cam_head frame.
   A frame is a candidate if the real arm mask covers >= 5 % of the image, the arm is > 20 deg
   away from its start/end rest pose, and the arm moves < 20 deg/s (less motion blur and less
   video/joint-state sync error). From the candidates we take 5 frames with farthest-point
   sampling in joint space (each new frame is the one most different from those already picked),
   so no two frames show nearly the same pose. Sorted by time, frames 1, 3, 5 are FIT frames and
   2, 4 are HELD-OUT. Joint values are observation.state (follower arm), not action (leader arm).

2. Real arm mask (`real_arm_mask`). The arm is black on a light table: pixel is "arm" if
   max(R,G,B) < DARK_V and it is not green (the green cube's shadowed faces are dark too).
   A small closing (CLOSE_PX) fills specular speckle, then an opening with a disk of OPEN_PX pixels (at 640x480) removes thin black wires.
   Known leaks: the thick black USB cable of the wrist camera survives where it is wider than
   OPEN_PX; shiny arm faces that reflect the sky are missed; white tape on the real wrist is missed.
   So we also report a "thick-removed" IoU where BOTH masks get the same larger opening.

3. Sim mask (`SimArm`). Menagerie so101.xml, qpos = lerobot_to_sim(state, ArmCalib), forward
   kinematics only (no physics), MuJoCo segmentation render: a pixel is "arm" if it hits any
   robot geom (the scene contains nothing else).

4. Camera fit, per episode (`fit_camera`). The camera aims at table point (lx, ly, 0) from
   distance d, azimuth az, elevation el, with roll and vertical fov fovy: 7 numbers. We maximise
   the mean IoU on the episode's FIT frames: random search over a wide box at 64x48, then
   Nelder-Mead from the best few starts at 160x120 (IoU of slightly blurred masks so the optimiser
   sees a slope). The camera is then frozen and we report hard IoU at 640x480 on the HELD-OUT
   frames: their joint angles were never used to place the camera.

5. Sensitivity. The same fit+held-out evaluation is repeated with deliberately wrong conversions
   (each one gets its own camera fit, so the camera can try to hide the error). If a wrong mapping
   scores about as well as the default, the gate cannot tell right from wrong.

6. Ceiling (`ceiling`). Every held-out frame gets its OWN camera. This is the best IoU these masks
   and this robot model can reach when the camera may absorb any error; if it is far below the pass
   mark, the pass mark cannot be reached whatever the joint mapping.

7. Offsets (`fit-offsets`). Per-joint offsets (+-15 deg) fitted with the cameras on FIT frames
   (alternating offsets / cameras), evaluated on HELD-OUT frames.

Usage (inside the `sprint` container, MUJOCO_GL=osmesa; CPU only, keep --workers <= 4):
  python -m calib.overlay_gate frames --episodes 1 155 205 265 373 --out /work/outputs_large/calib/overlay
  python -m calib.overlay_gate run --out /work/outputs_large/calib/overlay --only default --tag default
  python -m calib.overlay_gate run --out ... --only shoulder_lift_sign_flipped elbow_flex_plus20 \
      wrist_flex_plus20 wrist_roll_offset_0 --tag wrong --all-overlays
  python -m calib.overlay_gate ceiling --out ... --split heldout
  python -m calib.overlay_gate fit-offsets --out ... --init results_default.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from calib.so101_units import JOINTS, ArmCalib, lerobot_to_sim

REPO = Path(__file__).resolve().parents[1]
SO101_XML = REPO / "third_party/menagerie_so101/robotstudio_so101/so101.xml"
DATASET = "MinhDuk/final_dataset_annotated"

DARK_V = 70          # max(R,G,B) below this = black arm (table ~150-210, arm ~10-60)
OPEN_PX = 7          # opening disk diameter at 640x480: drops wires thinner than this
THICK_OPEN_PX = 21   # larger opening applied to BOTH masks for the "thick-removed" IoU
MIN_FRAC = 0.05      # frames where the real arm covers less than this are skipped
REST_DEG = 20.0      # frames within this of the start/end pose are skipped (no pose variety)
MAX_SPEED = 20.0     # deg/s; faster frames are skipped (blur, sync error)
CLOSE_PX = 5         # closing before the opening fills specular speckle inside the black arm
W, H = 640, 480

# Search box for the camera: lookat x, y (m), distance (m), azimuth, elevation, roll, fovy (deg).
# The search is also limited to physically plausible cameras (`plausible`): camera height
# CAM_Z m above the table and at least MIN_CLEAR m from every robot link origin (otherwise the
# optimiser "wins" by putting the camera inside the arm so the arm covers the whole image).
CAM_LO = np.array([-0.20, -0.50, 0.15, -180.0, 35.0, -180.0, 35.0])
CAM_HI = np.array([0.60, 0.50, 1.00, 180.0, 89.0, 180.0, 75.0])
CAM_Z = (0.20, 0.90)
MIN_CLEAR = 0.10
CAM_NAMES = ["lookat_x", "lookat_y", "dist", "azimuth", "elevation", "roll", "fovy"]


# ----------------------------------------------------------------------------- masks / metrics

def _disk(px: int):
  import cv2
  return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def open_mask(m: np.ndarray, px: int) -> np.ndarray:
  """Morphological opening with a disk of diameter px (scaled to the mask's width vs 640)."""
  import cv2
  px = max(1, int(round(px * m.shape[1] / W)))
  if px <= 1:
    return m.astype(bool)
  return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, _disk(px)).astype(bool)


def real_arm_mask(img: np.ndarray, dark_v: int = DARK_V, open_px: int = OPEN_PX) -> np.ndarray:
  """img: uint8 [H,W,3] RGB -> bool [H,W] arm mask (see module docstring, step 2)."""
  x = img.astype(np.int16)
  r, g, b = x[..., 0], x[..., 1], x[..., 2]
  dark = x.max(-1) < dark_v
  green = (g > r + 12) & (g > b + 4)
  m = (dark & ~green).astype(np.uint8)
  import cv2
  m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(CLOSE_PX))
  return open_mask(m, open_px)


def resize_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
  import cv2
  if m.shape[1] == w and m.shape[0] == h:
    return m.astype(bool)
  return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
  u = np.logical_or(a, b).sum()
  return float(np.logical_and(a, b).sum() / u) if u else 1.0


def soft_iou(a: np.ndarray, b: np.ndarray, sigma: float = 1.0) -> float:
  """IoU of Gaussian-blurred masks (sum min / sum max): same optimum, smoother landscape."""
  import cv2
  a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)
  b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), sigma)
  den = np.maximum(a, b).sum()
  return float(np.minimum(a, b).sum() / den) if den > 0 else 1.0


# ----------------------------------------------------------------------------- camera / sim

def cam_frame(p) -> tuple[np.ndarray, np.ndarray]:
  """7 camera numbers -> (position [3], 3x3 rotation whose columns are camera x, y, z axes).
  MuJoCo cameras look along their -z axis with +y up in the image."""
  lx, ly, dist, az, el, roll = p[:6]
  az, el, roll = np.deg2rad([az, el, roll])
  back = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
  pos = np.array([lx, ly, 0.0]) + dist * back
  f = -back
  x = np.cross(f, [0.0, 0.0, 1.0])
  x /= np.linalg.norm(x)
  y = np.cross(x, f)
  xr = np.cos(roll) * x + np.sin(roll) * y
  yr = -np.sin(roll) * x + np.cos(roll) * y
  return pos, np.column_stack([xr, yr, -f])


class SimArm:
  """Menagerie SO-101 with one free camera; renders binary robot masks."""

  def __init__(self, w: int = W, h: int = H):
    import mujoco
    self.mj = mujoco
    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.worldbody.add_camera(name="fit", pos=[0, 0, 1], fovy=60)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    self.cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "fit")
    self.renderers = {}
    self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                 for j in JOINTS]

  def _renderer(self, w, h):
    if (w, h) not in self.renderers:
      r = self.mj.Renderer(self.m, h, w)
      r.enable_segmentation_rendering()
      self.renderers[(w, h)] = r
    return self.renderers[(w, h)]

  def set_camera(self, p) -> None:
    pos, R = cam_frame(p)
    q = np.zeros(4)
    self.mj.mju_mat2Quat(q, R.flatten())
    self.m.cam_pos[self.cam], self.m.cam_quat[self.cam], self.m.cam_fovy[self.cam] = pos, q, p[6]

  def mask(self, qpos_sim: np.ndarray, w: int = W, h: int = H) -> np.ndarray:
    self.d.qpos[self.qadr] = qpos_sim
    self.mj.mj_kinematics(self.m, self.d)
    self.mj.mj_camlight(self.m, self.d)  # camera poses are NOT updated by mj_kinematics
    r = self._renderer(w, h)
    r.update_scene(self.d, camera=self.cam)
    seg = r.render()
    return seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)


def plausible(p, link_pos) -> bool:
  """In the search box, camera height within CAM_Z, and not closer than MIN_CLEAR to any link."""
  p = np.asarray(p)
  if np.any(p < CAM_LO) or np.any(p > CAM_HI):
    return False
  pos, _ = cam_frame(p)
  return CAM_Z[0] <= pos[2] <= CAM_Z[1] and np.linalg.norm(link_pos - pos, axis=-1).min() >= MIN_CLEAR


def link_positions(sim: SimArm, qs) -> np.ndarray:
  """World positions of all robot body origins in the given poses, [n, 3]."""
  out = []
  for q in qs:
    sim.d.qpos[sim.qadr] = q
    sim.mj.mj_kinematics(sim.m, sim.d)
    out.append(sim.d.xpos[1:].copy())
  return np.concatenate(out)


def score(sim: SimArm, p, qs, reals, w, h, soft=True) -> float:
  sim.set_camera(p)
  f = soft_iou if soft else iou
  return float(np.mean([f(sim.mask(q, w, h), r) for q, r in zip(qs, reals)]))


def fit_camera(qs, real_masks, seed=0, n_random=4000, n_starts=8, nm_iters=600, sim=None):
  """Maximise mean IoU over (qpos, real mask) pairs. Returns (camera params, soft IoU @160x120).
  Random starts aim at a point near the gripper (the gripper is in view in nearly every frame);
  everything else is uniform in the search box. Implausible cameras score -1 (a wall for NM)."""
  from scipy.optimize import minimize
  sim = sim or SimArm()
  rng = np.random.default_rng(seed)
  links = link_positions(sim, qs)
  grip = links.reshape(len(qs), -1, 3)[:, -3:].reshape(-1, 3)  # gripper, camera mount, jaw
  lo_r = [resize_mask(m, 64, 48) for m in real_masks]
  hi_r = [resize_mask(m, 160, 120) for m in real_masks]
  cands = []
  while len(cands) < n_random:
    p = CAM_LO + rng.random(7) * (CAM_HI - CAM_LO)
    p[:2] = grip[rng.integers(len(grip)), :2] + rng.normal(0, 0.08, 2)
    if plausible(p, links):
      cands.append(p)
  cands = np.array(cands)
  s = np.array([score(sim, p, qs, lo_r, 64, 48) for p in cands])
  starts = cands[np.argsort(-s)[:n_starts]]
  scale = (CAM_HI - CAM_LO) * 0.03

  def obj(z, p0):
    p = p0 + z * scale
    return 1.0 if not plausible(p, links) else -score(sim, p, qs, hi_r, 160, 120)

  best = (None, -1.0)
  for p0 in starts:
    res = minimize(obj, np.zeros(7), args=(p0,), method="Nelder-Mead",
                   options={"maxiter": nm_iters, "xatol": 1e-3, "fatol": 1e-4,
                            "initial_simplex": np.vstack([np.zeros(7), np.eye(7)])})
    if -res.fun > best[1]:
      best = (p0 + res.x * scale, -res.fun)
  return best


# ----------------------------------------------------------------------------- variants

def variants() -> dict[str, ArmCalib]:
  """Default mapping plus deliberately wrong ones for the sensitivity check."""
  d = ArmCalib()
  def off(j, v):
    o = list(d.offset_deg); o[j] = v; return replace(d, offset_deg=o, source=f"wrong: offset[{j}]={v}")
  def sgn(j):
    s = list(d.sign); s[j] = -s[j]; return replace(d, sign=s, source=f"wrong: sign[{j}] flipped")
  return {"default": d,
          "shoulder_lift_sign_flipped": sgn(1),
          "elbow_flex_plus20": off(2, 20.0),
          "wrist_flex_plus20": off(3, 20.0),
          "wrist_roll_offset_0": off(4, 0.0)}


# ----------------------------------------------------------------------------- stage: frames

def cmd_frames(a):
  import glob
  from lerobot.datasets.lerobot_dataset import LeRobotDataset
  root = a.root or glob.glob(f"/work/hf_home/hub/datasets--{DATASET.replace('/', '--')}/snapshots/*")[0]
  ds = LeRobotDataset(DATASET, root=root, video_backend="pyav")
  out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
  imgs, states, epi, fidx, split, speed, frac = [], [], [], [], [], [], []
  for e in a.episodes:
    ep = ds.meta.episodes[e]
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    st = np.stack([np.asarray(s) for s in ds.hf_dataset.select(range(lo, hi))["observation.state"]])
    v = np.zeros(len(st))  # joint speed, deg/s, max over arm joints (central difference)
    v[1:-1] = np.abs(st[2:, :5] - st[:-2, :5]).max(1) * ds.fps / 2
    v[0], v[-1] = v[1], v[-2]
    # "active" = the arm has left its rest pose (> REST_DEG from the first and last state)
    away = np.minimum(np.abs(st[:, :5] - st[0, :5]).max(1), np.abs(st[:, :5] - st[-1, :5]).max(1))
    cand = []
    for k in range(0, hi - lo, a.step):
      if away[k] < REST_DEG:
        continue
      img = (ds[lo + k]["observation.images.cam_head"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
      fr = real_arm_mask(img).mean()
      if fr >= MIN_FRAC:
        cand.append((k, img, fr))
    slow = [c for c in cand if v[c[0]] < MAX_SPEED]
    cand = slow if len(slow) >= 5 else cand
    if len(cand) < 5:
      print(f"episode {e}: only {len(cand)} usable frames, skipped"); continue
    pose = np.array([st[c[0], :5] for c in cand])
    pick = [int(np.argmax(np.abs(pose - st[0, :5]).max(1)))]  # start: farthest from rest
    while len(pick) < 5:  # farthest-point sampling (max-abs joint difference, deg)
      dmin = np.min([np.abs(pose - pose[j]).max(1) for j in pick], axis=0)
      pick.append(int(np.argmax(dmin)))
    for c, j in enumerate(sorted(pick, key=lambda j: cand[j][0])):
      k, img, fr = cand[j]
      imgs.append(img); states.append(st[k]); epi.append(e); fidx.append(k)
      split.append("fit" if c % 2 == 0 else "heldout"); speed.append(v[k]); frac.append(fr)
    print(f"episode {e}: {len(cand)} usable, picked {fidx[-5:]} speed {np.round(speed[-5:], 1)}")
  np.savez_compressed(out / "frames.npz", head=np.stack(imgs), state=np.stack(states),
                      episode=np.array(epi), frame_index=np.array(fidx), split=np.array(split),
                      speed=np.array(speed), arm_frac=np.array(frac))


# ----------------------------------------------------------------------------- stage: run

def _fit_job(args):
  """One (variant, episode) camera fit; runs in a worker process with its own renderer."""
  name, cal_dict, e, states, real_masks, seed, effort = args
  cal = ArmCalib(**cal_dict)
  qs = [lerobot_to_sim(s, cal) for s in states]
  p, s = fit_camera(qs, real_masks, seed=seed, **effort)
  return name, e, p.tolist(), s


def overlay_png(path, img, real, sim_m, text):
  import cv2
  o = img.copy()
  for m, col in ((real, (0, 255, 0)), (sim_m, (255, 0, 255))):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cs, -1, col, 2)
  cv2.putText(o, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
  cv2.imwrite(str(path), o[..., ::-1])


def cmd_run(a):
  import multiprocessing as mp
  out = Path(a.out)
  z = np.load(out / "frames.npz")
  imgs, states, epi, split = z["head"], z["state"], z["episode"], z["split"]
  reals = [real_arm_mask(im) for im in imgs]
  eps = sorted(set(epi.tolist()))
  vs = variants()
  if a.only:
    vs = {k: vs[k] for k in a.only}
  from dataclasses import asdict
  jobs = []
  for name, cal in vs.items():
    for e in eps:
      ix = np.where((epi == e) & (split == "fit"))[0]
      jobs.append((name, asdict(cal), e, states[ix], [reals[i] for i in ix], a.seed, _effort(a)))
  with mp.get_context("spawn").Pool(a.workers) as pool:
    fits = pool.map(_fit_job, jobs, chunksize=1)
  cams = {(n, e): (p, s) for n, e, p, s in fits}

  res = {"method": __doc__.split("Usage")[0].strip(), "params": _params(), "variants": {}}
  sim = SimArm()
  tag = f"_{a.tag}" if a.tag else ""
  for name, cal in vs.items():
    res["variants"][name] = summarize(name + tag, cal, {e: cams[(name, e)] for e in eps}, z, reals, sim,
                                      overlays=(name == "default" or a.all_overlays), out=out)
  (out / f"results{tag}.json").write_text(json.dumps(res, indent=1))


def _effort(a):
  return dict(n_random=a.n_random, n_starts=a.n_starts, nm_iters=a.nm_iters)


def _params():
  return dict(DARK_V=DARK_V, OPEN_PX=OPEN_PX, CLOSE_PX=CLOSE_PX, THICK_OPEN_PX=THICK_OPEN_PX,
              MIN_FRAC=MIN_FRAC, REST_DEG=REST_DEG, MAX_SPEED=MAX_SPEED,
              cam_search_lo=CAM_LO.tolist(), cam_search_hi=CAM_HI.tolist(), cam_names=CAM_NAMES)


def summarize(name, cal, cams, z, reals, sim, overlays, out):
  """Hard IoU at 640x480 for every frame with each episode's frozen camera; overlay PNGs."""
  from dataclasses import asdict
  imgs, states, epi, split = z["head"], z["state"], z["episode"], z["split"]
  eps = sorted(cams)
  rec = {"calib": asdict(cal), "episodes": {}, "frames": []}
  ovl = Path(out) / "overlays"; ovl.mkdir(parents=True, exist_ok=True)
  for e in eps:
    p, s_fit = cams[e]
    rec["episodes"][str(e)] = {"camera": dict(zip(CAM_NAMES, np.round(p, 4).tolist())),
                                "fit_soft_iou_160": s_fit}
    sim.set_camera(p)
    for i in np.where(epi == e)[0]:
      m = sim.mask(lerobot_to_sim(states[i], cal))
      u = iou(m, reals[i])
      ut = iou(open_mask(m, THICK_OPEN_PX), open_mask(reals[i], THICK_OPEN_PX))
      rec["frames"].append({"episode": int(e), "frame_index": int(z["frame_index"][i]),
                            "split": str(split[i]), "iou": u, "iou_thick_removed": ut,
                            "speed_deg_s": float(z["speed"][i]), "arm_frac": float(z["arm_frac"][i])})
      if overlays:
        overlay_png(ovl / f"{name}_ep{e}_f{int(z['frame_index'][i])}_{split[i]}.png", imgs[i],
                    reals[i], m, f"{name} ep{e} {split[i]} IoU {u:.2f} (thick-removed {ut:.2f})")
  fr = rec["frames"]
  for sp in ("fit", "heldout"):
    rec[f"{sp}_mean_iou"] = float(np.mean([f["iou"] for f in fr if f["split"] == sp]))
    rec[f"{sp}_mean_iou_thick_removed"] = float(np.mean([f["iou_thick_removed"] for f in fr
                                                         if f["split"] == sp]))
  rec["heldout_mean_iou_per_episode"] = {
    str(e): round(float(np.mean([f["iou"] for f in fr if f["split"] == "heldout" and f["episode"] == e])), 4)
    for e in eps}
  print(f"{name:28s} fit {rec['fit_mean_iou']:.3f}  heldout {rec['heldout_mean_iou']:.3f}  "
        f"(thick-removed {rec['fit_mean_iou_thick_removed']:.3f} / "
        f"{rec['heldout_mean_iou_thick_removed']:.3f})  per-ep {rec['heldout_mean_iou_per_episode']}",
        flush=True)
  return rec


def cmd_ceiling(a):
  """Upper bound: give EVERY frame its own camera (default mapping). No held-out meaning; it shows
  how high IoU can get with these masks and this robot model when the camera can absorb anything."""
  import multiprocessing as mp
  from dataclasses import asdict
  out = Path(a.out)
  z = np.load(out / "frames.npz")
  reals = [real_arm_mask(im) for im in z["head"]]
  cal = ArmCalib()
  keep = [i for i in range(len(reals)) if a.split in ("all", str(z["split"][i]))]
  jobs = [("ceiling", asdict(cal), i, z["state"][i:i + 1], [reals[i]], a.seed, _effort(a)) for i in keep]
  with mp.get_context("spawn").Pool(a.workers) as pool:
    fits = pool.map(_fit_job, jobs, chunksize=1)
  sim, rows = SimArm(), []
  for _, i, p, _ in fits:
    sim.set_camera(p)
    m = sim.mask(lerobot_to_sim(z["state"][i], cal))
    rows.append({"episode": int(z["episode"][i]), "frame_index": int(z["frame_index"][i]),
                 "split": str(z["split"][i]), "iou": iou(m, reals[i]), "camera": p})
    overlay_png(out / "overlays" / f"ceiling_ep{z['episode'][i]}_f{z['frame_index'][i]}.png", z["head"][i],
                reals[i], m, f"per-frame camera IoU {rows[-1]['iou']:.2f}")
  res = {"method": cmd_ceiling.__doc__, "mean_iou": float(np.mean([r["iou"] for r in rows])), "frames": rows}
  print(f"per-frame-camera ceiling: mean IoU {res['mean_iou']:.3f}", flush=True)
  (out / "results_ceiling.json").write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- stage: fit offsets

def _refit_job(args):
  """Nelder-Mead refinement of one episode's camera from a given start (no random search)."""
  from scipy.optimize import minimize
  p0, cal_dict, states, real_masks = args
  cal, sim = ArmCalib(**cal_dict), SimArm()
  qs = [lerobot_to_sim(s, cal) for s in states]
  hi_r = [resize_mask(m, 160, 120) for m in real_masks]
  scale = (CAM_HI - CAM_LO) * 0.02
  p0 = np.asarray(p0)
  links = link_positions(sim, qs)
  res = minimize(lambda z: 1.0 if not plausible(p0 + z * scale, links)
                 else -score(sim, p0 + z * scale, qs, hi_r, 160, 120), np.zeros(7),
                 method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4,
                                                "initial_simplex": np.vstack([np.zeros(7), np.eye(7)])})
  return (p0 + res.x * scale).tolist(), -res.fun


def cmd_fit_offsets(a):
  """Step 6: per-joint offsets (+-BOUND deg around the default) fitted jointly with the cameras,
  on FIT frames only, by alternating (a) offsets with cameras frozen, (b) cameras with offsets frozen.
  Held-out frames are only used for the final report."""
  import multiprocessing as mp
  from dataclasses import asdict
  from scipy.optimize import minimize
  out = Path(a.out)
  z = np.load(out / "frames.npz")
  states, epi, split = z["state"], z["episode"], z["split"]
  reals = [real_arm_mask(im) for im in z["head"]]
  eps = sorted(set(epi.tolist()))
  prev = json.loads((out / a.init).read_text())["variants"]["default"]["episodes"]
  cams = {e: np.array([prev[str(e)]["camera"][k] for k in CAM_NAMES]) for e in eps}
  base = ArmCalib()
  fit_ix = {e: np.where((epi == e) & (split == "fit"))[0] for e in eps}
  hi_r = {e: [resize_mask(reals[i], 160, 120) for i in fit_ix[e]] for e in eps}
  sim = SimArm()
  delta = np.zeros(5)

  def cal_of(dl):
    return replace(base, offset_deg=(np.asarray(base.offset_deg) + np.clip(dl, -a.bound, a.bound)).tolist())

  def obj(dl):
    cal = cal_of(dl)
    return -np.mean([score(sim, cams[e], [lerobot_to_sim(states[i], cal) for i in fit_ix[e]],
                           hi_r[e], 160, 120) for e in eps])

  for rnd in range(a.rounds):
    res = minimize(obj, delta, method="Nelder-Mead",
                   options={"maxiter": 250, "xatol": 0.2, "fatol": 1e-4,
                            "initial_simplex": np.vstack([delta, delta + 4 * np.eye(5)])})
    delta = np.clip(res.x, -a.bound, a.bound)
    print(f"round {rnd}: offsets delta {np.round(delta, 1).tolist()} fit soft IoU {-res.fun:.3f}", flush=True)
    jobs = [(cams[e].tolist(), asdict(cal_of(delta)), states[fit_ix[e]], [reals[i] for i in fit_ix[e]])
            for e in eps]
    with mp.get_context("spawn").Pool(len(eps)) as pool:
      r = pool.map(_refit_job, jobs)
    cams = {e: np.array(p) for e, (p, _) in zip(eps, r)}
    print(f"round {rnd}: cameras refit, fit soft IoU {np.mean([s for _, s in r]):.3f}", flush=True)
  cal = cal_of(delta)
  cal.source = (f"overlay_gate fit-offsets: arm offsets fitted (bound +-{a.bound} deg around defaults) jointly "
                f"with per-episode cameras on FIT frames of episodes {eps} of {DATASET} cam_head; "
                f"delta from default {np.round(delta, 2).tolist()}")
  cal.save(REPO / "calib/arm_calib_friend.json")
  final = {e: (cams[e], -obj(delta)) for e in eps}
  res = {"method": cmd_fit_offsets.__doc__, "params": _params(),
         "variants": {"fitted_offsets": summarize("fitted", cal, final, z, reals, sim, True, out)}}
  (out / "results_fitted.json").write_text(json.dumps(res, indent=1))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  f = sub.add_parser("frames")
  f.add_argument("--episodes", type=int, nargs="+", required=True)
  f.add_argument("--out", required=True)
  f.add_argument("--root", default=None)
  f.add_argument("--step", type=int, default=5)
  r = sub.add_parser("run")
  r.add_argument("--out", required=True)
  r.add_argument("--only", nargs="*", default=None, help="variant names to run")
  r.add_argument("--workers", type=int, default=4, help="worker processes (CPU is shared with training: keep <= 4)")
  r.add_argument("--seed", type=int, default=0)
  r.add_argument("--tag", default="")
  r.add_argument("--all-overlays", action="store_true")
  c = sub.add_parser("ceiling")
  c.add_argument("--out", required=True)
  c.add_argument("--workers", type=int, default=4, help="worker processes (CPU is shared with training: keep <= 4)")
  c.add_argument("--seed", type=int, default=0)
  c.add_argument("--split", default="heldout", choices=["all", "fit", "heldout"])
  for x in (r, c):
    x.add_argument("--n-random", type=int, default=4000, help="random cameras scored at 64x48")
    x.add_argument("--n-starts", type=int, default=8, help="Nelder-Mead starts at 160x120")
    x.add_argument("--nm-iters", type=int, default=600)
  o = sub.add_parser("fit-offsets")
  o.add_argument("--out", required=True)
  o.add_argument("--init", default="results.json", help="results file with default-mapping cameras")
  o.add_argument("--bound", type=float, default=15.0)
  o.add_argument("--rounds", type=int, default=3)
  a = ap.parse_args()
  {"frames": cmd_frames, "run": cmd_run, "ceiling": cmd_ceiling, "fit-offsets": cmd_fit_offsets}[a.cmd](a)


if __name__ == "__main__":
  main()
