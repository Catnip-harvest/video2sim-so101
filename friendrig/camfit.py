"""Stage 1: fit the overhead camera (cam_head) of the friend's rig, using the arm as the target.

Plain-English method (each step can be said out loud):

1. Frames (`frames`). For every episode we pick up to K frames where the arm is clearly in view
   and moving slowly (< MAX_SPEED deg/s, so video/joint sync error is small), spread out in joint
   space (farthest-point sampling). We keep the real arm mask (dark pixels, see masks.py) and an
   IGNORE mask (thin dark things such as the wrist-camera cable). Joint values are
   observation.state (the follower arm). Units: calib/so101_units.lerobot_to_sim.

2. Sim silhouette. The Menagerie SO-101 is posed at the same joint values. Inside the optimiser
   the silhouette comes from PointArm (surface points, no OpenGL, ~1 ms); every number we REPORT
   is re-measured with MuJoCo's own renderer on the full meshes at 640x480 (`eval`).

3. Score. Per frame: IoU of sim mask vs real arm mask, with ignore pixels left out on both sides.
   Inside the optimiser we use a slightly blurred ("soft") IoU at 160x120 so it sees a slope, and
   a trimmed mean over frames (worst TRIM fraction dropped) so a few bad frames (operator's hand,
   bad sync) cannot drag the camera.

4. The camera is NOT fixed over the whole dataset (montage of first frames, 13 Oct: table lines
   and the parked arm move between recording sessions). So:
   a. `fit-global`: one camera for all TRAIN frames (random search, then CMA-ES).
   b. `fit-episodes`: each TRAIN episode refits the camera from the global one on its own frames.
   c. `sessions`: consecutive TRAIN episodes whose cameras agree (table points move < SESSION_PX
      pixels) form one session.
   d. `fit-sessions`: one camera per session on all TRAIN frames of the session (optionally with
      shared per-joint offsets, `--offsets`).
   A HELD-OUT episode uses the camera of the session it was recorded in (by episode index: the
   train episodes before and after it). None of its frames is used to place that camera.

5. `eval`: hard IoU at 640x480 on HELD-OUT frames (and TRAIN frames for comparison), overlays.

Usage (container `sprint`, cwd = repo, MUJOCO_GL=osmesa, PYTHONPATH += pylib for cma):
  python -m friendrig.camfit frames   --out OUT --workers 8
  python -m friendrig.camfit fit-global --out OUT --workers 8
  python -m friendrig.camfit fit-episodes --out OUT --workers 8
  python -m friendrig.camfit sessions --out OUT
  python -m friendrig.camfit fit-sessions --out OUT --workers 8 [--offsets]
  python -m friendrig.camfit eval --out OUT --workers 8 --cams sessions.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from calib.so101_units import ArmCalib, lerobot_to_sim
from friendrig.masks import arm_masks, iou, resize_bool
from friendrig.simcam import CAM_NAMES, PointArm, SimArm, backproject_to_plane, cam_pose, project

K = 10
MAX_SPEED = 40.0     # deg/s
MIN_ARM = 0.02       # fraction of the image that must be arm (core mask)
TRIM = 0.1
LW, LH = 160, 120    # optimiser resolution
SESSION_PX = 12.0    # max table-point motion (px at 640x480) inside one session
# broad random-search box for the global init
BOX_LO = np.array([-0.20, -0.50, 0.20, -np.pi, -0.7, -0.7, 35.0])
BOX_HI = np.array([0.60, 0.50, 1.00, np.pi, 0.7, 0.7, 85.0])
# a 5x5 grid of table points in front of the robot, used to compare cameras in pixels
TABLE_PTS = np.array([[x, y, 0.0] for x in np.linspace(0.10, 0.35, 5) for y in np.linspace(-0.2, 0.2, 5)])


def split_of() -> dict[int, str]:
  from friendrig.realdata import split
  s = split()
  return {**{e: "train" for e in s["train"]}, **{e: "held_out" for e in s["held_out"]}}


# ------------------------------------------------------------------------------------- frames

def _frames_job(ep):
  from friendrig.realdata import FriendData
  D = _G.setdefault("D", FriendData())
  st = D.states(ep)
  n = len(st)
  v = np.zeros(n)
  v[1:-1] = np.abs(st[2:, :5] - st[:-2, :5]).max(1) * D.fps / 2
  v[0], v[-1] = v[1], v[-2]
  cand = [k for k in range(0, n, 3) if v[k] < MAX_SPEED]
  if len(cand) < K:
    return ep, []
  pose = st[cand, :5]
  pick = [int(np.argmax(np.abs(pose - st[0, :5]).max(1)))]
  while len(pick) < 2 * K:
    dmin = np.min([np.abs(pose - pose[j]).max(1) for j in pick], axis=0)
    pick.append(int(np.argmax(dmin)))
  ks = sorted(cand[j] for j in pick)
  imgs = D.frames(ep, "cam_head", ks)
  rows = []
  for k, img in zip(ks, imgs):
    arm, ign = arm_masks(img)
    if arm.mean() >= MIN_ARM:
      rows.append((k, st[k], float(v[k]), np.packbits(arm), np.packbits(ign)))
  # keep K spread over time
  if len(rows) > K:
    rows = [rows[i] for i in np.linspace(0, len(rows) - 1, K).round().astype(int)]
  return ep, rows


_G: dict = {}


def cmd_frames(a):
  import multiprocessing as mp
  from friendrig.realdata import FriendData
  out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
  eps = FriendData().episodes
  sp = split_of()
  t0 = time.time()
  with mp.get_context("fork").Pool(a.workers) as pool:
    res = pool.map(_frames_job, eps, chunksize=4)
  E, F, S, V, A, I, SP = [], [], [], [], [], [], []
  for ep, rows in res:
    for k, s, v, arm, ign in rows:
      E.append(ep); F.append(k); S.append(s); V.append(v); A.append(arm); I.append(ign); SP.append(sp.get(ep, "?"))
  np.savez_compressed(out / "frames.npz", episode=np.array(E), frame_index=np.array(F), state=np.stack(S),
                      speed=np.array(V), arm=np.stack(A), ignore=np.stack(I), split=np.array(SP))
  per = {}
  for ep, rows in res:
    per[ep] = len(rows)
  print(f"frames: {len(E)} from {sum(1 for x in per.values() if x)} / {len(eps)} episodes "
        f"(train {SP.count('train')}, held_out {SP.count('held_out')}), {time.time() - t0:.0f} s", flush=True)


def load_frames(out: Path):
  z = dict(np.load(Path(out) / "frames.npz"))
  n = len(z["episode"])
  z["arm"] = np.unpackbits(z["arm"], axis=1)[:, :640 * 480].reshape(n, 480, 640).astype(bool)
  z["ignore"] = np.unpackbits(z["ignore"], axis=1)[:, :640 * 480].reshape(n, 480, 640).astype(bool)
  return z


# ------------------------------------------------------------------------------------- scoring

def lowres(arm, ign, w=LW, h=LH):
  """Real masks at optimiser resolution: arm as float coverage, keep = not-ignore."""
  import cv2
  a = cv2.resize(arm.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
  k = cv2.resize((~ign).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
  return cv2.GaussianBlur(a, (0, 0), 0.8), k


def soft_iou(sim_m, real_soft, keep):
  import cv2
  s = cv2.GaussianBlur(sim_m.astype(np.float32), (0, 0), 0.8)
  s, r = s[keep], real_soft[keep]
  den = np.maximum(s, r).sum()
  return float(np.minimum(s, r).sum() / den) if den > 0 else 0.0


def trimmed(x, trim=TRIM):
  x = np.sort(np.asarray(x))
  k = int(len(x) * trim)
  return float(x[k:].mean()) if len(x) - k > 0 else float(x.mean())


class Scorer:
  """Holds the frames of one fitting problem; scores a camera (and optional joint offsets)."""

  def __init__(self, pa: PointArm, states, reals, offsets=None, w=LW, h=LH):
    self.pa, self.w, self.h = pa, w, h
    self.states = np.asarray(states)
    self.reals = reals
    self.set_offsets(offsets)

  def set_offsets(self, offsets):
    cal = ArmCalib() if offsets is None else ArmCalib(offset_deg=list(np.asarray(ArmCalib().offset_deg) + offsets))
    self.qs = lerobot_to_sim(self.states, cal)
    self.pts = [self.pa.world_points(q) for q in self.qs]
    self.links = np.concatenate([self.pa.d.xpos[1:].copy() for q in self.qs[:1]])

  def per_frame(self, p):
    return [soft_iou(self.pa.mask(p, w=self.w, h=self.h, pts=P), r, k) for P, (r, k) in zip(self.pts, self.reals)]

  def __call__(self, p, trim=TRIM):
    p = np.asarray(p)
    if p[2] < 0.12 or p[2] > 1.5 or not (20 <= p[6] <= 100):
      return 0.0
    return trimmed(self.per_frame(p), trim)


def cma_fit(score, x0, sigma_vec, iters=120, popsize=16, seed=0):
  import cma
  es = cma.CMAEvolutionStrategy(np.zeros(len(x0)), 1.0,
                                {"popsize": popsize, "seed": seed + 1, "verbose": -9, "maxiter": iters,
                                 "tolfun": 1e-5, "tolx": 1e-4})
  x0, sv = np.asarray(x0, float), np.asarray(sigma_vec, float)
  best = (x0, score(x0))
  while not es.stop():
    zs = es.ask()
    fs = [-score(x0 + z * sv) for z in zs]
    es.tell(zs, fs)
    i = int(np.argmin(fs))
    if -fs[i] > best[1]:
      best = (x0 + zs[i] * sv, -fs[i])
  return best


CAM_SIGMA = np.array([0.03, 0.03, 0.04, 0.10, 0.06, 0.06, 4.0])


# ------------------------------------------------------------------------------------- global

def _init_worker(out):
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  _G["pa"] = PointArm(n_points=int(os.environ.get("FR_POINTS", "20000")))
  _G["out"] = out


def _rand_job(args):
  idx_states, real_list, cams = args
  pa = _G["pa"]
  sc = Scorer(pa, idx_states, real_list, w=80, h=60)
  return [sc(p, trim=0.0) for p in cams]


def cmd_fit_global(a):
  import multiprocessing as mp
  out = Path(a.out)
  z = load_frames(out)
  rng = np.random.default_rng(a.seed)
  tr = np.where(z["split"] == "train")[0]
  sub = rng.choice(tr, min(a.n_frames_rand, len(tr)), replace=False)
  reals80 = [lowres(z["arm"][i], z["ignore"][i], 80, 60) for i in sub]
  cams = BOX_LO + rng.random((a.n_random, 7)) * (BOX_HI - BOX_LO)
  chunks = np.array_split(cams, a.workers * 4)
  t0 = time.time()
  with mp.get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(str(out),)) as pool:
    sc = pool.map(_rand_job, [(z["state"][sub], reals80, c) for c in chunks])
    s = np.concatenate([np.asarray(x) for x in sc])
    order = np.argsort(-s)[:a.n_starts]
    print(f"random search: {a.n_random} cameras on {len(sub)} frames @80x60, best {s[order[:5]].round(3)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    # CMA-ES from the best starts on more frames at 160x120
    sub2 = rng.choice(tr, min(a.n_frames_cma, len(tr)), replace=False)
    jobs = [(z["state"][sub2], [lowres(z["arm"][i], z["ignore"][i]) for i in sub2], cams[j], a.seed + k)
            for k, j in enumerate(order)]
    res = pool.map(_cma_job, jobs)
  best = max(res, key=lambda r: r[1])
  rec = {"camera": dict(zip(CAM_NAMES, np.asarray(best[0]).tolist())), "fit_soft_iou_trimmed": best[1],
         "starts": [{"camera": np.asarray(r[0]).tolist(), "score": r[1]} for r in res],
         "n_random": a.n_random, "n_frames_rand": int(len(sub)), "n_frames_cma": int(len(sub2))}
  (out / "cam_global.json").write_text(json.dumps(rec, indent=1))
  print(f"global camera {np.round(best[0], 3).tolist()} soft IoU (trimmed) {best[1]:.3f} "
        f"({time.time() - t0:.0f} s)", flush=True)


def _cma_job(args):
  states, reals, p0, seed = args[:4]
  iters = args[4] if len(args) > 4 else 150
  sigma = args[5] if len(args) > 5 else CAM_SIGMA * 2
  sc = Scorer(_G["pa"], states, reals)
  return cma_fit(sc, p0, sigma, iters=iters, seed=seed)


# ------------------------------------------------------------------------------------- per episode

def cmd_fit_episodes(a):
  import multiprocessing as mp
  out = Path(a.out)
  z = load_frames(out)
  g = json.loads((out / "cam_global.json").read_text())
  p0 = np.array([g["camera"][k] for k in CAM_NAMES])
  eps = sorted(set(z["episode"][z["split"] == "train"].tolist()))
  if a.all_splits:
    eps = sorted(set(z["episode"].tolist()))
  jobs = []
  for e in eps:
    ix = np.where(z["episode"] == e)[0]
    jobs.append((z["state"][ix], [lowres(z["arm"][i], z["ignore"][i]) for i in ix], p0, int(e), 100, CAM_SIGMA * 3))
  t0 = time.time()
  with mp.get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(str(out),)) as pool:
    res = pool.map(_cma_job, jobs, chunksize=1)
  rec = {str(e): {"camera": np.asarray(r[0]).tolist(), "soft_iou": r[1]} for e, r in zip(eps, res)}
  (out / "cam_episodes.json").write_text(json.dumps(rec, indent=1))
  s = np.array([r[1] for r in res])
  print(f"per-episode fits: {len(eps)} episodes, soft IoU median {np.median(s):.3f} p10 {np.percentile(s, 10):.3f} "
        f"({time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------------------------- sessions

def cam_px_distance(p, q) -> float:
  """Max pixel distance (640x480) between where two cameras see the same table grid points."""
  return float(np.linalg.norm(project(p, TABLE_PTS) - project(q, TABLE_PTS), axis=1).max())


def cmd_sessions(a):
  out = Path(a.out)
  ce = json.loads((out / "cam_episodes.json").read_text())
  eps = sorted(int(e) for e in ce)
  cams = {e: np.array(ce[str(e)]["camera"]) for e in eps}
  score = {e: ce[str(e)]["soft_iou"] for e in eps}
  # a session grows while each new episode's camera is within SESSION_PX of the running median
  # camera of the session; an episode whose own fit is poor (soft IoU < 0.5) is attached to the
  # current session (its camera is unreliable, not evidence of a move).
  sessions, cur = [], [eps[0]]
  for e in eps[1:]:
    med = np.median([cams[x] for x in cur[-8:] if score[x] >= 0.5] or [cams[cur[-1]]], axis=0)
    if score[e] < 0.5 or cam_px_distance(cams[e], med) <= a.session_px:
      cur.append(e)
    else:
      # one-episode blips: if the NEXT episode agrees with the current session, treat e as an outlier
      sessions.append(cur); cur = [e]
  sessions.append(cur)
  # merge single-episode sessions into the neighbour they are closest to
  merged = []
  for s in sessions:
    if merged and len(s) == 1:
      merged[-1] = merged[-1] + s
    else:
      merged.append(s)
  sessions = merged
  from friendrig.realdata import split
  ho = split()["held_out"]
  rec = {"session_px": a.session_px, "sessions": []}
  for k, s in enumerate(sessions):
    lo, hi = s[0], s[-1]
    nxt = sessions[k + 1][0] if k + 1 < len(sessions) else 10**9
    held = [h for h in ho if lo <= h < nxt and (k > 0 or h >= 0)]
    if k == 0:
      held = [h for h in ho if h < nxt]
    rec["sessions"].append({"id": k, "train_episodes": s, "held_out": held, "first": lo, "last": hi,
                            "boundary_held_out": [h for h in held if h > hi]})
  (out / "sessions.json").write_text(json.dumps(rec, indent=1))
  d = [cam_px_distance(cams[e], cams[f]) for e, f in zip(eps[:-1], eps[1:])]
  print(f"consecutive-episode camera distance px: pct 25/50/75/90 {np.percentile(d, [25, 50, 75, 90]).round(1)}")
  print(f"{len(sessions)} sessions: " + ", ".join(f"{s[0]}-{s[-1]}({len(s)})" for s in sessions))


# ------------------------------------------------------------------------------------- per session

def _session_job(args):
  states, reals, p0, seed, iters = args
  sc = Scorer(_G["pa"], states, reals)
  return cma_fit(sc, p0, CAM_SIGMA, iters=iters, seed=seed)


def _session_offsets_obj(args):
  offs, items = args
  pa = _G["pa"]
  tot, n = 0.0, 0
  for states, reals, p in items:
    sc = Scorer(pa, states, reals, offsets=np.asarray(offs))
    s = sc.per_frame(p)
    tot += sum(s); n += len(s)
  return tot, n


def cmd_fit_sessions(a):
  import multiprocessing as mp
  out = Path(a.out)
  z = load_frames(out)
  ss = json.loads((out / "sessions.json").read_text())["sessions"]
  ce = json.loads((out / "cam_episodes.json").read_text())
  rng = np.random.default_rng(a.seed)
  items, p0s = [], []
  for s in ss:
    ix = np.where(np.isin(z["episode"], s["train_episodes"]) & (z["split"] == "train"))[0]
    if len(ix) > a.max_frames:
      ix = rng.choice(ix, a.max_frames, replace=False)
    good = [ce[str(e)]["camera"] for e in s["train_episodes"] if ce[str(e)]["soft_iou"] >= 0.5] or \
           [ce[str(e)]["camera"] for e in s["train_episodes"]]
    p0s.append(np.median(np.array(good), axis=0))
    items.append((z["state"][ix], [lowres(z["arm"][i], z["ignore"][i]) for i in ix]))
  offsets = np.array(a.init_offsets, float) if a.init_offsets else np.zeros(5)
  centre = offsets.copy()
  t0 = time.time()
  with mp.get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(str(out),)) as pool:
    if a.init_cams:  # start from an earlier session fit instead of the per-episode medians
      prev = json.loads((out / a.init_cams).read_text())["sessions"]
      p0s = [np.array([s_["camera"][k] for k in CAM_NAMES]) for s_ in prev]
    res = pool.map(_session_job_off, [(st, re, p0, a.seed + k, a.iters, offsets.tolist())
                                      for k, ((st, re), p0) in enumerate(zip(items, p0s))], chunksize=1)
    cams = [np.asarray(r[0]) for r in res]
    print(f"session cameras (offsets fixed at start values): soft IoU {[round(r[1], 3) for r in res]} ({time.time() - t0:.0f} s)", flush=True)
    log = []
    if a.offsets:
      import cma
      for rnd in range(a.rounds):
        # (a) shared per-joint offsets with cameras frozen (CMA-ES, sigma 3 deg, bound +-OFF_BOUND)
        es = cma.CMAEvolutionStrategy(offsets, 3.0, {"popsize": 10, "seed": a.seed + 100 + rnd, "verbose": -9,
                                                     "maxiter": a.offset_iters,
                                                     "bounds": [(centre - a.bound).tolist(), (centre + a.bound).tolist()]})
        sub_items = []
        for (st, re), p in zip(items, cams):
          k = min(len(st), a.offset_frames)
          sub_items.append((st[:k], re[:k], p))
        chunks = [sub_items[i::a.workers] for i in range(a.workers)]
        best = (offsets, -1.0)
        while not es.stop():
          xs = es.ask()
          fs = []
          for x in xs:
            r = pool.map(_session_offsets_obj, [(x, c) for c in chunks if c])
            v = sum(t for t, _ in r) / max(1, sum(n for _, n in r))
            fs.append(-v)
            if v > best[1]:
              best = (np.asarray(x), v)
          es.tell(xs, fs)
        offsets = best[0]
        print(f"round {rnd}: offsets {np.round(offsets, 2).tolist()} mean soft IoU {best[1]:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        # (b) cameras with offsets frozen
        global _OFFS
        res = pool.map(_session_job_off, [(st, re, p, a.seed + k, a.iters // 2, offsets.tolist())
                                          for k, ((st, re), p) in enumerate(zip(items, cams))], chunksize=1)
        cams = [np.asarray(r[0]) for r in res]
        log.append({"round": rnd, "offsets": offsets.tolist(), "offset_score": best[1],
                    "session_scores": [r[1] for r in res]})
        print(f"round {rnd}: cameras refit, soft IoU {[round(r[1], 3) for r in res]}", flush=True)
  tag = ("_offsets" if a.offsets else "") + (f"_{a.tag}" if a.tag else "")
  rec = {"offsets_deg_added_to_default": offsets.tolist(), "default_calib": asdict(ArmCalib()), "log": log,
         "sessions": [{**s, "camera": dict(zip(CAM_NAMES, c.tolist())), "n_fit_frames": len(it[0])}
                      for s, c, it in zip(ss, cams, items)]}
  (out / f"cam_sessions{tag}.json").write_text(json.dumps(rec, indent=1))
  print(f"wrote cam_sessions{tag}.json ({time.time() - t0:.0f} s)", flush=True)


def _session_job_off(args):
  states, reals, p0, seed, iters, offs = args
  sc = Scorer(_G["pa"], states, reals, offsets=np.asarray(offs))
  return cma_fit(sc, p0, CAM_SIGMA * 0.5, iters=iters, seed=seed)


# ------------------------------------------------------------------------------------- eval

def _eval_init():
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  _G["sim"] = SimArm()


def _eval_job(args):
  i, state, arm, ign, p, offs = args
  sim = _G["sim"]
  sim.set_camera(p)
  cal = ArmCalib(offset_deg=list(np.asarray(ArmCalib().offset_deg) + np.asarray(offs)))
  m = sim.mask(lerobot_to_sim(state, cal), 640, 480)
  return i, iou(m, arm, ign), iou(m, arm | ign), np.packbits(m)


def cmd_eval(a):
  import multiprocessing as mp
  out = Path(a.out)
  z = load_frames(out)
  cs = json.loads((out / a.cams).read_text())
  offs = np.asarray(cs.get("offsets_deg_added_to_default", [0] * 5))
  if a.no_offsets:
    offs = np.zeros(5)
  cam_of = {}
  for s in cs["sessions"]:
    p = np.array([s["camera"][k] for k in CAM_NAMES])
    for e in s["train_episodes"] + s["held_out"]:
      cam_of[e] = p
  if a.global_cam:
    g = json.loads((out / "cam_global.json").read_text())
    pg = np.array([g["camera"][k] for k in CAM_NAMES])
    cam_of = {e: pg for e in cam_of}
  if a.cable_px:  # re-split the dark mask with a different cable width (same frames)
    import cv2
    from friendrig.masks import disk
    for i in range(len(z["episode"])):
      dm = z["arm"][i] | z["ignore"][i]
      core = cv2.morphologyEx(dm.astype(np.uint8), cv2.MORPH_OPEN, disk(a.cable_px)).astype(bool)
      z["arm"][i], z["ignore"][i] = core, dm & ~core
  idx = [i for i in range(len(z["episode"])) if int(z["episode"][i]) in cam_of and
         (z["split"][i] == "held_out" or a.with_train)]
  jobs = [(i, z["state"][i], z["arm"][i], z["ignore"][i], cam_of[int(z["episode"][i])], offs) for i in idx]
  t0 = time.time()
  with mp.get_context("fork").Pool(a.workers, initializer=_eval_init) as pool:
    res = pool.map(_eval_job, jobs, chunksize=8)
  rows = []
  sim_masks = {}
  for i, u, u_raw, m in res:
    rows.append({"episode": int(z["episode"][i]), "frame_index": int(z["frame_index"][i]),
                 "split": str(z["split"][i]), "iou": u, "iou_raw_no_ignore": u_raw,
                 "speed_deg_s": float(z["speed"][i])})
    sim_masks[i] = m
  rec = {"cams": a.cams, "global_cam": bool(a.global_cam), "offsets_deg": offs.tolist(),
         "cable_px": a.cable_px or 15, "frames": rows}
  for sp in ("held_out", "train"):
    v = [r["iou"] for r in rows if r["split"] == sp]
    vr = [r["iou_raw_no_ignore"] for r in rows if r["split"] == sp]
    if v:
      per_ep = {}
      for r in rows:
        if r["split"] == sp:
          per_ep.setdefault(r["episode"], []).append(r["iou"])
      rec[sp] = {"n_frames": len(v), "n_episodes": len(per_ep), "mean_iou": float(np.mean(v)),
                 "median_iou": float(np.median(v)), "p10_iou": float(np.percentile(v, 10)),
                 "mean_iou_raw": float(np.mean(vr)), "median_iou_raw": float(np.median(vr)),
                 "frac_ge_0.8": float(np.mean(np.array(v) >= 0.8)),
                 "episode_mean_iou_median": float(np.median([np.mean(x) for x in per_ep.values()]))}
      print(f"{sp}: n={len(v)} frames / {len(per_ep)} episodes, IoU mean {np.mean(v):.3f} median {np.median(v):.3f} "
            f"p10 {np.percentile(v, 10):.3f}, raw (cable counted) mean {np.mean(vr):.3f}, >=0.8: "
            f"{np.mean(np.array(v) >= 0.8):.2f}", flush=True)
  tag = a.tag or Path(a.cams).stem
  (out / f"eval_{tag}.json").write_text(json.dumps(rec, indent=1))
  if a.overlays:
    _overlays(out, z, rows, sim_masks, idx, tag, a.overlays)
  print(f"eval done ({time.time() - t0:.0f} s)", flush=True)


def _overlays(out, z, rows, sim_masks, idx, tag, n):
  import cv2
  from friendrig.realdata import FriendData
  D = FriendData()
  od = out / f"overlays_{tag}"; od.mkdir(exist_ok=True)
  ho = [(r, i) for r, i in zip(rows, idx) if r["split"] == "held_out"]
  ho.sort(key=lambda t: t[0]["iou"])
  pick = [ho[int(j)] for j in np.linspace(0, len(ho) - 1, n).round().astype(int)]
  for r, i in pick:
    img = D.frames(r["episode"], "cam_head", [r["frame_index"]])[0].copy()
    sim_m = np.unpackbits(sim_masks[i])[:640 * 480].reshape(480, 640)
    for m, col in ((z["arm"][i], (0, 255, 0)), (sim_m, (255, 0, 255))):
      cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
      cv2.drawContours(img, cs, -1, col, 2)
    ign = z["ignore"][i]
    img[ign] = (0.5 * img[ign] + [0, 0, 127]).astype(np.uint8)
    cv2.putText(img, f"held-out ep{r['episode']} f{r['frame_index']} IoU {r['iou']:.2f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imwrite(str(od / f"ep{r['episode']:03d}_f{r['frame_index']:04d}_iou{r['iou']:.2f}.jpg"), img[..., ::-1])


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  for name in ("frames", "fit-global", "fit-episodes", "sessions", "fit-sessions", "eval"):
    p = sub.add_parser(name)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
  g = sub.choices["fit-global"]
  g.add_argument("--n-random", type=int, default=20000)
  g.add_argument("--n-frames-rand", type=int, default=60)
  g.add_argument("--n-frames-cma", type=int, default=200)
  g.add_argument("--n-starts", type=int, default=8)
  sub.choices["fit-episodes"].add_argument("--all-splits", action="store_true")
  sub.choices["sessions"].add_argument("--session-px", type=float, default=SESSION_PX)
  f = sub.choices["fit-sessions"]
  f.add_argument("--max-frames", type=int, default=300)
  f.add_argument("--iters", type=int, default=120)
  f.add_argument("--offsets", action="store_true")
  f.add_argument("--rounds", type=int, default=2)
  f.add_argument("--bound", type=float, default=20.0)
  f.add_argument("--offset-iters", type=int, default=40)
  f.add_argument("--offset-frames", type=int, default=60)
  f.add_argument("--init-offsets", type=float, nargs=5, default=None,
                 help="start (and centre of the +-bound box) for the offsets, deg added to the defaults")
  f.add_argument("--init-cams", default=None, help="session camera JSON to start from")
  f.add_argument("--tag", default="")
  e = sub.choices["eval"]
  e.add_argument("--cams", default="cam_sessions.json")
  e.add_argument("--tag", default="")
  e.add_argument("--with-train", action="store_true")
  e.add_argument("--global-cam", action="store_true")
  e.add_argument("--no-offsets", action="store_true")
  e.add_argument("--overlays", type=int, default=0)
  e.add_argument("--cable-px", type=int, default=0, help="re-split arm/ignore with this opening (0 = as sampled, 15)")
  a = ap.parse_args()
  {"frames": cmd_frames, "fit-global": cmd_fit_global, "fit-episodes": cmd_fit_episodes, "sessions": cmd_sessions,
   "fit-sessions": cmd_fit_sessions, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
  main()
