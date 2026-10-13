"""Stages 3-4: measure the green cube and the white bowl, and where they start, from cam_head.

With the fitted session camera, a known 3D shape projects to a known silhouette. So instead of
back-projecting blobs (which smears side faces onto the table), we fit the shape:
  cube : centre (x, y) on the table, yaw, side s           -> convex hull of its 8 corners
  bowl : centre (x, y), rim radius rt, height h, base radius rb (a frustum, which is convex)
         -> convex hull of its rim and base circles
and maximise IoU with the colour mask (green / white) in the FIRST cam_head frame of each
episode, ignoring pixels covered by the dark arm. Step 1 fits everything per episode and takes
the median size over episodes; step 2 freezes the sizes and refits only positions (and cube yaw):
that is the placement table.

Episodes used: TRAIN episodes whose task is "green cube(s) from the table into a white bowl/cup"
(regex below; bowl-to-bowl, plate and "out of the bowl" tasks are excluded), with exactly one
green blob and one white blob in the first frame. Everything else is counted as "not detected"
with its reason.

  python -m friendrig.objects fit --out OUT --cams cam_sessions.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from friendrig.masks import dark_mask, green_mask, white_mask
from friendrig.simcam import CAM_NAMES, backproject_to_plane, project

TARGET = re.compile(r"(bowl|cup|container)", re.I)
EXCLUDE = re.compile(r"(from (one|a|the|this|white|small)?\s*(white )?(bowl|bowls|plate|plates|cup|cups|container|conveyor|belt|"
                     r"black|bin|pyramid|floor|dish)|other|another|second|plate|bin|conveyor|belt|pyramid|rotating|"
                     r"black object|bowl stack|vertical stack|between|out of|remove|extract|larger|one table)", re.I)
MIN_GREEN, MIN_WHITE = 400, 4000


def is_table_to_bowl(task: str) -> bool:
  return bool(TARGET.search(task)) and not EXCLUDE.search(task)


def bowl_mask(img: np.ndarray) -> np.ndarray:
  """White bowl. Seed: the largest blob of bright unsaturated pixels (masks.white_mask = lit inner
  wall). Grow: add pixels within GROW_PX of the seed that are brighter than the table by >= 8
  (HSV V vs the image median) and unsaturated (the shaded wall and rim). Convex hull filled.
  (Growing without the distance limit merged the bowl with table glare: tried 13 Oct, rejected.)"""
  b = blobs(white_mask(img), MIN_WHITE)
  out = np.zeros(img.shape[:2], np.uint8)
  if not b:
    return out.astype(bool)
  seed = max(b, key=lambda x: x[1])[0].astype(np.uint8)
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
  near = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW_PX + 1, 2 * GROW_PX + 1))).astype(bool)
  grow = near & (v >= np.median(v) + 8) & (s <= 45)
  m = cv2.morphologyEx((grow | seed.astype(bool)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  bb = blobs(m, MIN_WHITE)
  big = max(bb, key=lambda x: x[1])[0] if bb else seed.astype(bool)
  cv2.fillConvexPoly(out, cv2.convexHull(cv2.findNonZero(big.astype(np.uint8))), 1)
  return out.astype(bool)


GROW_PX = 40


def blobs(m: np.ndarray, min_px: int):
  n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
  return [(lab == i, st[i, cv2.CC_STAT_AREA], cen[i]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]


def cube_pts(x, y, yaw, s):
  c = np.array([[dx, dy, dz] for dx in (-.5, .5) for dy in (-.5, .5) for dz in (0, 1)]) * s
  R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
  return c @ R.T + [x, y, 0]


def bowl_pts(x, y, rt, h, rb, n=48):
  a = np.linspace(0, 2 * np.pi, n, endpoint=False)
  rim = np.stack([x + rt * np.cos(a), y + rt * np.sin(a), np.full(n, h)], 1)
  base = np.stack([x + rb * np.cos(a), y + rb * np.sin(a), np.zeros(n)], 1)
  return np.vstack([rim, base])


def hull_mask(cam, pts, w=640, h=480, scale=1.0):
  uv = project(cam, pts) * scale
  m = np.zeros((int(h * scale), int(w * scale)), np.uint8)
  if not np.all(np.isfinite(uv)):
    return m.astype(bool)
  hull = cv2.convexHull(np.round(uv).astype(np.int32))
  cv2.fillConvexPoly(m, hull, 1)
  return m.astype(bool)


def masked_iou(a, b, keep):
  a, b = a & keep, b & keep
  u = (a | b).sum()
  return float((a & b).sum() / u) if u else 0.0


S = 0.5  # fit at 320x240


def fit_cube(cam, g, keep, init_xy, size=None):
  from scipy.optimize import minimize
  gs, ks = cv2.resize(g.astype(np.uint8), None, fx=S, fy=S).astype(bool), cv2.resize(keep.astype(np.uint8), None, fx=S, fy=S).astype(bool)
  best = (None, -1)
  for yaw0 in np.deg2rad([0, 22.5, 45, 67.5]):
    x0 = [init_xy[0], init_xy[1], yaw0] + ([] if size else [0.04])

    def f(x):
      s = size or x[3]
      if not 0.015 < s < 0.09:
        return 1.0
      return -masked_iou(hull_mask(cam, cube_pts(x[0], x[1], x[2], s), scale=S), gs, ks)
    r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400,
                                                         "initial_simplex": np.vstack([x0, np.array(x0) + np.diag([0.01, 0.01, 0.3] + ([] if size else [0.008]))])})
    if -r.fun > best[1]:
      best = (r.x, -r.fun)
  x = best[0]
  s = size or x[3]
  yaw = float(np.mod(x[2], np.pi / 2))
  return {"x": float(x[0]), "y": float(x[1]), "yaw": yaw, "size": float(s), "iou": best[1]}


def fit_bowl(cam, wm, keep, init_xy, dims=None):
  from scipy.optimize import minimize
  ws, ks = cv2.resize(wm.astype(np.uint8), None, fx=S, fy=S).astype(bool), cv2.resize(keep.astype(np.uint8), None, fx=S, fy=S).astype(bool)
  x0 = [init_xy[0], init_xy[1]] + ([] if dims else [0.08, 0.06, 0.045])

  def f(x):
    rt, h, rb = dims or x[2:5]
    if not (0.03 < rt < 0.2 and 0.01 < h < 0.15 and 0.015 < rb <= rt):
      return 1.0
    return -masked_iou(hull_mask(cam, bowl_pts(x[0], x[1], rt, h, rb), scale=S), ws, ks)
  r = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})
  rt, h, rb = dims or r.x[2:5]
  return {"x": float(r.x[0]), "y": float(r.x[1]), "rim_radius": float(rt), "height": float(h), "base_radius": float(rb),
          "iou": float(-r.fun)}


_G: dict = {}


def _job(args):
  ep, cam, dims = args
  from friendrig.realdata import FriendData
  D = _G.setdefault("D", FriendData())
  img = D.frames(ep, "cam_head", [0])[0]
  arm = dark_mask(img)
  keep = ~cv2.dilate(arm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
  gb, wb = blobs(green_mask(img), MIN_GREEN), blobs(bowl_mask(img), MIN_WHITE)
  rec = {"episode": ep, "n_green_blobs": len(gb), "n_white_blobs": len(wb)}
  if len(gb) != 1:
    rec["skip"] = f"{len(gb)} green blobs"
    return rec
  if len(wb) < 1:
    rec["skip"] = "no white blob"
    return rec
  wb = max(wb, key=lambda b: b[1])
  g, w = gb[0][0], wb[0]
  # touching the image border -> partly out of view; keep but flag
  rec["green_touches_border"] = bool(g[0].any() or g[-1].any() or g[:, 0].any() or g[:, -1].any())
  rec["white_touches_border"] = bool(w[0].any() or w[-1].any() or w[:, 0].any() or w[:, -1].any())
  gxy = backproject_to_plane(cam, gb[0][2][None], z0=0.02)[0, :2]
  wxy = backproject_to_plane(cam, wb[2][None], z0=0.03)[0, :2]
  # green mask may include the cube's shadow-free faces only; white blob may include glare far away
  rec["cube"] = fit_cube(cam, g, keep, gxy, size=dims and dims["cube"])
  rec["bowl"] = fit_bowl(cam, w, keep, wxy, dims=dims and (dims["rim_radius"], dims["height"], dims["base_radius"]))
  return rec


def cmd_fit(a):
  import multiprocessing as mp
  from friendrig.realdata import FriendData, split
  out = Path(a.out)
  cs = json.loads((out / a.cams).read_text())
  cam_of = {}
  for s in cs["sessions"]:
    p = [s["camera"][k] for k in CAM_NAMES]
    for e in s["train_episodes"] + s["held_out"]:
      cam_of[e] = p
  D = FriendData()
  tr = split()["train"]
  tasks = {e: D.task(e) for e in D.episodes}
  sel = [e for e in tr if is_table_to_bowl(tasks[e]) and e in cam_of]
  t0 = time.time()
  with mp.get_context("fork").Pool(a.workers) as pool:
    r1 = pool.map(_job, [(e, cam_of[e], None) for e in sel], chunksize=2)
    ok = [r for r in r1 if "cube" in r and r["cube"]["iou"] > 0.6 and r["bowl"]["iou"] > 0.6
          and not r["green_touches_border"] and not r["white_touches_border"]]
    dims = {"cube": float(np.median([r["cube"]["size"] for r in ok])),
            "rim_radius": float(np.median([r["bowl"]["rim_radius"] for r in ok])),
            "height": float(np.median([r["bowl"]["height"] for r in ok])),
            "base_radius": float(np.median([r["bowl"]["base_radius"] for r in ok]))}
    spread = {k: [float(np.percentile([r[o][f] for r in ok], q)) for q in (10, 90)]
              for k, o, f in (("cube", "cube", "size"), ("rim_radius", "bowl", "rim_radius"),
                              ("height", "bowl", "height"), ("base_radius", "bowl", "base_radius"))}
    print(f"step 1: {len(ok)} clean episodes; median dims {dims}; p10/p90 {spread}", flush=True)
    r2 = pool.map(_job, [(e, cam_of[e], dims) for e in sel], chunksize=2)
  det = [r for r in r2 if "cube" in r and r["cube"]["iou"] > 0.5 and r["bowl"]["iou"] > 0.5]
  reasons = {}
  for r in r2:
    if r not in det:
      k = r.get("skip") or "low fit IoU"
      reasons[k] = reasons.get(k, 0) + 1
  task_counts = {}
  for e in tr:
    k = "table_to_bowl" if is_table_to_bowl(tasks[e]) else "other"
    task_counts[k] = task_counts.get(k, 0) + 1
  rec = {"method": __doc__, "cams": a.cams, "n_train": len(tr), "task_counts_train": task_counts,
         "n_selected": len(sel), "n_detected": len(det), "detection_rate": len(det) / max(1, len(sel)),
         "not_detected_reasons": reasons, "dims_m": dims, "dims_p10_p90_m": spread,
         "n_clean_for_dims": len(ok), "episodes": r2}
  (out / "objects.json").write_text(json.dumps(rec, indent=1))
  # per-episode placement table (server only; derived from the friend's data)
  import csv
  with open(out / "placements_train.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["episode", "task", "cube_x", "cube_y", "cube_yaw", "cube_iou", "bowl_x", "bowl_y", "bowl_iou"])
    for r in det:
      wr.writerow([r["episode"], tasks[r["episode"]], r["cube"]["x"], r["cube"]["y"], r["cube"]["yaw"], r["cube"]["iou"],
                   r["bowl"]["x"], r["bowl"]["y"], r["bowl"]["iou"]])
  C = np.array([[r["cube"]["x"], r["cube"]["y"]] for r in det])
  B = np.array([[r["bowl"]["x"], r["bowl"]["y"]] for r in det])
  summ = {"cube_xy_mean": C.mean(0).tolist(), "cube_xy_std": C.std(0).tolist(),
          "cube_xy_p5": np.percentile(C, 5, 0).tolist(), "cube_xy_p95": np.percentile(C, 95, 0).tolist(),
          "bowl_xy_mean": B.mean(0).tolist(), "bowl_xy_std": B.std(0).tolist(),
          "bowl_xy_p5": np.percentile(B, 5, 0).tolist(), "bowl_xy_p95": np.percentile(B, 95, 0).tolist(),
          "cube_bowl_dist_median": float(np.median(np.linalg.norm(C - B, axis=1))),
          "cube_bowl_dist_p5_p95": np.percentile(np.linalg.norm(C - B, axis=1), [5, 95]).tolist()}
  rec["summary"] = summ
  (out / "objects.json").write_text(json.dumps(rec, indent=1))
  print(f"selected {len(sel)} table->bowl train episodes, detected {len(det)} ({len(det) / max(1, len(sel)):.2%}); "
        f"reasons {reasons}", flush=True)
  print(json.dumps(summ, indent=1), flush=True)
  _plot(out, C, B)
  print(f"done ({time.time() - t0:.0f} s)", flush=True)


def _plot(out, C, B):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(C[:, 1], C[:, 0], s=10, c="tab:green", label=f"cube start (n={len(C)})")
  ax.scatter(B[:, 1], B[:, 0], s=10, c="tab:gray", label=f"bowl (n={len(B)})")
  ax.scatter([0], [0], marker="s", c="k", label="robot base origin")
  ax.set_xlabel("y (m, robot left +)"); ax.set_ylabel("x (m, robot forward +)")
  ax.invert_xaxis(); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
  ax.set_title("Real start positions, robot base frame (train episodes)")
  fig.tight_layout(); fig.savefig(out / "placements_train.png", dpi=120)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  f = sub.add_parser("fit")
  f.add_argument("--out", required=True)
  f.add_argument("--cams", default="cam_sessions.json")
  f.add_argument("--workers", type=int, default=8)
  a = ap.parse_args()
  {"fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
  main()
