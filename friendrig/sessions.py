"""Find camera sessions: runs of consecutive episodes in which cam_head did not move.

Observation (13 Oct, montage of first frames every 22 episodes): cam_head is NOT fixed across the
whole dataset; the table grain lines and the rest-pose arm move in the image between recording
sessions. Inside a session it is fixed. So the overhead camera is calibrated per session.

Method: per episode, a background image = per-pixel median of N frames spread over the episode
(the arm moves, so the median is mostly table). Keep only table pixels (not dark, not green, not
white), band-pass the grey image (difference of Gaussians, so only the grain lines and table
marks remain) and compare two episodes with normalised cross-correlation over pixels valid in
both. Same camera -> NCC high; a camera moved by a few pixels -> NCC drops sharply (the lines
are thin). A new session starts where NCC with the previous episode falls below THRESH.

  python -m friendrig.sessions --out /work/outputs_large/friendrig/sessions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from friendrig.masks import dark_mask, green_mask, white_mask
from friendrig.realdata import FriendData

W, H = 320, 240
THRESH = 4.0  # px: max image-corner motion between two episodes' table homography


def background(D: FriendData, ep: int, n: int = 9) -> np.ndarray:
  L = D.length(ep)
  idx = np.linspace(0, L - 1, n).round().astype(int)
  fr = np.stack(D.frames(ep, "cam_head", idx))
  return np.median(fr, axis=0).astype(np.uint8)


_SIFT = None


def descriptor(bg: np.ndarray):
  """SIFT keypoints on table pixels only (objects, arm, bowl and glare masked out)."""
  global _SIFT
  _SIFT = _SIFT or cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01)
  g = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
  bad = dark_mask(bg) | green_mask(bg) | white_mask(bg) | (g < 100)
  bad = cv2.dilate(bad.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
  kp, des = _SIFT.detectAndCompute(clahe, (~bad).astype(np.uint8) * 255)
  return np.float32([k.pt for k in kp]), des


CORNERS = np.float32([[0, 0], [639, 0], [639, 479], [0, 479], [320, 240]]).reshape(-1, 1, 2)


def displacement(a, b) -> tuple[float, int]:
  """Homography a->b from ratio-test SIFT matches + RANSAC; returns (max corner motion px, inliers).
  inf when fewer than 15 inliers (cannot tell)."""
  (pa, da), (pb, db) = a, b
  if da is None or db is None or len(da) < 20 or len(db) < 20:
    return float("inf"), 0
  m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
  good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
  if len(good) < 15:
    return float("inf"), len(good)
  src = pa[[g.queryIdx for g in good]].reshape(-1, 1, 2)
  dst = pb[[g.trainIdx for g in good]].reshape(-1, 1, 2)
  Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
  n = int(inl.sum()) if inl is not None else 0
  if Hm is None or n < 15:
    return float("inf"), n
  moved = cv2.perspectiveTransform(CORNERS, Hm) - CORNERS
  return float(np.linalg.norm(moved.reshape(-1, 2), axis=1).max()), n


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--thresh", type=float, default=THRESH)
  a = ap.parse_args()
  out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
  D = FriendData()
  eps = D.episodes
  cache = out / "backgrounds.npz"
  if cache.exists():
    bgs = dict(np.load(cache))
    bgs = {int(k[2:]): v for k, v in bgs.items()}
  else:
    bgs = {}
    for e in eps:
      bgs[e] = background(D, e)
    np.savez_compressed(cache, **{f"ep{e}": v for e, v in bgs.items()})
  desc = {e: descriptor(bgs[e]) for e in eps}
  prev = [(float("nan"), 0)] + [displacement(desc[eps[i - 1]], desc[eps[i]]) for i in range(1, len(eps))]
  sessions, cur = [], [eps[0]]
  for i in range(1, len(eps)):
    d, n = prev[i]
    if not np.isfinite(d):  # too few matches with the previous one: try the session's first episode
      d, n = displacement(desc[cur[0]], desc[eps[i]])
    if d <= a.thresh:
      cur.append(eps[i])
    else:
      sessions.append(cur); cur = [eps[i]]
  sessions.append(cur)
  # merge back sessions that look like an earlier one (camera returned to the same place)
  rep = {}
  for s in sessions:
    rep[s[0]] = s
  res = {"method": __doc__, "thresh": a.thresh, "disp_prev_px_inliers": dict(zip(map(str, eps), prev)),
         "sessions": [{"first": s[0], "last": s[-1], "n": len(s), "episodes": s} for s in sessions]}
  (out / "sessions.json").write_text(json.dumps(res, indent=1))
  pv = np.array([p[0] for p in prev[1:]])
  print("corner motion vs previous episode (px): pct 5/25/50/75/95", np.percentile(pv[np.isfinite(pv)], [5, 25, 50, 75, 95]).round(2),
        "n inf", int((~np.isfinite(pv)).sum()), "inliers median", np.median([p[1] for p in prev[1:]]))
  print(f"{len(sessions)} sessions:", [(s[0], s[-1], len(s)) for s in sessions])
  # montage of each session's first background
  t = []
  for s in sessions[:48]:
    im = cv2.resize(bgs[s[0]], (240, 180)).copy()
    cv2.putText(im, f"{s[0]}-{s[-1]}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    t.append(im)
  while len(t) % 8:
    t.append(np.zeros_like(t[0]))
  g = np.concatenate([np.concatenate(t[i:i + 8], 1) for i in range(0, len(t), 8)], 0)
  cv2.imwrite(str(out / "sessions_montage.jpg"), g[..., ::-1])


if __name__ == "__main__":
  main()
