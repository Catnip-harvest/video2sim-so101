"""Stage 3 (look): clean background plate per camera session, table orthophoto, compositing.

Why not Gaussian splatting: within one session cam_head does not move and the table is a plane,
so an empty-table image (the "plate") plus a plane-to-image homography already describes every
background pixel that camera can see.

1. Plate (`plate`). Per session, a per-pixel median over many cam_head frames from its TRAIN
   episodes, where each frame's arm (dark mask, dilated), green cube and white bowl/glare
   (dilated) pixels are left out. Pixels that were covered in (almost) every frame are filled by
   OpenCV inpainting (Telea) and counted.
2. Orthophoto (`plate`). The plate is resampled onto the table plane z = 0 in the robot base frame
   (1 mm per pixel) through the fitted camera: for every table point we look up the pixel it
   projects to. Table points the camera cannot see get the plate's median table colour.
   This image becomes the MuJoCo table texture (friendrig/scene.py places it at its true
   position and scale), so the WRIST camera sees the real table surface where it really is.
3. Compositing (`composite`). For cam_head renders: wherever the sim pixel is table/background
   (segmentation), take the plate pixel; elsewhere (arm, cube, bowl) keep the sim pixel.
   Plain-MuJoCo rendering stays available (switch in friendrig/scene.py), for an A/B.

All outputs are derived from the friend's frames: server and Drive only, never git.

  python -m friendrig.plate plate --out OUT --cams cam_sessions.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from friendrig.masks import dark_mask, green_mask, white_mask
from friendrig.simcam import CAM_NAMES, project

PER_EP = 6
MAX_FRAMES = 240
MIN_VALID = 3
ORTHO_RES = 0.001                         # m per orthophoto pixel
ORTHO_X = (-0.15, 0.60)                   # base-frame extent of the orthophoto (m)
ORTHO_Y = (-0.45, 0.45)


def exclusion(img: np.ndarray) -> np.ndarray:
  bad = cv2.dilate(dark_mask(img, 90).astype(np.uint8), np.ones((31, 31), np.uint8))
  ob = (green_mask(img) | white_mask(img)).astype(np.uint8)
  bad |= cv2.dilate(ob, np.ones((25, 25), np.uint8))
  return bad.astype(bool)


def build_plate(D, episodes, rng, per_ep=PER_EP, max_frames=MAX_FRAMES):
  picks = []
  for e in episodes:
    L = D.length(e)
    for k in np.linspace(0, L - 1, per_ep).round().astype(int):
      picks.append((e, int(k)))
  if len(picks) > max_frames:
    picks = [picks[i] for i in sorted(rng.choice(len(picks), max_frames, replace=False))]
  stack = np.full((len(picks), 480, 640, 3), np.nan, np.float32)
  by_ep = {}
  for j, (e, k) in enumerate(picks):
    by_ep.setdefault(e, []).append((j, k))
  for e, jk in by_ep.items():
    imgs = D.frames(e, "cam_head", [k for _, k in jk])
    for (j, _), img in zip(jk, imgs):
      f = img.astype(np.float32)
      f[exclusion(img)] = np.nan
      stack[j] = f
  valid = np.isfinite(stack[..., 0]).sum(0)
  plate = np.zeros((480, 640, 3), np.float32)
  for r0 in range(0, 480, 60):  # row chunks keep nanmedian memory low
    plate[r0:r0 + 60] = np.nanmedian(stack[:, r0:r0 + 60], axis=0)
  hole = valid < MIN_VALID
  plate = np.nan_to_num(plate, nan=0.0).clip(0, 255).astype(np.uint8)
  if hole.any():
    plate = cv2.inpaint(plate, hole.astype(np.uint8), 7, cv2.INPAINT_TELEA)
  return plate, {"n_frames": len(picks), "n_episodes": len(by_ep), "hole_frac": float(hole.mean()),
                 "valid_median": float(np.median(valid))}


def orthophoto(plate: np.ndarray, cam, res=ORTHO_RES, xr=ORTHO_X, yr=ORTHO_Y):
  """Image rows = base x from xr[1] (top) down to xr[0]; columns = base y from yr[1] (left) to yr[0]
  (i.e. a top-down view with the robot's forward direction up). Returns (rgb, seen-mask)."""
  xs = np.arange(xr[1], xr[0], -res)
  ys = np.arange(yr[1], yr[0], -res)
  X, Y = np.meshgrid(xs, ys, indexing="ij")
  pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
  uv = project(cam, pts).reshape(X.shape + (2,)).astype(np.float32)
  seen = (uv[..., 0] >= 0) & (uv[..., 0] <= 639) & (uv[..., 1] >= 0) & (uv[..., 1] <= 479)
  out = cv2.remap(plate, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
  fill = np.median(plate.reshape(-1, 3), 0).astype(np.uint8)
  out[~seen] = fill
  return out, seen


def composite(sim_rgb: np.ndarray, sim_is_fg: np.ndarray, plate: np.ndarray) -> np.ndarray:
  """sim_is_fg: bool [H,W], True where the sim pixel is robot/object (not table/sky)."""
  out = plate.copy()
  out[sim_is_fg] = sim_rgb[sim_is_fg]
  return out


def _plate_job(args):
  sid, eps, cam, outdir, seed = args
  from friendrig.realdata import FriendData
  D = FriendData()
  rng = np.random.default_rng(seed)
  plate, info = build_plate(D, eps, rng)
  ortho, seen = orthophoto(plate, cam)
  cv2.imwrite(str(Path(outdir) / f"plate_s{sid:02d}.png"), plate[..., ::-1])
  cv2.imwrite(str(Path(outdir) / f"ortho_s{sid:02d}.png"), ortho[..., ::-1])
  info.update({"session": sid, "ortho_seen_frac": float(seen.mean()), "plate": f"plate_s{sid:02d}.png",
               "ortho": f"ortho_s{sid:02d}.png"})
  return info


def cmd_plate(a):
  import multiprocessing as mp
  out = Path(a.out)
  pd = out / "plates"; pd.mkdir(exist_ok=True)
  cs = json.loads((out / a.cams).read_text())
  jobs = []
  for s in cs["sessions"]:
    cam = [s["camera"][k] for k in CAM_NAMES]
    jobs.append((s["id"], s["train_episodes"], cam, str(pd), a.seed + s["id"]))
  t0 = time.time()
  with mp.get_context("fork").Pool(min(a.workers, 4)) as pool:  # each job holds a ~0.9 GB stack
    res = pool.map(_plate_job, jobs, chunksize=1)
  rec = {"method": __doc__, "cams": a.cams, "ortho_res_m": ORTHO_RES, "ortho_x_m": ORTHO_X, "ortho_y_m": ORTHO_Y,
         "sessions": res}
  (pd / "plates.json").write_text(json.dumps(rec, indent=1))
  for r in res:
    print(f"session {r['session']}: plate from {r['n_frames']} frames / {r['n_episodes']} episodes, "
          f"holes {r['hole_frac']:.3%}, ortho seen {r['ortho_seen_frac']:.1%}", flush=True)
  print(f"plates done ({time.time() - t0:.0f} s)", flush=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  p = sub.add_parser("plate")
  p.add_argument("--out", required=True)
  p.add_argument("--cams", default="cam_sessions.json")
  p.add_argument("--workers", type=int, default=8)
  p.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()
  {"plate": cmd_plate}[a.cmd](a)


if __name__ == "__main__":
  main()
