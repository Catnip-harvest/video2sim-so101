"""Reachability sweep behind the scattered-stacking zone (stack.ZONE_R) and the expert's
reach-ceiling table (stack_expert.CEILING_MM).

  python -m r2s.stack_reach --out sweep.json [--workers 6]      (from datagen/, ~2 min)

For each distance r from the shoulder-pan axis, finger tilt (outward, in the arm's vertical
plane) and closing direction (0 / 30 / 45 deg away from perpendicular-to-the-arm, i.e. the
cube yaw), it finds the lowest and highest grasp-site height the expert's IK reaches:
position error <= 3 mm, finger axis <= 5 deg, joints inside 95 % of their range, no arm-arm or
arm-table contact; 10 random restarts per point, 2.5 mm height steps. The azimuth is fixed
at 30 deg (an earlier sweep found no azimuth effect within +-60 deg; the test checks the
zone corners at -60 / 0 / +60 deg).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time

import mujoco
import numpy as np

from . import scene as S
from .paths import ik, load_task_spec, rm
from .variations import PAN_AXIS, sample_variation

POS_TOL = 0.003
AXIS_TOL_DEG = 5.0
RANGE_FRAC = 0.95
RESTARTS = 10
_G: dict = {}


def init(spec_path: str) -> None:
  """Build the arm alone (no bin, the cube far away) with soft joint limits."""
  spec = load_task_spec(spec_path)
  v = sample_variation(spec, 0, 0)
  v.extras = {"no_bin": True}
  v.obj_xy = [0.6, 0.6]
  sc = S.build_episode_scene(v)
  m = copy.deepcopy(sc.model)
  jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in rm.ARM_JOINTS]
  lo, hi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
  mid, half = (lo + hi) / 2, (hi - lo) / 2
  m.jnt_range[jids, 0] = mid - RANGE_FRAC * half
  m.jnt_range[jids, 1] = mid + RANGE_FRAC * half
  _G.update(m=m, sc=sc, jids=jids, d=mujoco.MjData(sc.model))


def _collides(q, g: float = 0.3) -> bool:
  sc, d = _G["sc"], _G["d"]
  d.qpos[:] = sc.model.qpos0
  d.qpos[sc.arm_qadr] = q
  d.qpos[sc.g_qadr] = g
  mujoco.mj_fwdPosition(sc.model, d)
  n = d.ncon
  if not n:
    return False
  gg = d.contact.geom[:n]
  c1, c2 = sc.geom_cat[gg[:, 0]], sc.geom_cat[gg[:, 1]]
  robot = (5, 6, 7)
  r1, r2 = np.isin(c1, robot), np.isin(c2, robot)
  return bool(np.any((r1 & r2) | (r1 & (c2 == 1)) | (r2 & (c1 == 1))))


def feasible(xy, z: float, tilt_deg: float, closing_off_deg: float,
             rng: np.random.Generator) -> bool:
  """Can the grasp site reach (xy, z) with this finger tilt and closing direction?"""
  m, jids = _G["m"], _G["jids"]
  az = math.atan2(xy[1] - PAN_AXIS[1], xy[0] - PAN_AXIS[0])
  u = np.array([math.cos(az), math.sin(az), 0.0])
  t = np.array([-math.sin(az), math.cos(az), 0.0])
  th = math.radians(tilt_deg)
  app = np.array([0.0, 0.0, -math.cos(th)]) + u * math.sin(th)
  ph = math.radians(closing_off_deg)
  clo = t * math.cos(ph) + u * math.sin(ph)
  lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
  for _ in range(RESTARTS):
    q0 = rng.uniform(lo * 0.8, hi * 0.8)
    r = ik.solve(m, q0, np.array([xy[0], xy[1], z]), approach=app, closing=clo, iters=200)
    if r.pos_err <= POS_TOL and r.approach_err_deg <= AXIS_TOL_DEG and not _collides(r.q):
      return True
  return False


def zone_xy(r: float, az_deg: float) -> np.ndarray:
  return PAN_AXIS + r * np.array([math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))])


def job(args) -> dict:
  r, az_deg, tilt, coff = args
  rng = np.random.default_rng([int(round(r * 1000)), int(az_deg) + 90, int(tilt), int(coff)])
  xy = zone_xy(r, az_deg)
  zs = np.round(np.arange(0.015, 0.1501, 0.0025), 4)
  ok = [float(z) for z in zs if feasible(xy, float(z), tilt, coff, rng)]
  return {"r": r, "az": az_deg, "tilt": tilt, "closing_off": coff,
          "z_min": min(ok) if ok else None, "z_max": max(ok) if ok else None}


def main() -> None:
  from multiprocessing import Pool
  from pathlib import Path
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  default_spec = (Path(__file__).resolve().parents[2] / "demo" / "outputs" /
                  "pexels_7488188_sugar_cube" / "so101_task_spec.json")
  ap.add_argument("--spec", default=str(default_spec))
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  t0 = time.perf_counter()
  rs = [round(float(x), 2) for x in np.arange(0.11, 0.281, 0.01)]
  jobs = [(r, 30, tilt, coff) for r in rs for tilt in (0, 10, 15, 20) for coff in (0, 30, 45)]
  with Pool(a.workers, initializer=init, initargs=(a.spec,)) as p:
    rows = p.map(job, jobs, chunksize=1)
  json.dump(rows, open(a.out, "w"), indent=0)
  table = {}
  for row in rows:  # worst case over closing directions
    k = (row["tilt"], row["r"])
    table[k] = min(table.get(k, 1.0), row["z_max"] or 0.0)
  for tilt in (0, 10, 15, 20):
    print(f"tilt {tilt:>2}: " + " ".join(f"{table[(tilt, r)] * 1e3:.0f}" for r in rs))
  print(f"r: {rs}  ({len(rows)} rows, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
  main()
