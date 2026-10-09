"""Measure SO-101 gripper and workspace numbers (pure CPU MuJoCo, no torch).

Writes measurements/so101_measurements.json and prints a summary. Run:

  python -I manip/task/scripts/measure_so101.py [--samples 200000]

What it measures
----------------
1. Gripper: pad-to-pad gap versus gripper joint angle (mj_geomDistance between the
   fixed-jaw and moving-jaw pad boxes), the angle at which the jaws meet, and the
   maximum opening.
2. Workspace: uniform samples of the five arm joints (gripper half open). A sample
   counts as a *valid pose* when (a) no robot self-contact, (b) the static gravity
   torque on every joint is below the weak-arm STS3215 limit (1.62 N*m), and, for a
   given mount height H, (c) every robot collision geom's lowest AABB corner is >= 2 mm
   above the table (z = -H in the base frame). For each valid pose we record the
   grasp-site position and the approach angle (angle between the finger axis and
   straight down).
3. For each candidate mount height H: the table area over which the grasp site can sit
   at an object-centre height band with the fingers within 15 / 30 / 45 degrees of
   vertical, and the lowest grasp-site height reachable with a near-vertical approach.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT))

from so101_pickplace import mj_scene  # noqa: E402
from so101_pickplace import robot_model as rm  # noqa: E402

OUT = TASK_ROOT / "measurements" / "so101_measurements.json"
MOUNT_CANDIDATES = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
APPROACH_LIMITS_DEG = (15.0, 30.0, 45.0, 90.0)
WEAK_TORQUE = rm.STS3215_TORQUE_7V4_6V


def gripper_gap_table(scene: mj_scene.CpuScene) -> dict:
  m, d = scene.model, scene.data
  qg = scene.qadr(rm.GRIPPER_JOINT)
  pairs = [("fixed_jaw_box4", "moving_jaw_box2"), ("fixed_jaw_box5", "moving_jaw_box3"),
           ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")]
  fromto = np.zeros(6)
  grip_b = scene.bid("gripper")
  mtip = scene.sid("moving_tip_site")
  rows = []
  for q in np.linspace(m.jnt_range[scene.jid(rm.GRIPPER_JOINT)][0],
                       m.jnt_range[scene.jid(rm.GRIPPER_JOINT)][1], 40):
    d.qpos[:] = 0.0
    d.qpos[qg] = q
    mujoco.mj_forward(m, d)
    gaps = [mujoco.mj_geomDistance(m, d, scene.gid(a), scene.gid(b), 0.5, fromto)
            for a, b in pairs]
    # mouth: distance along the closing axis (gripper +x) from the fixed pad face to
    # the moving-jaw tip; the widest object that can slide between open jaws.
    R = d.xmat[grip_b].reshape(3, 3)
    tip_local = R.T @ (d.site_xpos[mtip] - d.xpos[grip_b])
    rows.append({"q": round(float(q), 4),
                 "mouth_at_moving_tip_mm": round((tip_local[0] - rm.FIXED_JAW_INNER_X) * 1e3, 1),
                 "moving_tip_height_above_fixed_jaw_lowest_mm": round((tip_local[2] - rm.FIXED_JAW_LOWEST_Z) * 1e3, 1),
                 "gap_tip_pads_mm": round(gaps[0] * 1e3, 1),
                 "gap_mid_pads_mm": round(gaps[1] * 1e3, 1),
                 "gap_tip_spheres_mm": round(gaps[2] * 1e3, 1)})
  touch = [r["q"] for r in rows if min(r["gap_tip_pads_mm"], r["gap_mid_pads_mm"]) <= 0.5]
  return {
    "rows": rows,
    "jaws_touch_at_rad_approx": max(touch) if touch else None,
    "max_gap_tip_pads_mm": max(r["gap_tip_pads_mm"] for r in rows),
    "max_gap_mid_pads_mm": max(r["gap_mid_pads_mm"] for r in rows),
    "max_mouth_at_moving_tip_mm": max(r["mouth_at_moving_tip_mm"] for r in rows),
  }


def geom_local_corners(m: mujoco.MjModel, gids: np.ndarray) -> np.ndarray:
  """(G, 8, 3) local AABB corners of each geom (geom frame)."""
  out = np.zeros((len(gids), 8, 3))
  signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
  for i, g in enumerate(gids):
    c = m.geom_aabb[g, :3]
    h = m.geom_aabb[g, 3:]
    out[i] = c + signs * h
  return out


def sample_workspace(n: int, seed: int = 0) -> dict:
  # Robot alone in free space (no table), base at the origin. Table is applied later
  # per mount height by thresholding geom lowest points.
  scene = mj_scene.build_cpu_scene(mount_height=0.0, with_object=False)
  m, d = scene.model, scene.data
  table_gid = scene.gid("table")
  m.geom_contype[table_gid] = 0
  m.geom_conaffinity[table_gid] = 0
  rng = np.random.default_rng(seed)
  arm_adr = np.array([scene.qadr(j) for j in rm.ARM_JOINTS])
  arm_rng = np.array([m.jnt_range[scene.jid(j)] for j in rm.ARM_JOINTS])
  qg = scene.qadr(rm.GRIPPER_JOINT)
  site = scene.sid("grasp_site")
  grip_b = scene.bid("gripper")
  robot_gids = np.array([g for g in range(m.ngeom) if m.geom_contype[g] == rm.ROBOT_CONTYPE])
  corners = geom_local_corners(m, robot_gids)
  dof_adr = np.array([m.jnt_dofadr[scene.jid(j)] for j in rm.ALL_JOINTS])

  Q = np.zeros((n, 5))
  P = np.zeros((n, 3))
  A = np.zeros((n, 3))
  C = np.zeros((n, 3))
  ZMIN = np.zeros(n)
  ZMIN_NONFINGER = np.zeros(n)
  SELF = np.zeros(n, bool)
  GRAV = np.zeros(n)
  finger_mask = np.array([bool(__import__("re").match(rm.FINGER_GEOM_REGEX, m.geom(g).name))
                          for g in robot_gids])
  t0 = time.time()
  for i in range(n):
    q = rng.uniform(arm_rng[:, 0], arm_rng[:, 1])
    q[0] = 0.0  # pan: the arm is rotationally symmetric about the pan axis
    d.qpos[:] = 0.0
    d.qpos[arm_adr] = q
    d.qpos[qg] = 0.6
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    Q[i] = q
    P[i] = d.site_xpos[site]
    R = d.xmat[grip_b].reshape(3, 3)
    A[i] = -R[:, 2]
    C[i] = -R[:, 0]
    SELF[i] = any(c.dist < -0.001 for c in d.contact[: d.ncon])
    GRAV[i] = np.max(np.abs(d.qfrc_bias[dof_adr]))
    # lowest point of every robot collision geom (AABB corners, conservative)
    xm = d.geom_xmat[robot_gids].reshape(-1, 3, 3)
    world = np.einsum("gij,gkj->gki", xm, corners) + d.geom_xpos[robot_gids][:, None, :]
    zlow = world[..., 2].min(axis=1)
    ZMIN[i] = zlow.min()
    ZMIN_NONFINGER[i] = zlow[~finger_mask].min()
  dt = time.time() - t0
  pan_axis_xy = d.xpos[scene.bid("shoulder")][:2].copy()
  pan_range = m.jnt_range[scene.jid("shoulder_pan")].copy()
  return dict(pan_axis_xy=pan_axis_xy, pan_range=pan_range,Q=Q, P=P, A=A, C=C, ZMIN=ZMIN, ZMIN_NONFINGER=ZMIN_NONFINGER, SELF=SELF,
              GRAV=GRAV, seconds=dt, joint_ranges=arm_rng.tolist())


def analyse(ws: dict, object_center_band: tuple[float, float]) -> dict:
  P, A, ZMIN = ws["P"], ws["A"], ws["ZMIN"]
  ok_static = (~ws["SELF"]) & (ws["GRAV"] < WEAK_TORQUE)
  approach_deg = np.degrees(np.arccos(np.clip(-A[:, 2], -1.0, 1.0)))
  # radial distance of the grasp site from the (vertical) pan axis, signed: >0 in
  # front of the arm (pan = 0 direction), <0 reaching back over the base.
  front = np.array([1.0, 0.0])
  rel = P[:, :2] - ws["pan_axis_xy"]
  r = rel @ front
  lateral = np.abs(rel[:, 1])
  pan_span_rad = float(ws["pan_range"][1] - ws["pan_range"][0])
  res_pan = {"pan_axis_xy_base_frame_m": [round(float(v), 4) for v in ws["pan_axis_xy"]],
             "pan_range_rad": [round(float(v), 4) for v in ws["pan_range"]]}
  res: dict = {
    "valid_static_fraction": float(ok_static.mean()),
    "self_collision_fraction": float(ws["SELF"].mean()),
    "gravity_over_limit_fraction": float((ws["GRAV"] >= WEAK_TORQUE).mean()),
    **res_pan,
    "note": ("radii are measured from the pan axis along the arm plane; reachable "
             "table area = annulus sector over the pan range"),
    "per_mount": {},
  }
  # Lowest grasp-site height (base frame) per approach limit, ignoring any table.
  low = {}
  for lim in APPROACH_LIMITS_DEG:
    sel = ok_static & (approach_deg <= lim)
    if sel.any():
      j = np.argmin(P[sel, 2])
      low[f"<= {lim:.0f} deg"] = {
        "min_site_z_base_frame_m": round(float(P[sel, 2][j]), 4),
        "at_radius_m": round(float(r[sel][j]), 3),
        "n_poses": int(sel.sum()),
      }
  res["lowest_grasp_site_no_table"] = low

  bins = np.arange(0.05, 0.45, 0.025)
  for H in MOUNT_CANDIDATES:
    clear = ws["ZMIN"] >= -H + 0.002  # all robot geoms >= 2 mm above the table
    site_h = P[:, 2] + H  # grasp-site height above the table
    entry = {}
    for lim in APPROACH_LIMITS_DEG:
      sel = ok_static & clear & (approach_deg <= lim)
      band = sel & (site_h >= object_center_band[0]) & (site_h <= object_center_band[1])
      # lowest reachable site height above the table with this approach, per radius bin
      lowest_by_r = {}
      highest_by_r = {}  # wrist_flex runs out of range for high near-vertical poses
      for b0 in bins:
        s = sel & (r >= b0) & (r < b0 + 0.025) & (lateral < 0.01)
        if s.any():
          lowest_by_r[f"{b0:.3f}-{b0 + 0.025:.3f}"] = round(float(site_h[s].min()), 4)
          highest_by_r[f"{b0:.3f}-{b0 + 0.025:.3f}"] = round(float(site_h[s].max()), 4)
      band = band & (r > 0) & (lateral < 0.01)
      if band.any():
        # radial coverage in 1 cm rings, then annulus-sector area over the pan range
        rings = sorted({int(math.floor(x * 100)) for x in r[band]})
        area_cm2 = sum(0.5 * pan_span_rad * ((k + 1) ** 2 - k**2) for k in rings)
        entry[f"<= {lim:.0f} deg"] = {
          "n_poses_in_band": int(band.sum()),
          "reachable_rings_cm": [rings[0], rings[-1] + 1],
          "missing_rings_cm": [k for k in range(rings[0], rings[-1] + 1) if k not in rings],
          "reachable_area_cm2": round(area_cm2, 0),
          "lowest_site_height_above_table_by_radius_m": lowest_by_r,
          "highest_site_height_above_table_by_radius_m": highest_by_r,
        }
      else:
        entry[f"<= {lim:.0f} deg"] = {"n_poses_in_band": 0,
                                      "lowest_site_height_above_table_by_radius_m": lowest_by_r,
                                      "highest_site_height_above_table_by_radius_m": highest_by_r}
    res["per_mount"][f"{H:.2f}"] = entry
  return res


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--samples", type=int, default=200_000)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  scene = mj_scene.build_cpu_scene(mount_height=0.0, with_object=False)
  grip = gripper_gap_table(scene)
  m = scene.model
  print(f"[gripper] max mouth at tip {grip['max_mouth_at_moving_tip_mm']} mm; jaws touch at ~{grip['jaws_touch_at_rad_approx']} rad; "
        f"max gap tip pads {grip['max_gap_tip_pads_mm']} mm, mid pads {grip['max_gap_mid_pads_mm']} mm")

  ws = sample_workspace(args.samples, args.seed)
  print(f"[workspace] {args.samples} samples in {ws['seconds']:.1f} s")
  half = mj_scene.object_half_height()
  # grasp-site height band for the nominal 30 mm cube: centre .. centre + 10 mm
  band = (half + 0.0, half + 0.010)
  res = analyse(ws, band)
  out = {
    "model": "MuJoCo Menagerie robotstudio_so101 (see manip/third_party/menagerie_so101)",
    "grasp_site_gripper_frame_m": list(rm.GRASP_SITE_POS),
    "gripper": grip,
    "workspace_samples": args.samples,
    "object_center_band_above_table_m": band,
    "weak_arm_torque_limit_nm": WEAK_TORQUE,
    "workspace": res,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(out, indent=1))
  np.savez_compressed(OUT.with_suffix(".npz"), Q=ws["Q"], P=ws["P"], A=ws["A"],
                      ZMIN=ws["ZMIN"], SELF=ws["SELF"], GRAV=ws["GRAV"])
  print(json.dumps(res["lowest_grasp_site_no_table"], indent=1))
  for H, e in res["per_mount"].items():
    line = []
    for k, v in e.items():
      if v.get("n_poses_in_band", 0):
        line.append(f"{k}: r={v['reachable_rings_cm']}cm gaps={len(v['missing_rings_cm'])} A={v['reachable_area_cm2']:.0f}cm2")
      else:
        line.append(f"{k}: none")
    print(f"H={H}: " + " | ".join(line))
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
