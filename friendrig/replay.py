"""Stage 5: twin replay (KINEMATIC, a visual check of geometry and cameras - not a physics claim).

For a held-out episode: the recorded joint angles (observation.state, converted with
calib/so101_units + the stage-1 offsets) are written straight into the sim arm every frame
(no physics). The cube and bowl start where stage 3/4 measured them in the first cam_head frame.
The cube is SCRIPTED: it stays put until the real gripper closes on it, then rides rigidly with
the gripper, and after the gripper opens it is placed where it was measured in the last frame.
Each output video is real | sim side by side, for cam_head (sim composited over the real plate,
or plain MuJoCo with --plain) and for cam_right (sim = wrist camera, table = real orthophoto).

  python -m friendrig.replay --profile PROFILE.json --episodes 41 128 296 --out OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "datagen", ROOT / "task"):
  if str(d) not in sys.path:
    sys.path.insert(0, str(d))


def grasp_window(g: np.ndarray, near: np.ndarray):
  """g: gripper % per frame; near: bool per frame (jaw within 4 cm of the cube's start).
  Grasp = first frame after a closing stroke (>= 8 % drop) that ends while near the cube;
  release = first later frame where the gripper has opened >= 8 % above its holding value."""
  n = len(g)
  for k in range(5, n - 5):
    if near[k] and g[k - 5] - g[k] >= 8 and abs(g[k + 5] - g[k]) < 2:
      hold = g[k]
      for r in range(k + 5, n):
        if g[r] - hold >= 8:
          return k, r
      return k, None
  return None, None


def main():
  import cv2
  import mujoco
  from calib.so101_units import ArmCalib, lerobot_to_sim
  from friendrig import objects as fo
  from friendrig import profile as fp
  from friendrig.realdata import FriendData
  from friendrig.simcam import JOINTS
  from r2s.paths import load_task_spec
  from r2s.render import _Encoder
  from r2s.scene import build_episode_scene
  from r2s.variations import sample_variation

  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--profile", required=True)
  ap.add_argument("--spec", default=str(ROOT / "task/specs/friend_bowl_task_spec.json"))
  ap.add_argument("--episodes", type=int, nargs="+", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--plain", action="store_true", help="cam_head: plain MuJoCo instead of composite")
  ap.add_argument("--step", type=int, default=1, help="use every k-th frame")
  ap.add_argument("--max-frames", type=int, default=100000)
  a = ap.parse_args()

  prof = fp.load_profile(a.profile)
  out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
  D = FriendData()
  spec = load_task_spec(a.spec)
  gm = prof.get("gripper_map_deg_at_0_100") or [ArmCalib().gripper_deg_at_0, ArmCalib().gripper_deg_at_100]
  cal = ArmCalib(offset_deg=list(np.asarray(ArmCalib().offset_deg) + np.asarray(prof.get("arm_offsets_deg", [0] * 5))),
                 gripper_deg_at_0=float(gm[0]), gripper_deg_at_100=float(gm[1]))
  sess_of = {e: s for s in prof["sessions"] for e in s.get("episodes", [])}
  summary = []
  for ep in a.episodes:
    t0 = time.time()
    s = sess_of[ep]
    cam = [s["camera"][k] for k in ("x", "y", "z", "yaw", "tilt_a", "tilt_b", "fovy")]
    st = D.states(ep)
    L = min(len(st), a.max_frames)
    first, last = D.frames(ep, "cam_head", [0, L - 1])
    # objects in the first / last frame (fixed sizes from stage 3)
    dims = {"cube": prof["cube"]["size"], "rim_radius": prof["bowl"]["rim_radius"],
            "height": prof["bowl"]["height"], "base_radius": prof["bowl"]["base_radius"]}

    def fit(img):
      arm = fo.dark_mask(img)
      keep = ~cv2.dilate(arm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
      gb, wb = fo.blobs(fo.green_mask(img), 150), fo.blobs(fo.bowl_mask(img), fo.MIN_WHITE)
      res = {}
      if gb:
        g = max(gb, key=lambda b: b[1])
        xy = fo.backproject_to_plane(cam, g[2][None], z0=0.02)[0, :2]
        res["cube"] = fo.fit_cube(cam, g[0], keep, xy, size=dims["cube"])
      if wb:
        w = max(wb, key=lambda b: b[1])
        xy = fo.backproject_to_plane(cam, w[2][None], z0=0.03)[0, :2]
        res["bowl"] = fo.fit_bowl(cam, w[0], keep, xy, dims=(dims["rim_radius"], dims["height"], dims["base_radius"]))
      return res

    f0, f1 = fit(first), fit(last)
    v = sample_variation(spec, 0, ep)
    v = fp.apply_to_variation(v, a.profile)
    v.extras["friend"]["session"] = s["id"]
    v.extras["friend"]["composite"] = not a.plain
    v.obj_xy = [f0["cube"]["x"], f0["cube"]["y"]]
    v.obj_yaw = f0["cube"]["yaw"]
    v.bin_xy = [f0["bowl"]["x"], f0["bowl"]["y"]]
    v.object_rgb, v.bin_rgb = prof["cube"]["rgb"], prof["bowl"]["rgb"]
    v.light_dir, v.light_diffuse, v.ambient, v.headlight = [0.0, 0.0, -1.0], 0.6, 0.3, 0.35
    sc = build_episode_scene(v, object_kind="box", render_size=(640, 480))
    m, d = sc.model, sc.data
    m.vis.quality.offsamples = 0
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    q_all = lerobot_to_sim(st[:L], cal)
    # grasp window from the gripper signal and the jaw-to-cube distance
    c0 = np.array([v.obj_xy[0], v.obj_xy[1], v.obj_size / 2])
    near = np.zeros(L, bool)
    for k in range(L):
      d.qpos[qadr] = q_all[k]
      mujoco.mj_kinematics(m, d)
      near[k] = np.linalg.norm(d.site_xpos[sc.site] - c0) < 0.04
    kg, kr = grasp_window(st[:L, 5], near)
    end_xyz = None
    if "cube" in f1:
      end_xyz = np.array([f1["cube"]["x"], f1["cube"]["y"], 0.0])
    plate = fp.plate_image(s["plate"]) if not a.plain else None
    table_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    r = mujoco.Renderer(m, 480, 640)
    tag = "plain" if a.plain else "composite"
    enc_h = _Encoder(out / f"twin_ep{ep:03d}_cam_head_{tag}.mp4", (1280, 480), "h264", fps=30 // a.step)
    enc_w = _Encoder(out / f"twin_ep{ep:03d}_cam_right.mp4", (1280, 480), "h264", fps=30 // a.step)
    ks = list(range(0, L, a.step))
    rh = D.frames(ep, "cam_head", ks)
    rw = D.frames(ep, "cam_right", ks)
    rel = None
    oq = sc.obj_qadr
    for j, k in enumerate(ks):
      d.qpos[qadr] = q_all[k]
      mujoco.mj_kinematics(m, d)
      Rg, pg = d.xmat[sc.grip_body].reshape(3, 3), d.xpos[sc.grip_body]
      if kg is not None and k >= kg and (kr is None or k < kr):
        if rel is None:  # cube pose in the gripper frame at the grasp frame
          rel = Rg.T @ (c0 - pg)
        p = pg + Rg @ rel
      elif kr is not None and k >= kr and end_xyz is not None:
        bz = fp.FLOOR_T + 0.001 if np.linalg.norm(end_xyz[:2] - np.array(v.bin_xy)) < prof["bowl"]["rim_radius"] else 0.0
        p = np.array([end_xyz[0], end_xyz[1], bz + v.obj_size / 2])
      elif kr is not None and k >= kr and rel is not None:
        p = pg + Rg @ rel
      else:
        p = c0
      d.qpos[oq:oq + 3] = p
      d.qpos[oq + 3:oq + 7] = [np.cos(v.obj_yaw / 2), 0, 0, np.sin(v.obj_yaw / 2)]
      mujoco.mj_forward(m, d)
      r.update_scene(d, camera="front")
      img = r.render()
      if plate is not None:
        r.enable_segmentation_rendering(); seg = r.render(); r.disable_segmentation_rendering()
        fg = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] != table_id)
        img = np.where(fg[..., None], img, plate)
      lab = lambda im, t: cv2.putText(im.copy(), t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
      enc_h.add(np.concatenate([lab(rh[j], f"REAL cam_head ep{ep} f{k}"),
                                lab(img, f"SIM kinematic replay ({tag})")], 1))
      r.update_scene(d, camera="wrist_cam")
      wimg = r.render()
      enc_w.add(np.concatenate([lab(rw[j], f"REAL cam_right ep{ep} f{k}"), lab(wimg, "SIM wrist_cam")], 1))
    enc_h.close(); enc_w.close(); r.close()
    row = {"episode": ep, "session": s["id"], "frames": len(ks), "grasp_frame": kg, "release_frame": kr,
           "cube_start": f0.get("cube"), "bowl": f0.get("bowl"), "cube_end": f1.get("cube"),
           "seconds": round(time.time() - t0, 1), "task": D.task(ep)}
    summary.append(row)
    print(json.dumps(row), flush=True)
  (out / f"twin_summary_{'plain' if a.plain else 'composite'}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
