"""Assemble the friend_bowl profile JSON from the stage 1-4 outputs (all on the server).

  python -m friendrig.build_profile --camfit OUT --cams cam_sessions_offsets_roll.json \
      --profile OUT/friend_bowl_profile.json [--wrist OUT/wrist/wrist_fit.json] [--no-composite]

Colours: median RGB of the green-cube and white-bowl pixels in the first cam_head frame of the
detected episodes (sRGB / 255, used as MuJoCo rgba).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--camfit", required=True)
  ap.add_argument("--cams", required=True)
  ap.add_argument("--profile", required=True)
  ap.add_argument("--wrist", default=None)
  ap.add_argument("--no-composite", action="store_true")
  ap.add_argument("--jitter", type=float, default=0.015)
  ap.add_argument("--look-dr", type=int, default=1)
  a = ap.parse_args()
  out = Path(a.camfit)
  cs = json.loads((out / a.cams).read_text())
  ob = json.loads((out / "objects.json").read_text())
  pl = json.loads((out / "plates" / "plates.json").read_text())
  pmap = {r["session"]: r for r in pl["sessions"]}
  sessions = []
  for s in cs["sessions"]:
    sessions.append({"id": s["id"], "weight": len(s["train_episodes"]), "camera": s["camera"],
                     "episodes": s["train_episodes"] + s["held_out"],
                     "plate": str(out / "plates" / pmap[s["id"]]["plate"]),
                     "ortho": str(out / "plates" / pmap[s["id"]]["ortho"])})
  # colours from first frames of detected episodes
  import cv2
  from friendrig.masks import green_mask, white_mask  # white_mask: bright bowl pixels only, for colour
  from friendrig.realdata import FriendData
  D = FriendData()
  det = [r["episode"] for r in ob["episodes"] if "cube" in r][:40]
  G, Wt = [], []
  for e in det:
    img = D.frames(e, "cam_head", [0])[0]
    G.append(img[green_mask(img)]); Wt.append(img[white_mask(img)])
  g_rgb = (np.median(np.concatenate(G), 0) / 255).round(3).tolist()
  w_rgb = (np.median(np.concatenate(Wt), 0) / 255).round(3).tolist()
  wrist = None
  if a.wrist:
    wf = json.loads(Path(a.wrist).read_text())
    c = wf["fitted"]
    wrist = {"dpos": c[:3], "drot": c[3:6], "fovy": c[6], "flip": bool(wf.get("flip", False)), "source": a.wrist}
  dims = ob["dims_m"]
  prof = {"name": "friend_bowl", "sessions": sessions,
          "ortho_extent": {"x": list(pl["ortho_x_m"]), "y": list(pl["ortho_y_m"])},
          "wrist_cam": wrist, "arm_offsets_deg": cs.get("offsets_deg_added_to_default", [0] * 5),
          "gripper_map_deg_at_0_100": (wf.get("gripper_map_deg_at_0_100") if a.wrist else None),
          "cube": {"size": dims["cube"], "rgb": g_rgb},
          "bowl": {"rim_radius": dims["rim_radius"], "height": dims["height"], "base_radius": dims["base_radius"],
                   "rgb": w_rgb},
          "placements_csv": str(out / "placements_train.csv"), "placement_jitter_m": a.jitter,
          "arm_rgb": [0.05, 0.05, 0.06], "composite": not a.no_composite, "look_dr": bool(a.look_dr),
          "sources": {"cams": a.cams, "objects": "objects.json", "plates": "plates/plates.json"}}
  Path(a.profile).write_text(json.dumps(prof, indent=1))
  # ArmCalib for the dataset export (sim radians <-> friend's lerobot degrees), same numbers
  from calib.so101_units import ArmCalib
  base = ArmCalib()
  gm = prof["gripper_map_deg_at_0_100"] or [base.gripper_deg_at_0, base.gripper_deg_at_100]
  cal = ArmCalib(offset_deg=list(np.asarray(base.offset_deg) + np.asarray(prof["arm_offsets_deg"])),
                 gripper_deg_at_0=float(gm[0]), gripper_deg_at_100=float(gm[1]),
                 source=f"friendrig stage 1 ({a.cams}) + stage 2 ({a.wrist}); fitted on TRAIN episodes")
  cal.save(Path(a.profile).with_name("friend_arm_calib.json"))
  print(json.dumps({k: prof[k] for k in ("cube", "bowl", "arm_offsets_deg", "wrist_cam")}, indent=1))


if __name__ == "__main__":
  main()
