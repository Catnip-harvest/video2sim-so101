"""4x4 contact sheet of --wide front-camera frames from a datagen run's staging dir.

  python wide_sheet.py RUN_DIR OUT.png [--frame first|mid] [--n 16]

Each tile is the first (or middle) frame of one accepted episode's stored front video, in
attempt order, captioned with its camera draw (height, pitch, yaw, distance, FOV) and
exposure (brightness, contrast, gamma, blur).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def frame_of(path: Path, which: str) -> np.ndarray:
  with av.open(str(path)) as c:
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
  return frames[0] if which == "first" else frames[len(frames) // 2]


def caption(meta: dict) -> list[str]:
  w = (meta.get("variation", {}).get("extras", {}) or {}).get("wide", {})
  c, e = w.get("camera", {}), w.get("exposure", {})
  a = meta["attempt"]
  if not c or c.get("fallback_narrow"):
    l1 = f"#{a} narrow camera"
  else:
    l1 = (f"#{a} h{100 * c['dz_m']:+.0f}cm p{c['dpitch_deg']:+.0f} y{c['dyaw_deg']:+.0f} "
          f"d{100 * c['ddist_m']:+.0f}cm f{c['fovy_deg']:.0f}")
  l2 = (f"b{e.get('brightness', 1):.2f} c{e.get('contrast', 1):.2f} g{e.get('gamma', 1):.2f} "
        f"bl{e.get('blur_sigma', 0):.1f} r{meta.get('wide_camera', {}).get('redraws', 0)}")
  return [l1, l2]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("run")
  ap.add_argument("out")
  ap.add_argument("--frame", default="first", choices=["first", "mid"])
  ap.add_argument("--n", type=int, default=16)
  ap.add_argument("--start", type=int, default=0, help="skip this many accepted episodes")
  args = ap.parse_args()
  st = Path(args.run) / "staging"
  metas = []
  for j in sorted(st.glob("attempt_*.json")):
    m = json.loads(j.read_text())
    if m.get("accepted") and Path(str(j)[:-5] + ".front.mp4").exists():
      metas.append((j, m))
  metas = metas[args.start: args.start + args.n]
  tiles = []
  for j, m in metas:
    img = Image.fromarray(frame_of(Path(str(j)[:-5] + ".front.mp4"), args.frame))
    tiles.append((img, caption(m)))
  if not tiles:
    raise SystemExit("no accepted episodes with front videos")
  w, h = tiles[0][0].size
  cols = 4
  rows = (len(tiles) + cols - 1) // cols
  cap = 26
  sheet = Image.new("RGB", (cols * w, rows * (h + cap)), "white")
  dr = ImageDraw.Draw(sheet)
  for k, (img, lines) in enumerate(tiles):
    x, y = (k % cols) * w, (k // cols) * (h + cap)
    sheet.paste(img, (x, y))
    for li, t in enumerate(lines):
      dr.text((x + 3, y + h + 1 + 12 * li), t, fill=(0, 0, 0))
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  sheet.save(args.out)
  print(f"wrote {args.out}: {len(tiles)} tiles ({args.frame} frame)")


if __name__ == "__main__":
  main()
