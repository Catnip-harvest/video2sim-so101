"""Contact sheet (PNG) and/or a short montage video of several rendered episodes.

Rows = episodes, columns = phases (start, pre-grasp, squeeze, mid-carry, lowered,
released, end), taken from the front camera (and the wrist camera with --wrist).
Needs a rendered dataset (generate.py --render).

  python contact_sheet.py --dataset runs/sugar/lerobot --out sheet.png --episodes 6
  python contact_sheet.py --dataset runs/sugar/lerobot --video montage.mp4 --episodes 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLUMNS = [("start", "start"), ("pre_close", "pre-grasp"), ("squeeze", "squeeze"),
           ("carry", "mid-carry"), ("pre_release", "lowered"), ("post_release", "released"),
           ("rest", "end")]


def read_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
  import av
  out = {}
  with av.open(str(path)) as c:
    for k, fr in enumerate(c.decode(c.streams.video[0])):
      if k in wanted:
        out[k] = fr.to_ndarray(format="rgb24")
      if len(out) == len(wanted):
        break
  return out


def video_path(root: Path, info: dict, key: str, ep: int) -> Path:
  return root / info["video_path"].format(video_key=key, chunk_index=ep // 1000,
                                          file_index=ep % 1000)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", required=True)
  ap.add_argument("--out", help="PNG contact sheet")
  ap.add_argument("--video", help="montage mp4 (episodes tiled, played in sync)")
  ap.add_argument("--episodes", type=int, default=6)
  ap.add_argument("--wrist", action="store_true")
  args = ap.parse_args()
  from PIL import Image, ImageDraw

  root = Path(args.dataset)
  info = json.loads((root / "meta" / "info.json").read_text())
  keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
  if not keys:
    raise SystemExit("dataset has no videos (generated with --no-render)")
  key = "observation.images.wrist" if args.wrist and "observation.images.wrist" in keys \
    else "observation.images.front"
  meta = [json.loads(x) for x in (root / "meta" / "datagen_episodes.jsonl").read_text().splitlines() if x]
  n = min(args.episodes, len(meta))
  h, w = info["features"][key]["shape"][:2]

  if args.out:
    sheet = Image.new("RGB", (w * len(COLUMNS), (h + 14) * n + 16), "white")
    dr = ImageDraw.Draw(sheet)
    for j, (_, label) in enumerate(COLUMNS):
      dr.text((j * w + 4, 2), label, fill="black")
    for i in range(n):
      ps = meta[i]["phase_starts"]
      L = max(ps.values()) + 15
      idx = []
      for ph, _ in COLUMNS:
        if ph == "carry" and "carry" in ps:
          nxt = ps.get("lower", ps["carry"])
          idx.append((ps["carry"] + nxt) // 2)
        elif ph == "rest":
          idx.append(L - 1)
        else:
          idx.append(ps.get(ph, 0) + (5 if ph in ("squeeze", "post_release") else 0))
      frames = read_frames(video_path(root, info, key, i), set(idx))
      last = max(frames) if frames else 0
      for j, k in enumerate(idx):
        img = frames.get(k, frames.get(last))
        if img is not None:
          sheet.paste(Image.fromarray(img), (j * w, 16 + i * (h + 14)))
      v = meta[i]["variation"]
      dr.text((4, 16 + i * (h + 14) + h), f"ep {i}  cube {v['obj_size']*1e3:.1f} mm  "
              f"{v['obj_mass']*1e3:.0f} g  servo {v['servo_variant']}  "
              f"cam dz {v['cam_dz']*100:+.1f} cm pitch {v['cam_dpitch_deg']:+.1f} deg",
              fill="black")
    sheet.save(args.out)
    print(f"wrote {args.out}")

  if args.video:
    import av
    from fractions import Fraction
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    readers = []
    for i in range(n):
      c = av.open(str(video_path(root, info, key, i)))
      readers.append((c, c.decode(c.streams.video[0])))
    out = av.open(args.video, "w")
    st = out.add_stream("libx264", rate=30)
    st.width, st.height, st.pix_fmt = cols * w, rows * h, "yuv420p"
    st.time_base = Fraction(1, 30)
    last = [np.zeros((h, w, 3), np.uint8) for _ in range(n)]
    alive = [True] * n
    k = 0
    while any(alive):
      canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
      for i, (_, it) in enumerate(readers):
        if alive[i]:
          try:
            last[i] = next(it).to_ndarray(format="rgb24")
          except StopIteration:
            alive[i] = False
        r, cc = divmod(i, cols)
        canvas[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = last[i]
      fr = av.VideoFrame.from_ndarray(canvas, format="rgb24")
      fr.pts = k
      k += 1
      for p in st.encode(fr):
        out.mux(p)
    for p in st.encode():
      out.mux(p)
    out.close()
    for c, _ in readers:
      c.close()
    print(f"wrote {args.video}")


if __name__ == "__main__":
  main()
