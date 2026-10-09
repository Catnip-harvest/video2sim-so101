"""Side-by-side video: the human demo clip (input) left, a generated robot episode right.

The two clocks are aligned piecewise-linearly on three events: start, grasp (human
grasp_time <-> robot "close" phase start) and release (human release_time <-> robot
"release" phase start); the robot plays in real time, the human clip is slowed to match.

  python side_by_side.py --dataset runs/sugar/lerobot --episode 0 \
      --spec ../demo/outputs/pexels_7488188_sugar_cube/so101_task_spec.json --out sbs.mp4
The human clip defaults to the spec's source.video.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def decode_all(path: Path) -> tuple[list[np.ndarray], list[float]]:
  import av
  frames, times = [], []
  with av.open(str(path)) as c:
    s = c.streams.video[0]
    for fr in c.decode(s):
      frames.append(fr.to_ndarray(format="rgb24"))
      times.append(float(fr.pts * s.time_base) if fr.pts is not None else len(times) / 30)
  t0 = times[0]
  return frames, [t - t0 for t in times]


def fit_h(img: np.ndarray, H: int):
  from PIL import Image
  h, w = img.shape[:2]
  W = int(round(w * H / h)) // 2 * 2
  return np.asarray(Image.fromarray(img).resize((W, H)))


def label(img: np.ndarray, text: str) -> np.ndarray:
  from PIL import Image, ImageDraw
  im = Image.fromarray(img)
  d = ImageDraw.Draw(im)
  d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
  d.text((6, 3), text, fill=(255, 255, 255))
  return np.asarray(im)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", required=True)
  ap.add_argument("--episode", type=int, default=0)
  ap.add_argument("--spec", required=True, help="so101_task_spec.json (grasp/release times)")
  ap.add_argument("--human-video", help="default: the spec's source.video")
  ap.add_argument("--camera", default="observation.images.front")
  ap.add_argument("--height", type=int, default=360)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  import av

  spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
  human = Path(args.human_video or spec["source"]["video"])
  traj = spec["object_trajectory"]
  g_h, r_h = float(traj["grasp_time"]), float(traj["release_time"])
  root = Path(args.dataset)
  info = json.loads((root / "meta" / "info.json").read_text())
  meta = [json.loads(x) for x in (root / "meta" / "datagen_episodes.jsonl").read_text().splitlines() if x]
  ps = meta[args.episode]["phase_starts"]
  rpath = root / info["video_path"].format(video_key=args.camera, chunk_index=args.episode // 1000,
                                           file_index=args.episode % 1000)
  hf, ht = decode_all(human)
  rf, _ = decode_all(rpath)
  L = len(rf)
  fps = info["fps"]
  # robot frame -> human time
  xr = [0, ps["close"], ps["release"], L - 1]
  xh = [0.0, g_h, r_h, ht[-1]]
  ht_arr = np.array(ht)
  H = args.height
  first_l = fit_h(hf[0], H)
  first_r = fit_h(rf[0], H)
  W = first_l.shape[1] + first_r.shape[1]
  out = av.open(args.out, "w")
  st = out.add_stream("libx264", rate=fps)
  st.width, st.height, st.pix_fmt = W, H, "yuv420p"
  st.time_base = Fraction(1, fps)
  for k in range(L):
    th = float(np.interp(k, xr, xh))
    j = int(np.argmin(np.abs(ht_arr - th)))
    left = label(fit_h(hf[j], H), f"human demo (input)  t={th:4.2f} s")
    right = label(fit_h(rf[k], H), f"generated SO-101 episode {args.episode} (MuJoCo, "
                                   f"physics-verified)  t={k / fps:4.1f} s")
    fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(np.concatenate([left, right], 1)),
                                    format="rgb24")
    fr.pts = k
    for p in st.encode(fr):
      out.mux(p)
  for p in st.encode():
    out.mux(p)
  out.close()
  print(f"wrote {args.out} ({L} frames)")


if __name__ == "__main__":
  main()
