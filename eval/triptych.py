"""Side-by-side eval videos: several models on the SAME scene (same seed, same episode id).

Each panel is one eval run's saved video (sim_eval.py --videos N), labelled with the model name
and its outcome from episodes.jsonl (L_full / L_task, first failure reason). A banner on top
says what the video is. Shorter episodes hold their last frame until the longest one ends.

  python eval/triptych.py --id 2 --title "simulation, scripted/expert-trained, seed 3000 id 2" \
      --panel "MolmoAct2=/work/outputs_large/eval/policy_final_n200_seed3000" \
      --panel "SmolVLA=/work/outputs_large/eval/smolvla_s3000" \
      --panel "ACT=/work/outputs_large/eval/act_s3000" \
      --out /work/outputs_large/deliverables/videos/triptych_inrange_s3000_id2.mp4 --png check.png

The script refuses to mix scenes: every panel's episode must have the same episode_seed.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path: str, size: int):
  try:
    return ImageFont.truetype(path, size)
  except OSError:
    return ImageFont.load_default()


def decode(path: Path) -> tuple[list[np.ndarray], float]:
  import av
  with av.open(str(path)) as c:
    s = c.streams.video[0]
    fps = float(s.average_rate or 10)
    return [f.to_ndarray(format="rgb24") for f in c.decode(s)], fps


def episode(run: Path, idx: int) -> dict:
  for line in (run / "episodes.jsonl").read_text().splitlines():
    if line.strip():
      e = json.loads(line)
      if e["attempt"] == idx:
        return e
  raise SystemExit(f"{run}: no episode with attempt id {idx}")


def outcome(e: dict) -> tuple[str, tuple[int, int, int]]:
  """Short outcome text and a colour: green = full pass, amber = task only, red = fail."""
  yn = lambda b: "pass" if b else "fail"
  txt = f"L_full {yn(e['L_full'])} / L_task {yn(e['L_task'])}"
  if e.get("reasons"):
    txt += f"  ({e['reasons'][0]})"
  col = (40, 160, 60) if e["L_full"] else (200, 140, 0) if e["L_task"] else (190, 40, 40)
  return txt, col


def panel(img: np.ndarray, name: str, out_txt: str, col, t: float, ended: bool) -> np.ndarray:
  W = img.shape[1]
  bar = Image.new("RGB", (W, 46), (0, 0, 0))
  d = ImageDraw.Draw(bar)
  d.text((6, 2), name, fill=(255, 255, 255), font=font(FONT_B, 18))
  d.text((6, 25), out_txt, fill=col, font=font(FONT, 14))
  im = Image.fromarray(img)
  dd = ImageDraw.Draw(im)
  dd.text((6, im.height - 18), f"t={t:4.1f} s" + ("  (ended, last frame held)" if ended else ""),
          fill=(255, 255, 255), font=font(FONT, 13), stroke_width=2, stroke_fill=(0, 0, 0))
  return np.concatenate([np.asarray(bar), np.asarray(im)], 0)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--panel", action="append", required=True, help="NAME=RUN_DIR (repeat)")
  ap.add_argument("--id", type=int, required=True, help="episode attempt id (same in all runs)")
  ap.add_argument("--title", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--png", help="also save the frame at --png-at seconds as a PNG")
  ap.add_argument("--png-at", type=float, default=8.0)
  args = ap.parse_args()
  import av

  names, runs = zip(*(p.split("=", 1) for p in args.panel))
  eps = [episode(Path(r), args.id) for r in runs]
  seeds = {e["episode_seed"] for e in eps}
  if len(seeds) != 1:
    raise SystemExit(f"panels are different scenes: episode_seeds {seeds}")
  clips, fps = [], None
  for r, e in zip(runs, eps):
    v = Path(r) / "videos" / Path(e["video"]).name
    frames, f = decode(v)
    if fps is not None and abs(f - fps) > 1e-6:
      raise SystemExit(f"fps differ: {fps} vs {f}")
    fps = f
    clips.append(frames)
  L = max(len(c) for c in clips)
  outs = [outcome(e) for e in eps]
  gap = 6
  first = [panel(c[0], n, o[0], o[1], 0, False) for c, n, o in zip(clips, names, outs)]
  W = sum(p.shape[1] for p in first) + gap * (len(first) - 1)
  H = first[0].shape[0] + 30
  W, H = W // 2 * 2, H // 2 * 2

  out = av.open(args.out, "w")
  st = out.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
  st.width, st.height, st.pix_fmt = W, H, "yuv420p"
  st.options = {"crf": "20"}
  png_k = min(L - 1, int(round(args.png_at * fps)))
  for k in range(L):
    cols = []
    for c, n, o in zip(clips, names, outs):
      j = min(k, len(c) - 1)
      cols.append(panel(c[j], n, o[0], o[1], j / fps, k >= len(c)))
      cols.append(np.zeros((cols[-1].shape[0], gap, 3), np.uint8))
    body = np.concatenate(cols[:-1], 1)
    banner = Image.new("RGB", (body.shape[1], 30), (30, 30, 30))
    ImageDraw.Draw(banner).text((8, 5), args.title, fill=(255, 255, 255), font=font(FONT_B, 17))
    frame = np.concatenate([np.asarray(banner), body], 0)[:H, :W]
    if args.png and k == png_k:
      Image.fromarray(frame).save(args.png)
    vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
    for p in st.encode(vf):
      out.mux(p)
  for p in st.encode():
    out.mux(p)
  out.close()
  for n, e, o in zip(names, eps, outs):
    print(f"  {n}: {o[0]}  steps={e['steps_10hz']}")
  print(f"wrote {args.out} ({L} frames at {fps:g} fps, {W}x{H})")


if __name__ == "__main__":
  main()
