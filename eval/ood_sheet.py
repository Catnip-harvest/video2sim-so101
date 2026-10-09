"""4x4 contact sheet of the first policy observation for an OOD rung (docs/OOD_LADDER.md).

  python eval/ood_sheet.py --ood camera --seed 4000 --out /work/outputs_large/eval/ood_sheets/camera.png

Each tile = ids 0..15: the scene is built and settled exactly as the harness does
(sim_eval.setup_scene), then rendered by the harness's own ObsRenderer (so --ood exposure
is applied, and the h264 round-trip with --codec), front camera large, wrist camera inset.
The label gives the id and what the rung changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_eval as E  # noqa: E402  (sets MUJOCO_GL=osmesa and sys.path)
from ood import RUNGS, apply_ood  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.variations import sample_variation  # noqa: E402


def _label(v) -> str:
  o = (v.extras or {}).get("ood") or {}
  parts = []
  if "camera" in o:
    c = o["camera"]
    parts.append(f"dz{c['dz_m']*100:+.0f}cm p{c['dpitch_deg']:+.0f} y{c['dyaw_deg']:+.0f} "
                 f"d{c['ddist_m']*100:+.0f}cm f{c['fovy_deg']:.0f}")
  if "exposure" in o:
    e = o["exposure"]
    parts.append({"brightness": f"bright x{e['brightness']}", "contrast": f"contrast x{e['contrast']}",
                  "white_balance": f"wb R{e['wb_gain'][0]} B{e['wb_gain'][2]}"}[e["kind"]])
  if "colour" in o:
    parts.append(o["colour"]["lights"])
  if "distractor" in o:
    parts.append("+" + ",".join(o["distractor"]["shapes"]))
  return " | ".join(parts)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ood", default="none", choices=list(RUNGS))
  ap.add_argument("--seed", type=int, default=4000)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--spec", default=str(E.SPEC))
  ap.add_argument("--codec", action="store_true", help="also apply the h264 round-trip")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  import numpy as np
  from PIL import Image, ImageDraw

  spec = load_task_spec(a.spec)
  tiles = []
  for i in range(a.start, a.start + 16):
    v = apply_ood(sample_variation(spec, a.seed, i, variety=True), a.ood)
    sc = E.setup_scene(spec, v)
    rend = E.ObsRenderer(spec, v, codec=a.codec)
    imgs = rend.render(sc.data.qpos)
    rend.close()
    front = Image.fromarray(imgs["observation.images.cam_head"])
    wrist = Image.fromarray(imgs["observation.images.cam_right"]).resize((96, 96))
    front.paste(wrist, (256 - 98, 256 - 98))
    dr = ImageDraw.Draw(front)
    dr.rectangle((0, 0, 256, 12), fill=(0, 0, 0))
    dr.text((2, 1), f"{i}: {_label(v)}"[:48], fill=(255, 255, 255))
    tiles.append(np.asarray(front))
  rows = [np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1) for r in range(4)]
  sheet = np.concatenate(rows, axis=0)
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(sheet).save(out)
  print(f"[ood_sheet] {a.ood}: wrote {out} {sheet.shape}")


if __name__ == "__main__":
  main()
