"""P3-A-lite: 4x4 contact sheets of the first observation (cam_head and cam_right) and a
segmentation visibility check that all three cubes are in cam_head.

  python eval/mt_sheet.py --seed 2 --out /work/outputs_large/eval/p3a_sheets/train_s2 [--row train]
  python eval/mt_sheet.py --seed 2 --visibility 200 --out ...    (counts only, no sheet)

Scenes are built and settled exactly as the eval harness does (sim_eval.setup_scene) and
rendered by its ObsRenderer (what a policy sees). Tile label: id, then the instruction.
The cube demos for P3-A come from a SCRIPTED EXPERT, not a human video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_eval as E  # noqa: E402  (sets MUJOCO_GL=osmesa and sys.path)
import multitask as MT  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.variations import sample_multitask_variation  # noqa: E402

MIN_PIXELS = 20  # a cube counts as visible in cam_head with at least this many pixels


def cube_pixels(spec, v, size=(256, 256)) -> dict[str, int]:
  """colour -> visible pixels of that cube in the front camera (segmentation render)."""
  import mujoco
  import numpy as np
  sc = E.setup_scene(spec, v, size)
  m, d = sc.model, sc.data
  ids = MT.cube_bodies(sc, v)
  r = mujoco.Renderer(m, size[1], size[0])
  r.enable_segmentation_rendering()
  r.update_scene(d, camera="front")
  seg = r.render()
  r.close()
  geom = seg[..., 0]
  typ = seg[..., 1]
  out = {}
  for c, b in ids.items():
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    out[c] = int(np.sum(np.isin(geom, gs) & (typ == int(mujoco.mjtObj.mjOBJ_GEOM))))
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--seed", type=int, default=2)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--row", default="datagen", choices=("datagen",) + MT.MT_ROWS,
                  help="datagen = the training sampler; else an eval row of eval/multitask.py")
  ap.add_argument("--layout", default="default")
  ap.add_argument("--visibility", type=int, default=0, help="also check N scenes for visibility")
  ap.add_argument("--spec", default=str(E.SPEC))
  ap.add_argument("--out", required=True, help="output prefix (…_cam_head.png etc.)")
  a = ap.parse_args()
  import numpy as np
  from PIL import Image, ImageDraw

  spec = load_task_spec(a.spec)

  def var(i, colour=None):
    if a.row == "datagen":
      return sample_multitask_variation(spec, a.seed, i, variety=True, layout=a.layout)
    return MT.eval_variation(spec, a.seed, i, a.row, colour)

  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  items = [(i, None) for i in range(a.start, a.start + 16)]
  if a.row == "swap":
    items = [(i, "red") for i in range(a.start, a.start + 16)]
  tiles = {"cam_head": [], "cam_right": []}
  for i, c in items:
    v = var(i, c)
    sc = E.setup_scene(spec, v)
    rend = E.ObsRenderer(spec, v)
    imgs = rend.render(sc.data.qpos)
    rend.close()
    for cam in tiles:
      im = Image.fromarray(imgs[f"observation.images.{cam}"])
      dr = ImageDraw.Draw(im)
      dr.rectangle((0, 0, 256, 12), fill=(0, 0, 0))
      dr.text((2, 1), f"{i}: {v.extras['multitask']['instruction']}"[:46], fill=(255, 255, 255))
      tiles[cam].append(np.asarray(im))
  for cam, t in tiles.items():
    sheet = np.concatenate([np.concatenate(t[r * 4:(r + 1) * 4], axis=1) for r in range(4)], axis=0)
    p = out.parent / f"{out.name}_{cam}.png"
    Image.fromarray(sheet).save(p)
    print(f"[mt_sheet] wrote {p} {sheet.shape}")
  if a.visibility:
    rows, worst = [], []
    for i in range(a.start, a.start + a.visibility):
      v = var(i, "red" if a.row == "swap" else None)
      px = cube_pixels(spec, v)
      rows.append(px)
      worst.append((min(px.values()), i))
    n_ok = sum(min(p.values()) >= MIN_PIXELS for p in rows)
    allpx = [x for p in rows for x in p.values()]
    rep = {"seed": a.seed, "row": a.row, "scenes": len(rows), "min_pixels": MIN_PIXELS,
           "all_three_visible": n_ok, "cube_pixels_min": int(min(allpx)),
           "cube_pixels_median": float(np.median(allpx)),
           "worst_ids": [i for _, i in sorted(worst)[:5]]}
    (out.parent / f"{out.name}_visibility.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
  main()
