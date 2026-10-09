"""P3-A-lite eval helpers: named-cube scenes, instruction swap, named-cube scoring, leakage.

The training demos for P3-A come from a SCRIPTED EXPERT (our planner), not a human video.

Eval rows (seed 5000; reported separately, never pooled, because they reuse scenes):
  train      ids 0-59: held-out scenes, training templates, target red/green/blue by id % 3
  paraphrase ids 0-29: the SAME scenes as "train", held-out paraphrase templates
  swap       ids 0-19: one scene with red, green and blue cubes, run once per named colour
  yellow     ids 100-119: yellow (never a training target) is named, training template

Named-cube scoring. A cube "moved" if its centre was displaced > 20 mm in xy from where it
settled, or lifted > 10 mm above its settled height, at any 10 Hz step or at the end. Each
run is one of: named (only the named cube moved), other (only non-named cubes moved, i.e.
one other cube), none (no cube moved), multiple (more than one cube moved). Named-cube rate
= named / all runs, with the none and multiple counts shown. ``named_in_tray`` adds the
task-level check: the named cube ends inside the tray walls (spec success margin) and no
other cube is in the tray.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "datagen", _ROOT / "eval"):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from r2s.variations import (HELDOUT_TEMPLATES, TRAIN_TARGETS, TRAIN_TEMPLATES, Variation,
                            sample_multitask_variation, retarget)

MOVED_XY = 0.020
LIFTED_Z = 0.010
MT_ROWS = ("train", "paraphrase", "swap", "yellow")
SWAP_COLOURS = ("red", "green", "blue")


def eval_items(row: str, ids: list[int]) -> list[tuple[int, str | None]]:
  """Work items (attempt id, named colour or None). Swap: three runs per scene id."""
  if row == "swap":
    return [(i, c) for i in ids for c in SWAP_COLOURS]
  return [(i, None) for i in ids]


def eval_variation(spec, seed: int, idx: int, row: str, colour: str | None = None) -> Variation:
  if row == "train":
    return sample_multitask_variation(spec, seed, idx, variety=True)
  if row == "paraphrase":  # same physical scene and target as "train", other wording
    return sample_multitask_variation(spec, seed, idx, variety=True,
                                      template=HELDOUT_TEMPLATES[idx % len(HELDOUT_TEMPLATES)])
  if row == "swap":  # red is the planner's object; green and blue are the extras
    base = sample_multitask_variation(spec, seed, idx, variety=True, target_colour="red",
                                      distractor_colours=("green", "blue"),
                                      template=TRAIN_TEMPLATES[0])
    return retarget(base, colour, TRAIN_TEMPLATES[idx % len(TRAIN_TEMPLATES)])
  if row == "yellow":
    return sample_multitask_variation(spec, seed, idx, variety=True, target_colour="yellow",
                                      allow_yellow_target=True)
  raise ValueError(row)


def cube_bodies(sc, v: Variation) -> dict[str, int]:
  """colour -> body id for the three cubes."""
  import mujoco
  cols = v.extras["multitask"]["colours"]
  return {c: mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_BODY, name) for name, c in cols.items()}


class CubeTracker:
  """Settled pose of every cube, and its largest xy displacement / lift since."""

  def __init__(self, sc, v: Variation):
    self.d = sc.data
    self.ids = cube_bodies(sc, v)
    self.rest = {c: self.d.xpos[b].copy() for c, b in self.ids.items()}
    self.max_xy = {c: 0.0 for c in self.ids}
    self.max_lift = {c: 0.0 for c in self.ids}

  def update(self) -> None:
    for c, b in self.ids.items():
      p = self.d.xpos[b]
      self.max_xy[c] = max(self.max_xy[c], float(np.linalg.norm(p[:2] - self.rest[c][:2])))
      self.max_lift[c] = max(self.max_lift[c], float(p[2] - self.rest[c][2]))

  def final(self) -> dict[str, list[float]]:
    return {c: self.d.xpos[b].tolist() for c, b in self.ids.items()}


def named_outcome(moved: dict[str, bool], named: str) -> str:
  m = {c for c, x in moved.items() if x}
  if not m:
    return "none"
  if len(m) > 1:
    return "multiple"
  return "named" if m == {named} else "other"


def score_named(tr: CubeTracker, v: Variation, spec) -> dict:
  tr.update()
  named = v.extras["multitask"]["target_colour"]
  moved = {c: (tr.max_xy[c] > MOVED_XY or tr.max_lift[c] > LIFTED_Z) for c in tr.ids}
  lim = 0.055 - spec.place.success_xy_margin
  bx, by = v.bin_xy
  fin = tr.final()
  in_tray = {c: bool(abs(p[0] - bx) <= lim and abs(p[1] - by) <= lim) for c, p in fin.items()}
  return {"named": named, "outcome": named_outcome(moved, named), "moved": moved,
          "in_tray": in_tray,
          "named_in_tray": bool(in_tray[named] and not any(x for c, x in in_tray.items() if c != named)),
          "max_xy_mm": {c: round(x * 1e3, 1) for c, x in tr.max_xy.items()},
          "max_lift_mm": {c: round(x * 1e3, 1) for c, x in tr.max_lift.items()}}


def summarize_named(rows: list[dict]) -> dict:
  sc = [r["named_cube"] for r in rows if "named_cube" in r]
  n = len(sc)
  cnt = Counter(s["outcome"] for s in sc)
  from sim_eval import wilson
  k = cnt.get("named", 0)
  kt = sum(s["named_in_tray"] for s in sc)
  by = {}
  for c in sorted({s["named"] for s in sc}):
    sub = [s for s in sc if s["named"] == c]
    by[c] = {"n": len(sub), "named": sum(s["outcome"] == "named" for s in sub),
             "named_in_tray": sum(s["named_in_tray"] for s in sub)}
  return {"n": n, "outcomes": {o: cnt.get(o, 0) for o in ("named", "other", "none", "multiple")},
          "named_cube_rate": round(k / max(n, 1), 4), "wilson95_named": [round(x, 4) for x in wilson(k, n)],
          "named_in_tray": kt, "named_in_tray_rate": round(kt / max(n, 1), 4),
          "wilson95_named_in_tray": [round(x, 4) for x in wilson(kt, n)], "by_named_colour": by}


# ---- leakage: eval scenes must not appear in the training set ---------------------------
def scene_signature(v: Variation | dict) -> dict[str, tuple[float, float]]:
  """colour -> cube xy for a multitask Variation (or its dict)."""
  d = v.to_dict() if isinstance(v, Variation) else v
  mt = d["extras"]["multitask"]
  xy = {"object": tuple(d["obj_xy"])}
  for b in d["extras"]["free_bodies"]:
    xy[b["name"]] = tuple(b["xy"])
  return {c: xy[name] for name, c in mt["colours"].items()}


def same_scene(a: dict, b: dict, tol: float = 0.001) -> bool:
  """Match = the same set of colours and every colour's cube xy within ``tol``."""
  if set(a) != set(b):
    return False
  return all(np.linalg.norm(np.subtract(a[c], b[c])) <= tol for c in a)


def training_signatures(sidecars: list[Path]) -> list[dict]:
  """Signatures from datagen sidecars: <dataset>/meta/datagen_episodes.jsonl rows or
  <run>/attempts.jsonl rows (both carry the episode's "variation")."""
  out = []
  for p in sidecars:
    for line in Path(p).read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      r = json.loads(line)
      v = r.get("variation") or {}
      if (v.get("extras") or {}).get("multitask"):
        out.append(scene_signature(v))
  return out


def leakage(eval_vs: list[Variation], train_sigs: list[dict], tol: float = 0.001) -> list[tuple[int, int]]:
  """(eval index, train index) pairs that match; [] = no overlap."""
  hits = []
  if not train_sigs:
    return hits
  # Cheap prefilter on the red/green/blue/yellow xy of any shared colour.
  for i, v in enumerate(eval_vs):
    a = scene_signature(v)
    for j, b in enumerate(train_sigs):
      if same_scene(a, b, tol):
        hits.append((i, j))
  return hits


def assert_no_leakage(spec, sidecars: list[Path], seed: int = 5000, rows=MT_ROWS) -> dict:
  ids = {"train": range(0, 60), "paraphrase": range(0, 30), "swap": range(0, 20),
         "yellow": range(100, 120)}
  evs = [eval_variation(spec, seed, i, r, c) for r in rows for i, c in eval_items(r, list(ids[r]))]
  sigs = training_signatures(sidecars)
  hits = leakage(evs, sigs)
  rep = {"eval_seed": seed, "eval_variations": len(evs), "train_signatures": len(sigs),
         "matches": len(hits), "tolerance_mm": 1.0}
  assert not hits, f"eval/train scene overlap: {hits[:5]}"
  return rep


if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser(description="P3-A leakage check: eval scenes vs training sidecars")
  ap.add_argument("sidecars", nargs="+", help="datagen_episodes.jsonl / attempts.jsonl files")
  ap.add_argument("--seed", type=int, default=5000)
  ap.add_argument("--spec", default=None)
  a = ap.parse_args()
  from r2s.paths import load_task_spec
  from sim_eval import SPEC
  rep = assert_no_leakage(load_task_spec(a.spec or SPEC), [Path(p) for p in a.sidecars], a.seed)
  print(json.dumps(rep))
  sys.exit(0)
