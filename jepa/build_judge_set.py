"""Assemble the JEPA-judge episode table: one row per rendered sim episode.

Sources (all generated with base seed 1, so attempt i is the SAME scene - object/bin
layout, colours, lights, camera - in every source; only the physics or expert differs):
  base           datagen/sugar_v10_n1000 (foh servos, normal DR). Accepted attempts with
                 id < POS_MAX_ATTEMPT are positives; ALL its rejects are "natural" negatives.
  zoh            --servo-hold zoh (servo goal jumps every frame; cubes shake loose)
  widen          --widen (heavier/slipperier cubes, weaker jaw and arm)
  no_close       --inject no_close (the expert never closes the jaw)
  early_release  --inject early_release (the expert opens the jaw mid-carry)
Every label is the physics verifier's verdict (rollout.py). An injected or zoh episode
the verifier passes is a POSITIVE, so "which run it came from" is not the label.

Labels:
  y_full  1 = the verifier accepted the episode (every check, incl. collisions/torque)
  y_task  1 = the task outcome is fine: object lifted, ends released and resting in the bin
          (a transient arm-table touch fails y_full but not y_task; it is mostly invisible
          in the last frame)

Splits (by scene = attempt id, never by frame or episode):
  cam_test  scenes whose front camera was raised by cam_dz >= CAM_BAND_LO (top ~20 % of the
            +-8 cm height range): a held-out camera-height band
  test      40 % of the remaining scenes (random, seed 0)
  train     the other 60 %
All episodes of one scene land in the same split.

  python jepa/build_judge_set.py   ->  /work/outputs_large/jepa/judge_set.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/work/outputs_large/datagen/sugar_v10_n1000")
GEN = Path("/work/outputs_large/jepa/gen")
OUT = Path("/work/outputs_large/jepa/judge_set.json")
POS_MAX_ATTEMPT = 200
CAM_BAND_LO = 0.048
TEST_FRAC = 0.4
TASK_FAIL = {"grasp_failed", "not_lifted", "not_in_place_region", "not_resting",
             "not_released", "unstable", "exception"}


def rows_of(run_dir: Path) -> list[dict]:
  return [json.loads(l) for l in (run_dir / "attempts.jsonl").read_text().splitlines() if l.strip()]


def main() -> None:
  sources = {"base": BASE} | {n: GEN / n for n in ("zoh", "widen", "no_close", "early_release")}
  eps = []
  for src, d in sources.items():
    if not (d / "attempts.jsonl").exists():
      print(f"missing {d}, skipped")
      continue
    for r in rows_of(d):
      i = r["attempt"]
      stem = d / "staging" / f"attempt_{i:06d}"
      front = Path(f"{stem}.front.mp4")
      if not front.exists():  # nothing was simulated (e.g. IK failed): no video to judge
        continue
      if src == "base" and r["accepted"] and i >= POS_MAX_ATTEMPT:
        continue
      reasons = r["reasons"]
      y_full = int(r["accepted"])
      y_task = int(not (set(reasons) & TASK_FAIL))
      if y_full:
        ftype = "positive"
      elif src == "base":
        ftype = "natural"
      else:
        ftype = src
      eps.append({
        "id": f"{src}/{i:06d}", "source": src, "attempt": i, "scene": i,
        "y_full": y_full, "y_task": y_task, "failure_type": ftype,
        "primary_reason": r["primary_reason"], "reasons": reasons,
        "cam_dz": r["variation"]["cam_dz"], "frames_30hz": r["frames"],
        "front": str(front), "wrist": f"{stem}.wrist.mp4",
      })
  scenes = sorted({e["scene"] for e in eps})
  cam_scenes = {e["scene"] for e in eps if e["cam_dz"] >= CAM_BAND_LO}
  rest = [s for s in scenes if s not in cam_scenes]
  perm = np.random.default_rng(0).permutation(rest)
  test_scenes = set(perm[: int(round(TEST_FRAC * len(rest)))].tolist())
  for e in eps:
    e["split"] = ("cam_test" if e["scene"] in cam_scenes else
                  "test" if e["scene"] in test_scenes else "train")
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(eps, indent=1))
  print(f"{len(eps)} episodes, {len(scenes)} scenes -> {OUT}")
  for split in ("train", "test", "cam_test"):
    sub = [e for e in eps if e["split"] == split]
    print(split, len(sub), "scenes", len({e['scene'] for e in sub}),
          "y_full", Counter(e["y_full"] for e in sub), "y_task", Counter(e["y_task"] for e in sub),
          dict(Counter(e["failure_type"] for e in sub)))
  print("by source x label:", dict(Counter((e["source"], e["y_full"]) for e in eps)))
  print("negative primary reasons:", dict(Counter((e["source"], e["primary_reason"])
                                                  for e in eps if not e["y_full"])))


if __name__ == "__main__":
  main()
