"""Phase-free verifier vs expert verifier on the SAME simulations (FINAL_DESIGN section 3 test).

  python datagen/tests/verify_agreement.py --n 60 --zoh 20 --workers 4 --out /work/outputs_large/eval/verify_agreement.json

Groups (all eval seeds, --seed 1000, variety on; training used --seed 1):
  expert    scripted expert, FOH servos (datagen default)       -> verdicts should agree
  zoh       same seeds with zero-order-hold servos (servo_foh=False): drops     -> agree
  no_close  PlanConfig(close_jaw=False): the jaw never closes   -> both must fail
  high_rel  PlanConfig(release_gap=0.06): opens 6 cm above the floor; the expert verifier
            has no release-height clause, the phase-free one must fail it (release_too_high)
For each episode both verdicts come from one simulation (verify_policy.run_expert_judged).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

DATAGEN = Path(__file__).resolve().parents[1]
SPEC = DATAGEN.parent / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"


def _job(args):
  group, seed, i = args
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  if str(DATAGEN) not in sys.path:
    sys.path.insert(0, str(DATAGEN))
  from dataclasses import replace

  from generate import pinch_width_from_spec
  from r2s.paths import load_task_spec
  from r2s.planner import plan_config_for
  from r2s.variations import sample_variation
  from r2s.verify_policy import run_expert_judged

  spec = load_task_spec(SPEC)
  v = sample_variation(spec, seed, i, variety=True)
  cfg = plan_config_for(v)
  foh = True
  if group == "zoh":
    foh = False
  elif group == "no_close":
    cfg = replace(cfg, close_jaw=False)
  elif group == "high_rel":
    cfg = replace(cfg, release_gap=0.06)
  t0 = time.perf_counter()
  res, pf = run_expert_judged(spec, v, cfg, servo_foh=foh, pinch_width=pinch_width_from_spec(SPEC),
                              keep_arrays=False)
  return {"group": group, "attempt": i, "expert_pass": bool(res.accepted),
          "expert_reasons": res.reasons, "pf_pass": bool(pf["passed"]), "pf_L_task": bool(pf["L_task"]),
          "pf_reasons": pf["reasons"], "pf_stats": pf["stats"], "expert_stats": {
            k: res.stats.get(k) for k in ("object_shift_before_close_mm", "max_obj_lift_mm",
                                          "in_hand_slip_mm", "carry_foreign_kinds")},
          "wall_s": round(time.perf_counter() - t0, 2)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, default=60)
  ap.add_argument("--zoh", type=int, default=20)
  ap.add_argument("--neg", type=int, default=10, help="episodes for no_close and high_rel")
  ap.add_argument("--seed", type=int, default=1000)
  ap.add_argument("--workers", type=int, default=4)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  jobs = ([("expert", a.seed, i) for i in range(a.n)] + [("zoh", a.seed, i) for i in range(a.zoh)]
          + [("no_close", a.seed, i) for i in range(a.neg)] + [("high_rel", a.seed, i) for i in range(a.neg)])
  t0 = time.perf_counter()
  rows = []
  with mp.get_context("spawn").Pool(a.workers) as pool:
    for r in pool.imap_unordered(_job, jobs):
      rows.append(r)
      print(f"[{len(rows)}/{len(jobs)}] {r['group']} {r['attempt']}: expert={r['expert_pass']} "
            f"{r['expert_reasons']} pf={r['pf_pass']} {r['pf_reasons']}", flush=True)
  rows.sort(key=lambda r: (r["group"], r["attempt"]))
  summary = {}
  for g in ("expert", "zoh", "no_close", "high_rel"):
    sub = [r for r in rows if r["group"] == g]
    if not sub:
      continue
    agree = sum(r["expert_pass"] == r["pf_pass"] for r in sub)
    summary[g] = {
      "n": len(sub), "expert_pass": sum(r["expert_pass"] for r in sub),
      "pf_pass": sum(r["pf_pass"] for r in sub), "pf_L_task": sum(r["pf_L_task"] for r in sub),
      "agree": agree,
      "disagreements": [{k: r[k] for k in ("attempt", "expert_pass", "expert_reasons", "pf_pass",
                                           "pf_reasons")} for r in sub if r["expert_pass"] != r["pf_pass"]],
      "pf_reason_counts": dict(Counter(x for r in sub for x in r["pf_reasons"])),
      "expert_reason_counts": dict(Counter(x for r in sub for x in r["expert_reasons"])),
    }
  out = {"spec": str(SPEC), "seed": a.seed, "variety": True, "wall_s": round(time.perf_counter() - t0, 1),
         "workers": a.workers, "summary": summary, "episodes": rows}
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  Path(a.out).write_text(json.dumps(out, indent=1))
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
