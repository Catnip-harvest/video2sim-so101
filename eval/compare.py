"""Paired comparison of two eval runs on the same seed list: exact McNemar test + Wilson CIs.

  python eval/compare.py RUN_A RUN_B [--metric L_full|L_task] [--json OUT.json]

RUN_* are sim_eval.py output dirs (episodes.jsonl) or episodes.jsonl files. Episodes are
paired by (seed, attempt); only ids present in both runs are used (and counted).
  b = A pass & B fail,  c = A fail & B pass  (the discordant pairs)
  McNemar exact two-sided p = min(1, 2 * P(X <= min(b, c))), X ~ Binomial(b + c, 0.5)
  mid-p variant also reported (less conservative; Fagerland et al. 2013).
  difference B - A = (c - b) / n with a Wald CI from the paired variance (Newcombe-style
  intervals are better for tiny n; with n = 400 the Wald interval is adequate).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n == 0:
    return (0.0, 1.0)
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (max(0.0, c - h), min(1.0, c + h))


def _binom_cdf(k: int, n: int) -> float:
  return sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n


def mcnemar(b: int, c: int) -> dict:
  n = b + c
  if n == 0:
    return {"p_exact": 1.0, "p_mid": 1.0}
  k = min(b, c)
  p = min(1.0, 2 * _binom_cdf(k, n))
  pmid = min(1.0, 2 * (_binom_cdf(k, n) - 0.5 * math.comb(n, k) / 2 ** n))
  return {"p_exact": p, "p_mid": pmid}


def load(run: str) -> dict:
  p = Path(run)
  f = p / "episodes.jsonl" if p.is_dir() else p
  out = {}
  for line in f.read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      out[(r.get("seed"), r.get("attempt", r.get("episode_index")))] = r
  return out


def compare(a: dict, b: dict, metric: str = "L_full") -> dict:
  keys = sorted(set(a) & set(b), key=lambda k: (k[0] or 0, k[1]))
  n = len(keys)
  xa = [bool(a[k][metric]) for k in keys]
  xb = [bool(b[k][metric]) for k in keys]
  ka, kb = sum(xa), sum(xb)
  bb = sum(1 for i in range(n) if xa[i] and not xb[i])
  cc = sum(1 for i in range(n) if xb[i] and not xa[i])
  d = (cc - bb) / n if n else 0.0
  var = ((bb + cc) / n - d * d) / n if n else 0.0
  se = math.sqrt(max(var, 0.0))
  return {"metric": metric, "n_paired": n, "only_in_a": len(set(a) - set(b)), "only_in_b": len(set(b) - set(a)),
          "a_success": ka, "a_rate": round(ka / n, 4) if n else None, "a_wilson95": [round(x, 4) for x in wilson(ka, n)],
          "b_success": kb, "b_rate": round(kb / n, 4) if n else None, "b_wilson95": [round(x, 4) for x in wilson(kb, n)],
          "a_pass_b_fail": bb, "a_fail_b_pass": cc, "diff_b_minus_a": round(d, 4),
          "diff_wald95": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)], **mcnemar(bb, cc)}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run_a")
  ap.add_argument("run_b")
  ap.add_argument("--metric", default="both", choices=["L_full", "L_task", "both"])
  ap.add_argument("--json", default=None)
  x = ap.parse_args()
  a, b = load(x.run_a), load(x.run_b)
  metrics = ["L_full", "L_task"] if x.metric == "both" else [x.metric]
  res = {"a": x.run_a, "b": x.run_b, "results": [compare(a, b, m) for m in metrics]}
  print(json.dumps(res, indent=1))
  if x.json:
    Path(x.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
