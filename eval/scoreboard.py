#!/usr/bin/env python
"""Build SCOREBOARD.md from eval run directories (P4 item 1). Never type the table by hand.

Plain words: every eval run writes `episodes.jsonl` (one line per episode, with
seed, attempt, L_full, L_task, reasons, ...). This script reads those files,
counts successes, puts a 95 % Wilson interval on each rate, and runs exact
McNemar tests on episodes that two runs share (joined on (seed, attempt)), with
a Holm correction across all comparisons in one table.

Usage (host or container; /work and /root/work paths are both accepted):

    python eval/scoreboard.py                      # manifest eval/scoreboard_manifest.json
    python eval/scoreboard.py --manifest M.json --out /root/work/SCOREBOARD.md

Manifest format (JSON):

    {"tables": [{"title": "...",
                 "rows": [{"model": "MolmoAct2", "params": "5.4B", "hardware": "MI300X",
                           "test": "in-range s3000",       # label shown in the Test column
                           "rung": "in-range",             # comparison key; s4000 rungs: none/camera/...
                           "runs": ["/work/outputs_large/eval/policy_final_n200_seed3000"],
                           "expected_n": 200,              # optional, marks partial runs
                           "wrong_object": "n/a", "named_cube": "n/a",
                           "exec_horizon": "auto",         # auto = config.json replan_every
                           "train_h": "8.9", "train_vram": "146 GB",
                           "ms_chunk": "auto",             # auto = med/p95 over episodes' chunk medians
                           "infer_vram": "12-14 GB",
                           "compare": true,                # false = shown, but not in paired tests
                           "hide_if_missing": false}]}]}   # true = row omitted until a run exists

Every cell not given (or with no data yet) is "-". Several run dirs in one row are
concatenated (e.g. a run resumed in two parts); duplicate (seed, attempt) keeps the last.

Comparisons (per table, one Holm family; L_full and L_task each count as one test):
  * model pairs: rows with the same rung and seed, different model, both compare=true;
  * drop vs none: each rung row vs the same model's `none` row on seed 4000.
Both are joined on the common (seed, attempt) ids; n is the size of that join.

Named-cube tables (P3-A-lite; table key `"kind": "named_cube"`): each episode row carries
`named_cube = {named, outcome: named|other|none|multiple, in_tray: {colour: bool}, named_in_tray}`.
In swap runs that name a non-red cube, the row's pass / L_task / L_full describe the planner's red
cube, so this path never reads them: L_task cell = named cube in the tray (`named_in_tray`, same
as `L_task_named`), L_full cell = `L_full_named` only if the harness writes it (else "-").
Extra row keys:
    "join": "run_key"            # swap runs: key (seed, run_key="idx:colour"); default (seed, attempt)
    "paired_metrics": ["named", "named_in_tray"]   # which metrics enter the paired tests
    "multitask": "swap"          # checked against config.json `multitask`
Paired tests use rows with the same rung + seed, different model, both compare=true and both
listing the metric; Holm runs over that table's tests only (its own family).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

Z = 1.96
COLUMNS = ["Model", "Params", "Hardware", "Test", "L_full", "L_task", "n", "Wrong-object",
           "Named-cube (n/o/none/mult)", "Exec horizon", "Train h", "Train VRAM alloc/reserved",
           "MI300X ms/chunk med/p95 (not laptop)", "Infer VRAM"]
HARNESS_KEYS = ["interp", "smooth", "replan_every", "max_sim_s", "quality", "tail_s", "obs_codec"]
FOOTER = [
  "- Wilson: `(p + z²/2n ± z·sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)`, z = 1.96.",
  "- **Paired comparison** (same seed set, joined on (seed, idx)): b = A passes and B fails, c = the "
  "reverse. Exact McNemar p = two-sided binomial test of b against Binomial(b+c, 0.5). Reported for "
  "L_task and L_full for every model pair on each seed set, every rung vs the same model's `none` run, "
  "and named-cube outcome on the swap runs. **Claim a difference only if p < 0.05 after Holm correction "
  "across the comparisons in that table; otherwise write \"not distinguishable at n=…\".** Overlapping "
  "intervals are not a test.",
  "- \"All results in simulation.\" \"Memory caps emulate the PyTorch allocator only, not context "
  "overhead or laptop speed.\"",
  "- Memory-cap wording: \"fits under an emulated 5.0 GiB PyTorch allocator cap (proxy for a 6 GB laptop "
  "card; context and driver overhead on CUDA not measured)\".",
]


# ----------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
  """Wilson score 95 % interval for k successes out of n (brief section 3 formula)."""
  if n <= 0:
    return (float("nan"), float("nan"))
  p = k / n
  den = 1 + z * z / n
  mid = p + z * z / (2 * n)
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
  return (max(0.0, (mid - half) / den), min(1.0, (mid + half) / den))


def mcnemar_exact(b: int, c: int) -> float:
  """Two-sided exact McNemar p: binomial test of b against Binomial(b+c, 0.5)."""
  n = b + c
  if n == 0:
    return 1.0
  k = min(b, c)
  tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
  return min(1.0, 2 * tail)


def holm(pvals: list[float]) -> list[float]:
  """Holm step-down adjusted p-values, returned in the input order."""
  m = len(pvals)
  order = sorted(range(m), key=lambda i: pvals[i])
  adj = [0.0] * m
  running = 0.0
  for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * pvals[i]))
    adj[i] = running
  return adj


def cell(k: int, n: int) -> str:
  lo, hi = wilson(k, n)
  return f"{k}/{n} ({100 * lo:.1f}-{100 * hi:.1f}%)"


# ----------------------------------------------------------------------------- loading

def resolve(p: str) -> Path:
  """Accept container (/work/...) and host (/root/work/...) spellings of the same path."""
  cand = [Path(p)]
  if p.startswith("/work/"):
    cand.append(Path("/root" + p))
  elif p.startswith("/root/work/"):
    cand.append(Path(p[len("/root"):]))
  for c in cand:
    if c.exists():
      return c
  return cand[0]


def load_run(dirs: list[str], join: str = "seed_attempt") -> dict | None:
  """Episodes keyed by (seed, attempt), plus the first config/summary found. None if no data.

  join="run_key" keys by (seed, run_key) instead: a swap run plays each scene id once per named
  colour, so (seed, attempt) repeats 3x there and would collapse 60 runs into 20.
  """
  eps: dict[tuple, dict] = {}
  configs, summaries, found, bad_lines = [], [], [], 0
  for d in dirs:
    dp = resolve(d)
    f = dp / "episodes.jsonl"
    if not f.is_file():
      continue
    found.append(str(dp))
    for line in f.read_text().splitlines():
      if not line.strip():
        continue
      try:
        r = json.loads(line)
      except json.JSONDecodeError:  # a half-written last line of a live run
        bad_lines += 1
        continue
      if join == "run_key":
        eps[(int(r["seed"]), str(r["run_key"]))] = r
      else:
        eps[(int(r["seed"]), int(r["attempt"]))] = r
    for name, acc in (("config.json", configs), ("summary.json", summaries)):
      q = dp / name
      if q.is_file():
        try:
          acc.append(json.loads(q.read_text()))
        except json.JSONDecodeError:
          pass
  if not eps:
    return None
  return {"eps": eps, "dirs": found, "configs": configs, "summaries": summaries, "bad_lines": bad_lines}


def ms_chunk_auto(eps: dict) -> str:
  meds = sorted(e["inference_ms"]["chunk_median"] for e in eps.values()
                if isinstance(e.get("inference_ms"), dict) and e["inference_ms"].get("chunk_median") is not None)
  if not meds:
    return "-"
  p95 = meds[min(len(meds) - 1, max(0, math.ceil(0.95 * len(meds)) - 1))]
  return f"{statistics.median(meds):.0f}/{p95:.0f}"


def run_seed(run: dict) -> str:
  seeds = sorted({k[0] for k in run["eps"]})
  return ",".join(map(str, seeds))


# ----------------------------------------------------------------------------- building

def build_rows(table: dict, checks: list[str]) -> list[dict]:
  rows = []
  for spec in table["rows"]:
    run = load_run(spec.get("runs", []))
    if run is None and spec.get("hide_if_missing"):
      continue
    row = {"spec": spec, "run": run}
    if run is not None:
      eps = run["eps"]
      n = len(eps)
      row["n"] = n
      row["kf"] = sum(bool(e.get("L_full")) for e in eps.values())
      row["kt"] = sum(bool(e.get("L_task")) for e in eps.values())
      row["seed"] = run_seed(run)
      cfg = run["configs"][0] if run["configs"] else {}
      row["cfg"] = cfg
      cfg_rung = cfg.get("ood")
      want = spec.get("rung")
      if cfg_rung and want and want != "in-range" and cfg_rung != want:
        checks.append(f"{spec['model']} / {spec['test']}: manifest rung `{want}` but config.json ood=`{cfg_rung}`")
      if run["bad_lines"]:
        checks.append(f"{spec['model']} / {spec['test']}: skipped {run['bad_lines']} unparsable line(s)")
      exp = spec.get("expected_n")
      if exp and n < exp:
        checks.append(f"{spec['model']} / {spec['test']}: partial, {n} of {exp} episodes so far")
      if len(run["dirs"]) == 1 and run["summaries"]:
        s = run["summaries"][0]
        for key, k in (("success_L_full", row["kf"]), ("success_L_task", row["kt"])):
          if key in s and s.get("n") == n and s[key] != k:
            checks.append(f"{spec['model']} / {spec['test']}: summary.json {key}={s[key]} but episodes.jsonl gives {k}")
    rows.append(row)
  return rows


def static(spec: dict, key: str) -> str:
  v = spec.get(key)
  return "-" if v in (None, "") else str(v)


def render_row(row: dict) -> list[str]:
  s, run = row["spec"], row["run"]
  if run is None:
    lf = lt = n = "-"
    ms = "-" if s.get("ms_chunk") == "auto" else static(s, "ms_chunk")
    eh = "-" if s.get("exec_horizon") == "auto" else static(s, "exec_horizon")
  else:
    lf, lt = cell(row["kf"], row["n"]), cell(row["kt"], row["n"])
    exp = s.get("expected_n")
    n = f"{row['n']}" + (f" (partial, of {exp})" if exp and row["n"] < exp else "")
    ms = ms_chunk_auto(run["eps"]) if s.get("ms_chunk") == "auto" else static(s, "ms_chunk")
    if s.get("exec_horizon") == "auto":
      re = row["cfg"].get("replan_every")
      eh = "-" if re is None or row["cfg"].get("mode") == "expert" else str(re)
    else:
      eh = static(s, "exec_horizon")
  return [static(s, "model"), static(s, "params"), static(s, "hardware"), static(s, "test"), lf, lt, n,
          static(s, "wrong_object"), static(s, "named_cube"), eh, static(s, "train_h"),
          static(s, "train_vram"), ms, static(s, "infer_vram")]


def paired(a: dict, b: dict, metric: str, fn=None) -> dict:
  """McNemar counts on the shared episode keys. fn(episode) -> bool; default bool(e[metric])."""
  if fn is None:
    fn = lambda e: bool(e.get(metric))  # noqa: E731
  ea, eb = a["run"]["eps"], b["run"]["eps"]
  keys = sorted(set(ea) & set(eb))
  bb = sum(fn(ea[k]) and not fn(eb[k]) for k in keys)
  cc = sum(not fn(ea[k]) and fn(eb[k]) for k in keys)
  ka = sum(fn(ea[k]) for k in keys)
  kb = sum(fn(eb[k]) for k in keys)
  seed_mismatch = sum(1 for k in keys if ea[k].get("episode_seed") != eb[k].get("episode_seed"))
  return {"n": len(keys), "b": bb, "c": cc, "ka": ka, "kb": kb, "p": mcnemar_exact(bb, cc),
          "seed_mismatch": seed_mismatch}


def label(row: dict) -> str:
  return f"{row['spec']['model']} [{row['spec']['test']}]"


def comparisons(rows: list[dict]) -> list[dict]:
  live = [r for r in rows if r["run"] is not None and r["spec"].get("compare", True)]
  out = []
  for a, b in combinations(live, 2):
    if a["spec"]["model"] == b["spec"]["model"]:
      continue
    if a["spec"].get("rung") != b["spec"].get("rung") or a["seed"] != b["seed"]:
      continue
    for m in ("L_full", "L_task"):
      r = paired(a, b, m)
      if r["n"]:
        out.append(dict(r, kind="pair", a_label=label(a), b_label=label(b), metric=m))
  nones = {r["spec"]["model"]: r for r in live if r["spec"].get("rung") == "none" and r["seed"] == "4000"}
  for r in live:
    rung = r["spec"].get("rung")
    if rung in (None, "none", "in-range") or r["seed"] != "4000" or r["spec"]["model"] not in nones:
      continue
    base = nones[r["spec"]["model"]]
    for m in ("L_full", "L_task"):
      p = paired(base, r, m)  # A = none, B = rung: b = passes on none, fails on the rung
      if p["n"]:
        out.append(dict(p, kind="drop", a_label=label(base), b_label=label(r), metric=m, model=r["spec"]["model"], rung=rung))
  if out:
    for c, adj in zip(out, holm([c["p"] for c in out])):
      c["p_holm"] = adj
  return out


def verdict(c: dict) -> str:
  if c["p_holm"] < 0.05:
    if c["kind"] == "drop":
      return "significant drop" if c["b"] > c["c"] else "significant gain"
    return f"{'A' if c['b'] > c['c'] else 'B'} better (significant)"
  return f"not distinguishable at n={c['n']}"


def fmt_p(p: float) -> str:
  return f"{p:.2g}" if p < 0.001 else f"{p:.4f}"


def render(manifest: dict, manifest_path: str) -> str:
  now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
  L = ["# SCOREBOARD", "",
       f"Generated by `eval/scoreboard.py` from `{manifest_path}` at {now}. Do not edit by hand; "
       "add rows to the manifest and re-run.", ""]
  for t in manifest["tables"]:
    if t.get("kind") == "named_cube":
      L += render_named_table(t)
      continue
    checks: list[str] = []
    rows = build_rows(t, checks)
    L += [f"## {t['title']}", ""]
    if t.get("note"):
      L += [t["note"], ""]
    L.append("| " + " | ".join(COLUMNS) + " |")
    L.append("|" + "---|" * len(COLUMNS))
    for r in rows:
      L.append("| " + " | ".join(render_row(r)) + " |")
    L.append("")
    L += FOOTER + [""]

    comps = comparisons(rows)
    m = len(comps)
    L += ["### Paired comparisons", "",
          f"Exact McNemar on episodes both runs share (joined on (seed, attempt)); Holm across all {m} "
          "tests in this table (model pairs and drops together).", ""]
    pairs = [c for c in comps if c["kind"] == "pair"]
    if pairs:
      L += ["| A | B | Metric | n | A pass | B pass | b (A only) | c (B only) | p | p Holm | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
      for c in pairs:
        L.append(f"| {c['a_label']} | {c['b_label']} | {c['metric']} | {c['n']} | {c['ka']} | {c['kb']} | {c['b']} | "
                 f"{c['c']} | {fmt_p(c['p'])} | {fmt_p(c['p_holm'])} | {verdict(c)} |")
    else:
      L.append("No model pair shares a seed set and rung yet.")
    L += ["", "### Drop vs the same model's `none` run on seed 4000", ""]
    drops = [c for c in comps if c["kind"] == "drop"]
    if drops:
      L += ["| Model | Rung | Metric | n | none pass | rung pass | drop (pp) | b (none only) | c (rung only) "
            "| p | p Holm | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
      for c in drops:
        drop = 100 * (c["ka"] - c["kb"]) / c["n"]
        L.append(f"| {c['model']} | {c['rung']} | {c['metric']} | {c['n']} | {c['ka']} | {c['kb']} | "
                 f"{drop:+.1f} | {c['b']} | {c['c']} | {fmt_p(c['p'])} | {fmt_p(c['p_holm'])} | {verdict(c)} |")
    else:
      L.append("No rung run has a matching `none` run of the same model yet.")
    mism = [c for c in comps if c["seed_mismatch"]]
    for c in mism:
      checks.append(f"{c['a_label']} vs {c['b_label']}: {c['seed_mismatch']} joined episodes have different episode_seed")

    L += ["", "### Sources and harness flags", "",
          "| Row | Run dir(s) | Seed | n | " + " | ".join(HARNESS_KEYS) + " |",
          "|---|---|---|---|" + "---|" * len(HARNESS_KEYS)]
    flags0 = None
    for r in rows:
      if r["run"] is None:
        continue
      cfg = r["cfg"]
      flags = [str(cfg.get(k, "-")) for k in HARNESS_KEYS]
      if r["spec"].get("compare", True) and cfg.get("mode") == "policy":
        if flags0 is None:
          flags0 = flags
        elif flags != flags0:
          checks.append(f"{label(r)}: harness flags differ from the first policy row: "
                        + ", ".join(f"{k}={v}" for k, v, v0 in zip(HARNESS_KEYS, flags, flags0) if v != v0))
      dirs = "<br>".join(f"`{d}`" for d in r["run"]["dirs"])
      L.append(f"| {label(r)} | {dirs} | {r['seed']} | {r['n']} | " + " | ".join(flags) + " |")
    L += ["", "### Data checks", ""]
    L += [f"- {c}" for c in checks] if checks else ["- none"]
    L += [""]
  return "\n".join(L)


# ----------------------------------------------------------------------------- named cube (P3-A-lite)

OUTCOMES = ("named", "other", "none", "multiple")
NAMED_METRICS = {  # metric name -> (label in the paired table, per-episode bool)
  "named": ("named cube moved (outcome = named)", lambda e: nc_outcome(e) == "named"),
  "named_in_tray": ("named cube in tray", lambda e: nc_in_tray(e)),
}
P3_FOOTER = [
  "- Wilson: `(p + z²/2n ± z·sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)`, z = 1.96.",
  "- A cube \"moved\" if displaced >20 mm or lifted >10 mm. Outcome per run: **named** (only the named "
  "cube moved) / **other** (only another cube moved) / **none** / **multiple**. Named-cube rate = named / all "
  "runs. Rows are reported separately, never pooled (they reuse seed-5000 scenes).",
  "- **L_task here = the named cube ends in the tray** (`named_in_tray`), not the row's `L_task`: in swap "
  "runs naming a non-red cube, `pass`/`L_task`/`L_full` describe the planner's red cube. L_full is \"-\" "
  "because no named-cube full verdict exists yet. Wrong-object = a cube other than the named one ends in the tray.",
  "- **Paired comparison**: exact McNemar (two-sided binomial test of b against Binomial(b+c, 0.5)); held-out "
  "rows joined on (seed, attempt), swap rows on (seed, run_key = \"idx:colour\"). **Holm across this table's "
  "tests only** (the single-task table is a separate family). Claim a difference only if p Holm < 0.05; "
  "otherwise \"not distinguishable at n=…\". Overlapping intervals are not a test.",
  "- \"All results in simulation.\" P3-A demos come from a scripted expert (the planner), not a human video.",
]


def nc_outcome(e: dict) -> str | None:
  nc = e.get("named_cube")
  return nc.get("outcome") if isinstance(nc, dict) else None


def nc_in_tray(e: dict) -> bool:
  nc = e.get("named_cube")
  if isinstance(nc, dict) and nc.get("named_in_tray") is not None:
    return bool(nc["named_in_tray"])
  return bool(e.get("L_task_named"))


def nc_wrong_in_tray(e: dict) -> bool:
  nc = e.get("named_cube")
  if not isinstance(nc, dict):
    return False
  return any(v for c, v in (nc.get("in_tray") or {}).items() if c != nc.get("named"))


def build_named_rows(table: dict, checks: list[str]) -> list[dict]:
  rows = []
  for spec in table["rows"]:
    run = load_run(spec.get("runs", []), spec.get("join", "seed_attempt"))
    if run is None and spec.get("hide_if_missing"):
      continue
    row = {"spec": spec, "run": run}
    if run is not None:
      eps = list(run["eps"].values())
      n = len(eps)
      tag = f"{spec['model']} / {spec['test']}"
      row["n"] = n
      row["seed"] = run_seed(run)
      row["cfg"] = run["configs"][0] if run["configs"] else {}
      row["outcomes"] = {o: sum(nc_outcome(e) == o for e in eps) for o in OUTCOMES}
      row["in_tray"] = sum(nc_in_tray(e) for e in eps)
      row["wrong"] = sum(nc_wrong_in_tray(e) for e in eps)
      fulls = [e.get("L_full_named") for e in eps]
      row["kf"] = None if all(v is None for v in fulls) else sum(bool(v) for v in fulls)
      missing = sum(nc_outcome(e) not in OUTCOMES for e in eps)
      if missing:
        checks.append(f"{tag}: {missing} episode(s) without a named_cube outcome")
      want = spec.get("multitask")
      got = row["cfg"].get("multitask")
      if want and got and want != got:
        checks.append(f"{tag}: manifest multitask `{want}` but config.json multitask=`{got}`")
      if run["bad_lines"]:
        checks.append(f"{tag}: skipped {run['bad_lines']} unparsable line(s)")
      exp = spec.get("expected_n")
      if exp and n < exp:
        checks.append(f"{tag}: partial, {n} of {exp} episodes so far")
      if len(run["dirs"]) == 1 and run["summaries"]:
        s = (run["summaries"][0] or {}).get("named_cube") or {}
        if s.get("n") == n:
          for o in OUTCOMES:
            if o in s.get("outcomes", {}) and s["outcomes"][o] != row["outcomes"][o]:
              checks.append(f"{tag}: summary.json named_cube {o}={s['outcomes'][o]} but episodes.jsonl gives {row['outcomes'][o]}")
          if "named_in_tray" in s and s["named_in_tray"] != row["in_tray"]:
            checks.append(f"{tag}: summary.json named_in_tray={s['named_in_tray']} but episodes.jsonl gives {row['in_tray']}")
    rows.append(row)
  return rows


def render_named_row(row: dict) -> list[str]:
  s, run = row["spec"], row["run"]
  if run is None:
    lf = lt = n = wrong = "-"
    named = static(s, "named_cube")
    ms = "-" if s.get("ms_chunk") == "auto" else static(s, "ms_chunk")
    eh = "-" if s.get("exec_horizon") == "auto" else static(s, "exec_horizon")
  else:
    o = row["outcomes"]
    lf = "-" if row["kf"] is None else cell(row["kf"], row["n"])
    lt = cell(row["in_tray"], row["n"])
    wrong = cell(row["wrong"], row["n"])
    named = f"{o['named']}/{o['other']}/{o['none']}/{o['multiple']}; named {cell(o['named'], row['n'])}"
    exp = s.get("expected_n")
    n = f"{row['n']}" + (f" (partial, of {exp})" if exp and row["n"] < exp else "")
    ms = ms_chunk_auto(run["eps"]) if s.get("ms_chunk") == "auto" else static(s, "ms_chunk")
    if s.get("exec_horizon") == "auto":
      re = row["cfg"].get("replan_every")
      eh = "-" if re is None or row["cfg"].get("mode") == "expert" else str(re)
    else:
      eh = static(s, "exec_horizon")
  return [static(s, "model"), static(s, "params"), static(s, "hardware"), static(s, "test"), lf, lt, n,
          wrong, named, eh, static(s, "train_h"), static(s, "train_vram"), ms, static(s, "infer_vram")]


def named_comparisons(rows: list[dict], checks: list[str]) -> list[dict]:
  """Model pairs on the same rung + seed set; metrics both rows list in paired_metrics. One Holm family."""
  live = [r for r in rows if r["run"] is not None and r["spec"].get("compare", True)]
  out = []
  for a, b in combinations(live, 2):
    sa, sb = a["spec"], b["spec"]
    if sa["model"] == sb["model"] or sa.get("rung") != sb.get("rung") or a["seed"] != b["seed"]:
      continue
    ja, jb = sa.get("join", "seed_attempt"), sb.get("join", "seed_attempt")
    if ja != jb:
      checks.append(f"{label(a)} vs {label(b)}: different join keys ({ja} vs {jb}); not compared")
      continue
    keys = set(a["run"]["eps"]) & set(b["run"]["eps"])
    bad = sum((a["run"]["eps"][k].get("named_cube") or {}).get("named")
              != (b["run"]["eps"][k].get("named_cube") or {}).get("named") for k in keys)
    if bad:
      checks.append(f"{label(a)} vs {label(b)}: {bad} joined runs name a different cube colour")
    for m in [m for m in sa.get("paired_metrics", []) if m in sb.get("paired_metrics", [])]:
      r = paired(a, b, m, NAMED_METRICS[m][1])
      if r["n"]:
        out.append(dict(r, kind="pair", a_label=label(a), b_label=label(b), metric=NAMED_METRICS[m][0],
                        join="(seed, run_key)" if ja == "run_key" else "(seed, attempt)"))
  if out:
    for c, adj in zip(out, holm([c["p"] for c in out])):
      c["p_holm"] = adj
  return out


def render_named_table(t: dict) -> list[str]:
  checks: list[str] = []
  rows = build_named_rows(t, checks)
  L = [f"## {t['title']}", ""]
  if t.get("note"):
    L += [t["note"], ""]
  L.append("| " + " | ".join(COLUMNS) + " |")
  L.append("|" + "---|" * len(COLUMNS))
  for r in rows:
    L.append("| " + " | ".join(render_named_row(r)) + " |")
  L += [""] + P3_FOOTER + [""]

  comps = named_comparisons(rows, checks)
  L += ["### Paired comparisons (this table only)", "",
        f"Exact McNemar on runs both rows share; Holm across all {len(comps)} tests in this table "
        "(its own family, separate from the single-task table).", ""]
  if comps:
    L += ["| A | B | Metric | Joined on | n | A yes | B yes | b (A only) | c (B only) | p | p Holm | Verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
      L.append(f"| {c['a_label']} | {c['b_label']} | {c['metric']} | {c['join']} | {c['n']} | {c['ka']} | {c['kb']} | "
               f"{c['b']} | {c['c']} | {fmt_p(c['p'])} | {fmt_p(c['p_holm'])} | {verdict(c)} |")
  else:
    L.append("No model pair has runs on the same rung and seed set yet.")
  for c in comps:
    if c["seed_mismatch"]:
      checks.append(f"{c['a_label']} vs {c['b_label']}: {c['seed_mismatch']} joined runs have different episode_seed")

  keys = HARNESS_KEYS + ["multitask"]
  L += ["", "### Sources and harness flags", "",
        "| Row | Run dir(s) | Seed | n | join | " + " | ".join(keys) + " |",
        "|---|---|---|---|---|" + "---|" * len(keys)]
  flags0 = None
  for r in rows:
    if r["run"] is None:
      continue
    cfg = r["cfg"]
    flags = [str(cfg.get(k, "-")) for k in HARNESS_KEYS]
    if r["spec"].get("compare", True) and cfg.get("mode") == "policy":
      if flags0 is None:
        flags0 = flags
      elif flags != flags0:
        checks.append(f"{label(r)}: harness flags differ from the first policy row: "
                      + ", ".join(f"{k}={v}" for k, v, v0 in zip(HARNESS_KEYS, flags, flags0) if v != v0))
    dirs = "<br>".join(f"`{d}`" for d in r["run"]["dirs"])
    L.append(f"| {label(r)} | {dirs} | {r['seed']} | {r['n']} | {r['spec'].get('join', 'seed_attempt')} | "
             + " | ".join(flags + [str(cfg.get('multitask', '-'))]) + " |")
  L += ["", "### Data checks", ""]
  L += [f"- {c}" for c in checks] if checks else ["- none"]
  L += [""]
  return L


def main(argv: list[str] | None = None) -> int:
  here = Path(__file__).resolve().parent
  ap = argparse.ArgumentParser(description="Build SCOREBOARD.md from eval run dirs.")
  ap.add_argument("--manifest", default=str(here / "scoreboard_manifest.json"))
  default_out = "/work/SCOREBOARD.md" if Path("/work").is_dir() else "/root/work/SCOREBOARD.md"
  ap.add_argument("--out", default=default_out)
  a = ap.parse_args(argv)
  manifest = json.loads(Path(a.manifest).read_text())
  text = render(manifest, Path(a.manifest).name)
  Path(a.out).write_text(text)
  print(f"wrote {a.out} ({len(text.splitlines())} lines)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
