"""Success probe on frozen V-JEPA features vs a trained last-frame baseline vs shuffled labels.

Question: does a frozen video encoder, plus a tiny classifier, tell a successful sim
episode from a failed one better than a classifier that only sees the LAST frame?
If not, video adds nothing a single picture of the end state could not, and "JEPA lost".

Contenders (identical recipe and splits; only the input features differ):
  probe          V-JEPA 2 on 16 frames spread over the episode, mean of the 8 time slots
                 (1024 numbers) -> standardise -> logistic regression (L2)
  last_vjepa     V-JEPA 2 on the last frame repeated 16 times (no motion), same pooling
  last_dino      DINOv2-small CLS + mean patch token of the last frame (768 numbers)
  shuffled       the probe's recipe trained on randomly permuted training labels (20 times)
Secondary (reported, not gated): the 8 time slots concatenated (8192 numbers, keeps the
order of events), a 1-hidden-layer MLP on the probe features, the wrist camera, front+wrist,
and DINOv2 on the wrist camera's last frame (the wrist variant's own last-frame baseline).

Recipe: the regularisation strength C is picked from a fixed grid by 5-fold cross-
validation on the TRAIN split, folds grouped by scene, scored by AUC. class_weight is
"balanced" so a 0.5 probability threshold is a sensible operating point. The model is then
refit on all of train and scored once on test and cam_test.

Gate (FINAL_DESIGN section 4, fixed before the results): on the seed-held-out test split,
AUC(probe) - AUC(best last-frame baseline) >= 0.02, with the paired DeLong test
significant (p < 0.05). Otherwise "JEPA lost". (Both are reported for the y_full label,
the verifier's verdict; y_task, the visible task outcome, is reported alongside.)

  python jepa/probe.py   ->  /work/outputs_large/jepa/probe_results.json + printed tables
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa import stats as S  # noqa: E402

OUT = Path("/work/outputs_large/jepa")
C_GRID = np.logspace(-4, 1, 11)
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GATE = 0.02
warnings.filterwarnings("ignore")


def logistic(C):
  return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                            max_iter=5000))


def mlp(alpha):
  return make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=alpha, max_iter=500,
                                                       early_stopping=True, random_state=0))


def pick_and_fit(make, grid, X, y, groups):
  """Grouped 5-fold CV on train to pick the hyper-parameter, then refit on all of train."""
  best, best_auc = None, -1.0
  for h in grid:
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(X, y, groups):
      oof[va] = make(h).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    a = S.auc(oof, y)
    if a > best_auc:
      best, best_auc = h, a
  return make(best).fit(X, y), best, best_auc


def load():
  eps = json.loads((OUT / "judge_set.json").read_text())
  f = np.load(OUT / "feats_sim.npz")
  pos = {k: i for i, k in enumerate(f["ids"])}
  idx = np.array([pos[e["id"]] for e in eps])
  fu = f["front_uniform16"][idx].astype(np.float32)
  fw = f["wrist_uniform16"][idx].astype(np.float32)
  feats = {
    "probe": fu.mean(1),
    "last_vjepa": f["front_last"][idx].astype(np.float32).mean(1),
    "last_dino": f["dino_last"][idx].astype(np.float32),
    "probe_time_concat": fu.reshape(len(fu), -1),
    "wrist": fw.mean(1),
    "front_plus_wrist": np.concatenate([fu.mean(1), fw.mean(1)], 1),
  }
  wl = OUT / "feats_wrist_last.npz"
  if wl.exists():
    g = np.load(wl)
    gpos = {k: i for i, k in enumerate(g["ids"])}
    feats["last_dino_wrist"] = g["dino_last"][[gpos[e["id"]] for e in eps]].astype(np.float32)
  return eps, feats


def per_type(score, y, ftype, thr=0.5) -> dict:
  """AUC of each failure type's negatives against all positives in the same split, and
  the share of that type caught (score < thr) at the fixed 0.5 operating point."""
  out = {}
  pos = y == 1
  for t in sorted(set(ftype[~pos])):
    neg = ftype == t
    m = pos | neg
    out[t] = {"n_neg": int(neg.sum()), "auc": round(S.auc(score[m], y[m]), 4),
              "caught": round(float(np.mean(score[neg] < thr)), 3)}
  out["positive"] = {"n": int(pos.sum()), "passed": round(float(np.mean(score[pos] >= thr)), 3)}
  return out


def evaluate(label: str, eps, feats) -> dict:
  y_all = np.array([e[label] for e in eps])
  split = np.array([e["split"] for e in eps])
  scene = np.array([e["scene"] for e in eps])
  ftype = np.array([e["failure_type"] if not e[label] else "positive" for e in eps])
  if label == "y_task":  # failure types are only defined for y_full; tag task failures by source
    ftype = np.array(["positive" if e["y_task"] else e["source"] for e in eps])
  tr = split == "train"
  res = {"label": label, "n": {s: {"pos": int(((split == s) & (y_all == 1)).sum()),
                                   "neg": int(((split == s) & (y_all == 0)).sum())}
                               for s in ("train", "test", "cam_test")}, "models": {}}
  scores = {}
  for name, X in feats.items():
    model, C, cv_auc = pick_and_fit(logistic, C_GRID, X[tr], y_all[tr], scene[tr])
    scores[name] = model.predict_proba(X)[:, 1]
    res["models"][name] = {"C": float(C), "train_cv_auc": round(cv_auc, 4)}
  model, alpha, cv_auc = pick_and_fit(mlp, ALPHA_GRID, feats["probe"][tr], y_all[tr], scene[tr])
  scores["probe_mlp"] = model.predict_proba(feats["probe"])[:, 1]
  res["models"]["probe_mlp"] = {"alpha": alpha, "train_cv_auc": round(cv_auc, 4)}
  # Shuffled-label control: same recipe, labels permuted within train.
  rng = np.random.default_rng(0)
  shuf = {"test": [], "cam_test": []}
  for _ in range(20):
    yp = rng.permutation(y_all[tr])
    m, _, _ = pick_and_fit(logistic, C_GRID[::2], feats["probe"][tr], yp, scene[tr])
    s = m.predict_proba(feats["probe"])[:, 1]
    for sp in shuf:
      shuf[sp].append(S.auc(s[split == sp], y_all[split == sp]))
  res["shuffled"] = {sp: {"mean": round(float(np.mean(v)), 4), "min": round(float(np.min(v)), 4),
                          "max": round(float(np.max(v)), 4)} for sp, v in shuf.items()}

  for sp in ("test", "cam_test"):
    m = split == sp
    y = y_all[m]
    r = res[sp] = {"auc": {}, "per_type": {}}
    for name, s in scores.items():
      a, lo, hi = S.auc_ci(s[m], y)
      r["auc"][name] = [round(a, 4), round(lo, 4), round(hi, 4)]
    best_last = max(("last_vjepa", "last_dino"), key=lambda k: r["auc"][k][0])
    r["best_last_frame"] = best_last
    r["delong_probe_vs"] = {b: S.delong_paired(scores["probe"][m], scores[b][m], y)
                            for b in ("last_vjepa", "last_dino")}
    # Secondary (NOT gated; chosen among several variants, so treat as hypotheses):
    r["secondary"] = {
      "probe_time_concat - " + best_last: S.delong_paired(scores["probe_time_concat"][m],
                                                          scores[best_last][m], y)}
    if "last_dino_wrist" in scores:
      r["secondary"]["wrist - last_dino_wrist"] = S.delong_paired(
        scores["wrist"][m], scores["last_dino_wrist"][m], y)
    r["bootstrap_probe_vs_best_last"] = S.cluster_bootstrap_diff(
      scores["probe"][m], scores[best_last][m], y, scene[m])
    for name in [n for n in ("probe", "probe_time_concat", "last_vjepa", "last_dino", "wrist",
                              "last_dino_wrist") if n in scores]:
      r["per_type"][name] = per_type(scores[name][m], y, ftype[m])
    d = r["delong_probe_vs"][best_last]
    r["gate"] = {"diff": round(d["diff"], 4), "p": d["p"],
                 "pass": bool(d["diff"] >= GATE and d["p"] < 0.05 and d["diff"] > 0)}
  np.savez(OUT / f"probe_scores_{label}.npz", ids=np.array([e["id"] for e in eps]),
           split=split, y=y_all, **scores)
  return res


def show(res: dict) -> None:
  print(f"\n===== label {res['label']}  sizes {res['n']}")
  for sp in ("test", "cam_test"):
    r = res[sp]
    print(f"-- {sp}: AUC [95% DeLong CI]")
    for k, v in r["auc"].items():
      print(f"   {k:20s} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    print(f"   shuffled (20x)       mean {res['shuffled'][sp]['mean']:.3f} "
          f"range [{res['shuffled'][sp]['min']:.3f}, {res['shuffled'][sp]['max']:.3f}]")
    for b, d in r["delong_probe_vs"].items():
      print(f"   probe - {b}: {d['diff']:+.4f} CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
            f"p={d['p']:.3g}")
    for k, d in r["secondary"].items():
      print(f"   [secondary] {k}: {d['diff']:+.4f} CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
            f"p={d['p']:.3g}")
    print(f"   scene bootstrap probe - {r['best_last_frame']}: {r['bootstrap_probe_vs_best_last']}")
    print(f"   GATE: {r['gate']}")
    names = [n for n in ("probe", "last_vjepa", "last_dino", "probe_time_concat", "wrist",
                         "last_dino_wrist") if n in r["per_type"]]
    print(f"   per type ({' | '.join(names)}): auc / caught")
    for t in r["per_type"]["probe"]:
      if t == "positive":
        continue
      row = " | ".join(f"{r['per_type'][n][t]['auc']:.3f}/{r['per_type'][n][t]['caught']:.2f}"
                       for n in names)
      print(f"   {t:15s} n={r['per_type']['probe'][t]['n_neg']:3d}  {row}")


def main() -> None:
  eps, feats = load()
  results = {lab: evaluate(lab, eps, feats) for lab in ("y_full", "y_task")}
  (OUT / "probe_results.json").write_text(json.dumps(results, indent=1))
  for r in results.values():
    show(r)


if __name__ == "__main__":
  main()
