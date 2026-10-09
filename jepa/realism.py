"""Realism score: how far our sim front-camera renders are from the friend's real cam_head
clips, measured in frozen V-JEPA 2 feature space (Frechet distance, as in FID/FVD).

Each clip -> one 1024-number vector (16 uniform frames, mean-pooled V-JEPA tokens). Fit a
Gaussian (mean mu, covariance C) to a set of clips. The Frechet distance between two sets:
    FD = |mu1 - mu2|^2 + trace(C1 + C2 - 2 (C1 C2)^(1/2))
0 means the two clouds of feature vectors have the same mean and spread.

FD has no natural unit and is biased upward at small sample sizes, so we only read it
against FLOORS computed with the same number of clips: two random halves of the real set
(real-vs-real) and two halves of the sim set (sim-vs-sim). "Sim looks real to V-JEPA" would
mean FD(sim, real) close to the real-vs-real floor. Every comparison uses n=200 vs 200
clips, repeated over 20 random half-splits (mean and min-max reported). A PCA-64 version
(PCA fitted on all clips) is reported too, because 200 clips cannot pin down a full
1024x1024 covariance. A third, plainer number: the cross-validated AUC of a logistic
classifier told to separate sim from real (0.5 = cannot tell them apart, 1.0 = trivially).

Camera height: the sim set is cut into cam_dz terciles; each tercile's FD to real (at
matched n) shows whether moving the sim camera up or down brings it closer to cam_head.

  python jepa/realism.py  ->  /work/outputs_large/jepa/realism_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

OUT = Path("/work/outputs_large/jepa")
BASE = Path("/work/outputs_large/datagen/sugar_v10_n1000")
N_HALF = 200
N_REP = 20


def frechet(a: np.ndarray, b: np.ndarray) -> float:
  mu1, mu2 = a.mean(0), b.mean(0)
  c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
  covmean = linalg.sqrtm(c1 @ c2)
  if np.iscomplexobj(covmean):
    covmean = covmean.real
  return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))


def summary(v) -> dict:
  return {"mean": round(float(np.mean(v)), 2), "min": round(float(np.min(v)), 2),
          "max": round(float(np.max(v)), 2)}


def main() -> None:
  sim_f = np.load(OUT / "feats_simfd.npz")
  real_f = np.load(OUT / "feats_real.npz")
  sim = sim_f["front_uniform16"].astype(np.float64).mean(1)
  real = real_f["cam_head_uniform16"].astype(np.float64).mean(1)
  rows = {json.loads(l)["attempt"]: json.loads(l) for l in
          (BASE / "attempts.jsonl").read_text().splitlines() if l.strip()}
  cam_dz = np.array([rows[int(i)]["variation"]["cam_dz"] for i in sim_f["ids"]])
  pca = PCA(64, random_state=0).fit(np.concatenate([sim, real]))
  spaces = {"full_1024": (sim, real), "pca_64": (pca.transform(sim), pca.transform(real))}
  rng = np.random.default_rng(0)
  res = {"n_sim": len(sim), "n_real": len(real), "n_per_half": N_HALF, "repeats": N_REP}
  for name, (S, R) in spaces.items():
    rr, ss, sr = [], [], []
    for _ in range(N_REP):
      ps, pr = rng.permutation(len(S)), rng.permutation(len(R))
      sA, sB = S[ps[:N_HALF]], S[ps[N_HALF:2 * N_HALF]]
      rA, rB = R[pr[:N_HALF]], R[pr[N_HALF:2 * N_HALF]]
      rr.append(frechet(rA, rB))
      ss.append(frechet(sA, sB))
      sr.append(0.5 * (frechet(sA, rA) + frechet(sB, rB)))
    res[name] = {"real_vs_real": summary(rr), "sim_vs_sim": summary(ss),
                 "sim_vs_real": summary(sr),
                 "ratio_sim_real_to_real_floor": round(float(np.mean(sr) / np.mean(rr)), 2)}
    # camera-height terciles (matched n = tercile size)
    terc = np.quantile(cam_dz, [1 / 3, 2 / 3])
    bands = {"low": cam_dz < terc[0], "mid": (cam_dz >= terc[0]) & (cam_dz < terc[1]),
             "high": cam_dz >= terc[1]}
    res[name]["cam_dz_terciles_m"] = [round(float(t), 4) for t in terc]
    for b, m in bands.items():
      n = int(m.sum())
      vals, floor = [], []
      for _ in range(N_REP):
        pr = rng.permutation(len(R))
        vals.append(frechet(S[m], R[pr[:n]]))
        floor.append(frechet(R[pr[:n]], R[pr[n:2 * n]]))
      res[name][f"cam_{b}"] = {"n": n, "sim_vs_real": summary(vals), "real_floor": summary(floor)}
  X = np.concatenate([sim, real])
  y = np.r_[np.zeros(len(sim)), np.ones(len(real))]
  p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)),
                        X, y, cv=5, method="predict_proba")[:, 1]
  res["domain_classifier_auc_5fold"] = round(float(roc_auc_score(y, p)), 4)
  (OUT / "realism_results.json").write_text(json.dumps(res, indent=1))
  print(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
