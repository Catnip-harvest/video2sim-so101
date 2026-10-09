"""AUC statistics: DeLong variance (single and paired), and a scene-cluster bootstrap.

DeLong (1988), fast O(n log n) form of Sun & Xu (2014). For a score s, positives X and
negatives Y, AUC = P(s(x) > s(y)) (ties count 1/2). DeLong writes the AUC as a mean of
per-positive and per-negative "placement" values; their sample variances give the AUC's
variance without resampling. For two scores on the SAME episodes ("paired"), the
covariance of the placements gives the variance of the AUC difference, hence a z-test.

DeLong treats episodes as independent. Several of our episodes share a scene (the same
layout and camera with different physics), so we also bootstrap whole scenes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
  return stats.rankdata(x, method="average")


def _placements(scores: np.ndarray, y: np.ndarray):
  """scores: (k, n) for k models. Returns aucs (k,), V10 (k, m), V01 (k, n_neg)."""
  pos, neg = scores[:, y == 1], scores[:, y == 0]
  m, n = pos.shape[1], neg.shape[1]
  aucs, v10, v01 = [], [], []
  for p, q in zip(pos, neg):
    tx, ty, tz = _midrank(p), _midrank(q), _midrank(np.r_[p, q])
    aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
    v10.append((tz[:m] - tx) / n)
    v01.append(1.0 - (tz[m:] - ty) / m)
  return np.array(aucs), np.array(v10), np.array(v01)


def delong(scores: list[np.ndarray] | np.ndarray, y: np.ndarray):
  """AUCs and their DeLong covariance matrix for one or more scores on the same episodes."""
  s = np.atleast_2d(np.asarray(scores, float))
  aucs, v10, v01 = _placements(s, np.asarray(y))
  cov = np.atleast_2d(np.cov(v10)) / v10.shape[1] + np.atleast_2d(np.cov(v01)) / v01.shape[1]
  return aucs, cov


def auc_ci(score: np.ndarray, y: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
  """AUC with a DeLong normal-approximation CI (clipped to [0, 1])."""
  a, c = delong([score], y)
  z = stats.norm.ppf(0.5 + level / 2)
  se = float(np.sqrt(max(c[0, 0], 0)))
  return float(a[0]), max(0.0, a[0] - z * se), min(1.0, a[0] + z * se)


def delong_paired(s1: np.ndarray, s2: np.ndarray, y: np.ndarray) -> dict:
  """Two-sided paired DeLong test of AUC(s1) - AUC(s2) on the same episodes."""
  a, c = delong([s1, s2], y)
  var = c[0, 0] + c[1, 1] - 2 * c[0, 1]
  d = float(a[0] - a[1])
  se = float(np.sqrt(max(var, 1e-12)))
  return {"auc1": float(a[0]), "auc2": float(a[1]), "diff": d,
          "ci95": [d - 1.96 * se, d + 1.96 * se], "z": d / se,
          "p": float(2 * stats.norm.sf(abs(d / se)))}


def auc(score: np.ndarray, y: np.ndarray) -> float:
  return float(_placements(np.atleast_2d(score), y)[0][0])


def cluster_bootstrap_diff(s1, s2, y, groups, n_boot: int = 2000, seed: int = 0) -> dict:
  """Resample whole scenes with replacement; CI of AUC(s1) - AUC(s2) and of each AUC."""
  rng = np.random.default_rng(seed)
  g = np.asarray(groups)
  uniq = np.unique(g)
  idx_of = {u: np.flatnonzero(g == u) for u in uniq}
  d, a1, a2 = [], [], []
  for _ in range(n_boot):
    idx = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq), replace=True)])
    yy = y[idx]
    if yy.min() == yy.max():
      continue
    x1, x2 = auc(s1[idx], yy), auc(s2[idx], yy)
    a1.append(x1)
    a2.append(x2)
    d.append(x1 - x2)
  q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
  return {"diff_ci95": q(d), "auc1_ci95": q(a1), "auc2_ci95": q(a2),
          "p_diff_le_0": float(np.mean(np.array(d) <= 0)),
          "p_diff_lt_0.02": float(np.mean(np.array(d) < 0.02))}
