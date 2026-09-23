"""
Exact analytic gradient for the Plackett-Luce (exploded-logit) 3-2-1
likelihood in src/models/plackett_luce.py. The original fit uses L-BFGS with
FINITE-DIFFERENCE gradients, which costs one likelihood evaluation per
feature per iteration -- fine at 68 features, prohibitive at 150-210. This
module computes the same objective (identical penalty l2 * sum(beta^2)) with
its closed-form gradient:

    dNLL/du_i = (s3_i - 1[i = pick3]) + (s2_i - 1[i = pick2]) + (s1_i - 1[i = pick1])
    dNLL/dbeta = X^T dNLL/du + 2 * l2 * beta

where s3/s2/s1 are the within-match softmax shares at each sequential pick
(the 3-voter removed at stage 2, the 3- and 2-voters removed at stage 1).
Equivalence to the legacy objective is verified in tests/test_2027_rd.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.plackett_luce import PlackettLuceModel, _LOG1P_CLIP, _segment_logsumexp


def nll_and_grad(beta, X, match_codes, n_matches, idx_3, idx_2, idx_1, l2):
    u = X @ beta
    if not np.isfinite(u).all():
        return np.inf, np.zeros_like(beta)
    log_d3 = _segment_logsumexp(u, match_codes, n_matches)
    u3, u2, u1 = u[idx_3], u[idx_2], u[idx_1]
    d3 = np.clip(u3 - log_d3, None, _LOG1P_CLIP); log_d2 = log_d3 + np.log1p(-np.exp(d3))
    d2 = np.clip(u2 - log_d2, None, _LOG1P_CLIP); log_d1 = log_d2 + np.log1p(-np.exp(d2))
    nll = -np.sum(u3 - log_d3) - np.sum(u2 - log_d2) - np.sum(u1 - log_d1) + l2 * np.sum(beta ** 2)
    s3 = np.exp(u - log_d3[match_codes])
    s2 = np.exp(u - log_d2[match_codes]); s2[idx_3] = 0.0
    s1 = np.exp(u - log_d1[match_codes]); s1[idx_3] = 0.0; s1[idx_2] = 0.0
    g_u = s3 + s2 + s1
    g_u[idx_3] -= 1.0; g_u[idx_2] -= 1.0; g_u[idx_1] -= 1.0
    grad = X.T @ g_u + 2.0 * l2 * beta
    if not np.isfinite(nll):
        return np.inf, np.zeros_like(beta)
    return nll, grad


def fit_pl_fast(model: PlackettLuceModel, df: pd.DataFrame, match_col: str = "match_id", vote_col: str = "brownlow_votes",
                l2: float = 1.0, maxiter: int = 500) -> PlackettLuceModel:
    """Same preprocessing contract as PlackettLuceModel.fit (complete-match
    filter, train-only scaler), gradient-based optimisation."""
    df = df.sort_values(match_col).reset_index(drop=True)
    vc = df.groupby(match_col)[vote_col].agg(n3=lambda s: (s == 3).sum(), n2=lambda s: (s == 2).sum(), n1=lambda s: (s == 1).sum())
    complete = vc[(vc.n3 == 1) & (vc.n2 == 1) & (vc.n1 == 1)].index
    df = df[df[match_col].isin(complete)].reset_index(drop=True)
    X_raw = df[model.feature_names].to_numpy(dtype=float)
    model._fit_scaler(X_raw)
    X = model._transform(X_raw)
    votes = df[vote_col].to_numpy(dtype=float)
    codes, _ = pd.factorize(df[match_col].to_numpy(), sort=False)
    n_m = codes.max() + 1
    idx_3 = np.full(n_m, -1); idx_2 = np.full(n_m, -1); idx_1 = np.full(n_m, -1)
    idx_3[codes[votes == 3]] = np.where(votes == 3)[0]; idx_2[codes[votes == 2]] = np.where(votes == 2)[0]; idx_1[codes[votes == 1]] = np.where(votes == 1)[0]
    res = minimize(nll_and_grad, np.zeros(X.shape[1]), args=(X, codes, n_m, idx_3, idx_2, idx_1, l2), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    if not np.isfinite(res.x).all():
        raise FloatingPointError("fast PL fit produced non-finite coefficients")
    model.beta = res.x
    model.fit_info = {"nit": int(res.nit), "nll": float(res.fun), "success": bool(res.success)}
    return model
