"""
Ensemble research: log-linear pooling of component P3 vectors within each
match, with non-negative weights summing to one, LEARNED from out-of-fold
predictions of strictly earlier seasons (walk-forward stacking). The pooled
utility is sum_m w_m * log p3_m; P3/P2/P1 come from the same Plackett-Luce
marginalisation as every component so the 6-votes-per-match structure holds.
Equal-weight pooling and the best single component are the comparators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.plackett_luce import _attach_pl_probabilities


def _stack(oof: dict[str, pd.DataFrame], seasons: list[int]) -> tuple[pd.DataFrame, list[str]]:
    names = sorted(oof)
    base = None
    for n in names:
        f = oof[n][oof[n]["season"].isin(seasons)][["season", "match_id", "player_id", "brownlow_votes", "p3"]].rename(columns={"p3": f"p3__{n}"})
        base = f if base is None else base.merge(f.drop(columns=["season", "brownlow_votes"]), on=["match_id", "player_id"], how="inner")
    return base, names


def pooled_predict(stacked: pd.DataFrame, names: list[str], w: np.ndarray) -> pd.DataFrame:
    logp = np.column_stack([np.log(np.clip(stacked[f"p3__{n}"].to_numpy(), 1e-9, 1)) for n in names])
    u = logp @ w
    out = stacked.copy(); out["_u"] = u
    parts = [_attach_pl_probabilities(g.drop(columns=["_u"]), g["_u"].to_numpy()) for _, g in out.groupby("match_id", sort=False)]
    return pd.concat(parts, axis=0)


def _nll(w: np.ndarray, stacked: pd.DataFrame, names: list[str]) -> float:
    pred = pooled_predict(stacked, names, w)
    a3 = pred[pred["brownlow_votes"] == 3]
    return float(-np.log(np.clip(a3["p3"], 1e-9, 1)).mean())


def fit_weights(oof: dict[str, pd.DataFrame], train_seasons: list[int]) -> tuple[np.ndarray, list[str]]:
    stacked, names = _stack(oof, train_seasons)
    k = len(names)
    def obj(theta):
        w = np.exp(theta) / np.exp(theta).sum()
        return _nll(w, stacked, names)
    res = minimize(obj, np.zeros(k), method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-5})
    w = np.exp(res.x) / np.exp(res.x).sum()
    return w, names


def walk_forward_ensemble(oof: dict[str, pd.DataFrame], test_seasons: list[int], min_train: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (learned-ensemble OOF preds, equal-weight OOF preds, weights by season)."""
    all_seasons = sorted(set.intersection(*[set(f["season"].unique()) for f in oof.values()]))
    learned, equal, wrows = [], [], []
    names = sorted(oof)
    for t in test_seasons:
        prior = [s for s in all_seasons if s < t]
        if len(prior) < min_train or t not in all_seasons:
            continue
        w, names = fit_weights(oof, prior)
        st, _ = _stack(oof, [t])
        le = pooled_predict(st, names, w); le["season"] = t; learned.append(le)
        eq = pooled_predict(st, names, np.ones(len(names)) / len(names)); eq["season"] = t; equal.append(eq)
        wrows.append({"test_season": t, **{f"w_{n}": float(x) for n, x in zip(names, w)}, "n_train_seasons": len(prior)})
    return pd.concat(learned, ignore_index=True), pd.concat(equal, ignore_index=True), pd.DataFrame(wrows)
