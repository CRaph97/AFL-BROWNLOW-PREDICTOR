"""
Phase 5, section 2: recent-history window comparison for the 2026 production
model choice. Extends Phase 4's finding (recent8 beat expanding for Model 1 on
CORE, docs/MODEL_BACKTEST.md) by testing narrower windows (3, 5 seasons) too,
to check whether even MORE recency-weighting helps -- directly relevant to the
2026 structural-break question (older voting behaviour may be less relevant to
a statistically-assisted umpiring process).

Bounded compute: test seasons restricted to 2021-2025 (5 seasons, the most
recent stretch) rather than the full 2015-2025 sweep Phase 4 used for its
headline comparison -- this question is specifically about how much recent
history matters for a NEAR-TERM (2026) prediction, so testing on the most
recent seasons is also the most relevant evidence, and keeps this an
essential-but-bounded addition to Phase 4 rather than a full re-run.

A recency-WEIGHTED variant is also tested: same training window as
"expanding", but each training match is weighted by an exponential decay
exp(-lambda * seasons_ago) in the log-likelihood (implemented via a small
patch to the L2-only optimiser call -- see _fit_weighted below), rather than
truncating older seasons outright.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.evaluation.metrics import match_level_metrics
from src.models.plackett_luce import PlackettLuceModel, _neg_log_likelihood
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
WINDOWS = {"recent3": 3, "recent5": 5, "recent8": 8, "expanding": None}

FULL_FEATURES = (
    feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]
    + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]
    + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]
    + feature_sets.FAMILIES["lagged_form"] + feature_sets.FAMILIES["win_margin_interaction"]
)

RECENCY_HALF_LIFE_SEASONS = 5.0  # exponential decay half-life for the recency-weighted variant


def _neg_weighted_log_likelihood(beta, X, match_codes, n_matches, idx_3, idx_2, idx_1, l2, match_weight):
    """Weighted variant of plackett_luce._neg_log_likelihood: each match's contribution to the
    log-likelihood is scaled by match_weight (per-match, broadcast to that match's 3/2/1 rows).
    Implemented by re-deriving the same log-space computation rather than importing the
    unweighted internals, to keep the weighting explicit and auditable here."""
    from src.models.plackett_luce import _segment_logsumexp, _LOG1P_CLIP
    u = X @ beta
    if not np.isfinite(u).all():
        return np.inf
    log_denom_3 = _segment_logsumexp(u, match_codes, n_matches)
    u3, u2, u1 = u[idx_3], u[idx_2], u[idx_1]
    d3 = np.clip(u3 - log_denom_3, None, _LOG1P_CLIP)
    log_denom_2 = log_denom_3 + np.log1p(-np.exp(d3))
    d2 = np.clip(u2 - log_denom_2, None, _LOG1P_CLIP)
    log_denom_1 = log_denom_2 + np.log1p(-np.exp(d2))
    ll_per_match = (u3 - log_denom_3) + (u2 - log_denom_2) + (u1 - log_denom_1)
    nll = -np.sum(match_weight * ll_per_match)
    if not np.isfinite(nll):
        return np.inf
    return nll + l2 * np.sum(beta ** 2)


def fit_recency_weighted(df: pd.DataFrame, feature_names: list, test_season: int, l2: float = 1.0) -> PlackettLuceModel:
    df = df.sort_values("match_id").reset_index(drop=True)
    vote_counts = df.groupby("match_id")["brownlow_votes"].agg(
        n3=lambda s: (s == 3).sum(), n2=lambda s: (s == 2).sum(), n1=lambda s: (s == 1).sum())
    complete = vote_counts[(vote_counts.n3 == 1) & (vote_counts.n2 == 1) & (vote_counts.n1 == 1)].index
    df = df[df["match_id"].isin(complete)].reset_index(drop=True)

    model = PlackettLuceModel(feature_names=feature_names)
    X_raw = df[feature_names].to_numpy(dtype=float)
    model._fit_scaler(X_raw)
    X = model._transform(X_raw)
    votes = df["brownlow_votes"].to_numpy(dtype=float)

    match_codes, _ = pd.factorize(df["match_id"].to_numpy(), sort=False)
    n_matches = match_codes.max() + 1
    idx_3 = np.full(n_matches, -1, dtype=int)
    idx_2 = np.full(n_matches, -1, dtype=int)
    idx_1 = np.full(n_matches, -1, dtype=int)
    idx_3[match_codes[votes == 3]] = np.where(votes == 3)[0]
    idx_2[match_codes[votes == 2]] = np.where(votes == 2)[0]
    idx_1[match_codes[votes == 1]] = np.where(votes == 1)[0]

    seasons_ago = test_season - df.groupby("match_id")["season"].first().to_numpy()
    decay_lambda = np.log(2) / RECENCY_HALF_LIFE_SEASONS
    match_weight = np.exp(-decay_lambda * seasons_ago)
    match_weight = match_weight * (n_matches / match_weight.sum())  # renormalise so total weight == n_matches

    beta0 = np.zeros(X.shape[1])
    result = minimize(_neg_weighted_log_likelihood, beta0,
                       args=(X, match_codes, n_matches, idx_3, idx_2, idx_1, l2, match_weight),
                       method="L-BFGS-B", options={"maxiter": 300})
    if not np.isfinite(result.x).all():
        raise FloatingPointError("recency-weighted fit produced non-finite coefficients")
    model.beta = result.x
    return model


def run():
    df = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    df = feature_sets.prepare_features(df, FULL_FEATURES, dropna=True)
    all_seasons = sorted(df["season"].unique())

    rows = []
    for test_season in TEST_SEASONS:
        prior = [s for s in all_seasons if s < test_season]
        test = df[df["season"] == test_season]
        if len(test) == 0:
            continue

        for name, window in WINDOWS.items():
            train_seasons = prior if window is None else prior[-window:]
            train = df[df["season"].isin(train_seasons)]
            if len(train) == 0:
                continue
            model = PlackettLuceModel(feature_names=FULL_FEATURES).fit(train)
            preds = model.predict(test)
            m = match_level_metrics(preds)
            rows.append({"window": name, "test_season": test_season, "n_train_seasons": len(train_seasons), **m})
            print(f"{name} / {test_season} (n_train_seasons={len(train_seasons)}): "
                  f"correct_3={m['correct_3_pct']:.3f} log_loss={m['log_loss']:.3f}")

        # recency-weighted variant: full expanding training set, exponentially down-weighted
        model_w = fit_recency_weighted(df[df["season"].isin(prior)], FULL_FEATURES, test_season)
        preds_w = model_w.predict(test)
        m_w = match_level_metrics(preds_w)
        rows.append({"window": "recency_weighted_halflife5", "test_season": test_season,
                     "n_train_seasons": len(prior), **m_w})
        print(f"recency_weighted / {test_season}: correct_3={m_w['correct_3_pct']:.3f} log_loss={m_w['log_loss']:.3f}")

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "window_comparison_2026.csv", index=False)

    summary = result.groupby("window").agg(
        n_seasons=("test_season", "nunique"),
        mean_correct_3=("correct_3_pct", "mean"), mean_exact_321=("exact_321_pct", "mean"),
        mean_log_loss=("log_loss", "mean"), mean_brier=("brier_score", "mean"),
        mean_rank_corr=("mean_rank_corr", "mean"),
    ).reset_index().sort_values("mean_correct_3", ascending=False)
    summary.to_csv(REPORTS_DIR / "window_comparison_2026_summary.csv", index=False)
    print("\n=== Window comparison summary (test seasons 2021-2025) ===")
    print(summary.to_string(index=False))
    return result, summary


if __name__ == "__main__":
    run()
