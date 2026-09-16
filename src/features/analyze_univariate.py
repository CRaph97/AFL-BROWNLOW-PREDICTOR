"""
Phase 3, section L/Q1-Q2: univariate relationships between each candidate
feature and Brownlow votes. Purely descriptive -- per the brief, univariate
association is NOT the same as causal importance or final predictive value,
and this script does not claim otherwise.

For every numeric feature, computes:
  - spearman_corr_votes: Spearman rank correlation with brownlow_votes (0-3, ordinal)
  - point_biserial_polled_any: correlation with the binary "received any votes" indicator
  - mean_when_0/1/2/3: mean feature value within each vote category (interpretability)
  - mutual_info_votes: mutual information between a 10-quantile-binned version of the
    feature and the 4-class vote outcome (captures nonlinear/non-monotonic association
    that Spearman would miss)
  - n_obs: non-null observation count (so a thin, era-limited feature's numbers can be
    read in context, not mistaken for full-sample evidence)

Restricted to the 2003-2025 window (the CORE model's recommended era, per
docs/DATA_COVERAGE.md) so results reflect the modern game consistently, not a
blend of eras with very different scoring/statistical environments -- era
differences are handled separately in analyze_temporal.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

ANALYSIS_START_SEASON = 2003
ANALYSIS_END_SEASON = 2025

EXCLUDE_COLS = {
    "season", "player_id", "team_id", "opponent_id", "match_id", "date", "venue",
    "player_name", "round", "home_away", "win_loss_draw", "substitute_status",
    "jumper_number", "brownlow_votes", "role", "is_proxy_position",
}


def build() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "analytical_features_v1.parquet")
    df = df[df["season"].between(ANALYSIS_START_SEASON, ANALYSIS_END_SEASON)].copy()
    votes = df["brownlow_votes"].astype(float)
    polled_any = (votes > 0).astype(int)

    numeric_cols = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for col in numeric_cols:
        x = df[col].astype(float)
        valid = x.notna()
        n_obs = int(valid.sum())
        if n_obs < 100:
            continue
        xv, vv, pv = x[valid], votes[valid], polled_any[valid]

        try:
            sp_corr, _ = spearmanr(xv, vv)
        except Exception:
            sp_corr = np.nan
        try:
            pb_corr, _ = pointbiserialr(pv, xv)
        except Exception:
            pb_corr = np.nan

        means = {f"mean_when_{v}": xv[vv == v].mean() for v in (0, 1, 2, 3)}

        try:
            bins = pd.qcut(xv, q=10, duplicates="drop").cat.codes.to_numpy().reshape(-1, 1)
            mi = mutual_info_classif(bins, vv.to_numpy(), discrete_features=True, random_state=0)[0]
        except Exception:
            mi = np.nan

        rows.append({
            "feature": col,
            "n_obs": n_obs,
            "spearman_corr_votes": round(sp_corr, 4) if sp_corr == sp_corr else np.nan,
            "point_biserial_polled_any": round(pb_corr, 4) if pb_corr == pb_corr else np.nan,
            "mutual_info_votes": round(mi, 4) if mi == mi else np.nan,
            **{k: round(v, 3) for k, v in means.items()},
        })

    result = pd.DataFrame(rows).sort_values("spearman_corr_votes", ascending=False, key=lambda s: s.abs())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "univariate_vote_relationships.csv", index=False)
    print(f"Wrote {len(result)} feature rows -> reports/univariate_vote_relationships.csv")
    return result


if __name__ == "__main__":
    r = build()
    print(r.head(25).to_string(index=False))
