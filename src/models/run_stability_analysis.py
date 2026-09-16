"""
Phase 4, section O: model stability -- do feature effects (Plackett-Luce
coefficients) stay stable across independent expanding-window folds, or does
an important-looking feature swing wildly from one training window to the
next? Refits Model 1 once per test season (2018-2025, expanding window,
reusing the same folds conceptually as run_season_backtest.py) and reports
each feature's coefficient across folds plus its rank-stability (how
consistently it lands in the top-10 by |coefficient|).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.plackett_luce import PlackettLuceModel
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
FEATURES = (
    feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]
    + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]
    + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]
    + feature_sets.FAMILIES["win_margin_interaction"]
)


def run():
    df = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    df = feature_sets.prepare_features(df, FEATURES, dropna=True)

    coef_rows = []
    for test_season in TEST_SEASONS:
        train = df[df["season"] < test_season]
        if len(train) == 0:
            continue
        model = PlackettLuceModel(feature_names=FEATURES).fit(train)
        for feat, coef in zip(FEATURES, model.beta):
            coef_rows.append({"test_season": test_season, "feature": feat, "coefficient": coef})

    coefs = pd.DataFrame(coef_rows)
    pivot = coefs.pivot(index="feature", columns="test_season", values="coefficient")
    pivot["mean_coef"] = pivot.mean(axis=1)
    pivot["std_coef"] = pivot[TEST_SEASONS].std(axis=1)
    pivot["cv_coef"] = pivot["std_coef"] / pivot["mean_coef"].abs().replace(0, np.nan)
    pivot["sign_changes"] = pivot[TEST_SEASONS].apply(lambda r: (np.sign(r) != np.sign(r.iloc[0])).sum(), axis=1)

    # rank stability: for each fold, rank features by |coefficient|; how often is each feature top-10?
    ranks = coefs.copy()
    ranks["abs_coef"] = ranks["coefficient"].abs()
    ranks["rank"] = ranks.groupby("test_season")["abs_coef"].rank(ascending=False)
    top10_rate = ranks.groupby("feature").apply(lambda g: (g["rank"] <= 10).mean(), include_groups=False)
    pivot["top10_rate"] = top10_rate

    pivot = pivot.sort_values("mean_coef", key=lambda s: s.abs(), ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(REPORTS_DIR / "feature_stability.csv")

    print("Most stable, highest-impact features (top 15 by |mean coefficient|):")
    print(pivot.head(15)[["mean_coef", "std_coef", "cv_coef", "sign_changes", "top10_rate"]].to_string())
    print("\nFeatures with sign changes across folds (unstable direction):")
    print(pivot[pivot["sign_changes"] > 0][["mean_coef", "sign_changes"]].to_string())
    return pivot


if __name__ == "__main__":
    run()
