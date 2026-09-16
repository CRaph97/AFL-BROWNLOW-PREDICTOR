"""
Phase 4, section P: the historical-reputation controlled experiment.

Compares Model 1 (Plackett-Luce) WITHOUT any reputation feature against the
SAME model WITH strictly lagged prior-vote-rate features added
(`brownlow_votes_prev5_mean`, `brownlow_votes_season_to_date_mean` -- both
already excluding the current match, built in build_lagged_form_features.py).
Per the brief: reputation is NOT included in the primary model unless it
shows a consistent, genuine out-of-sample improvement, and if it does, the
result must be reported with its interpretation caveat (the model may be
detecting umpire recognition of a known name, not superior football
performance).
"""
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import match_level_metrics
from src.models.plackett_luce import PlackettLuceModel
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

BASE_FEATURES = (
    feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]
    + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]
    + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]
    + feature_sets.FAMILIES["lagged_form"] + feature_sets.FAMILIES["win_margin_interaction"]
)
WITH_REPUTATION = BASE_FEATURES + feature_sets.FAMILIES["reputation"]


def run():
    df = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    df = feature_sets.prepare_features(df, WITH_REPUTATION, dropna=False)

    rows = []
    for variant_name, feats in [("without_reputation", BASE_FEATURES), ("with_reputation", WITH_REPUTATION)]:
        d = df.dropna(subset=[c for c in feats if c in df.columns])
        for test_season in TEST_SEASONS:
            train = d[d["season"] < test_season]
            test = d[d["season"] == test_season]
            if len(train) == 0 or len(test) == 0:
                continue
            model = PlackettLuceModel(feature_names=feats).fit(train)
            preds = model.predict(test)
            m = match_level_metrics(preds)
            rows.append({"variant": variant_name, "test_season": test_season, **m})
            print(f"{variant_name} / {test_season}: correct_3={m['correct_3_pct']:.3f} "
                  f"exact_321={m['exact_321_pct']:.3f}")

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "reputation_experiment.csv", index=False)

    summary = result.groupby("variant").agg(
        mean_correct_3=("correct_3_pct", "mean"), mean_exact_321=("exact_321_pct", "mean"),
        mean_log_loss=("log_loss", "mean"), mean_rank_corr=("mean_rank_corr", "mean"),
    )
    print("\n=== Reputation experiment summary ===")
    print(summary.to_string())
    return result, summary


if __name__ == "__main__":
    run()
