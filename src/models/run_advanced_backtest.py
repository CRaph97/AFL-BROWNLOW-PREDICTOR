"""
Phase 4, Model 4: repeat the winning CORE architecture (Model 1, Plackett-
Luce -- confirmed by run_backtest.py / docs/MODEL_COMPARISON.md) on the
ADVANCED dataset (2015-2025, richer footywire-sourced stats, shorter
history), and compare against the SAME architecture restricted to the SAME
test seasons but trained/evaluated with only CORE features -- an apples-to-
apples test of the brief's Q9 ("does ADVANCED beat CORE despite having fewer
training seasons?").
"""
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import match_level_metrics
from src.models.plackett_luce import PlackettLuceModel
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]

CORE_FEATURES = (
    feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]
    + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]
    + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]
    + feature_sets.FAMILIES["lagged_form"] + feature_sets.FAMILIES["win_margin_interaction"]
)
ADVANCED_FEATURES = CORE_FEATURES + feature_sets.ADVANCED_FAMILY["advanced_stats"]


def run():
    core_df = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    core_df = feature_sets.prepare_features(core_df, CORE_FEATURES, dropna=True)

    adv_df = pd.read_parquet(PROCESSED_DIR / "model_advanced.parquet")
    adv_df = feature_sets.prepare_features(adv_df, ADVANCED_FEATURES, dropna=True)

    rows = []
    for name, df, feats in [("CORE_same_test_seasons", core_df, CORE_FEATURES),
                              ("ADVANCED", adv_df, ADVANCED_FEATURES)]:
        for test_season in TEST_SEASONS:
            train = df[df["season"] < test_season]
            test = df[df["season"] == test_season]
            if len(train) == 0 or len(test) == 0:
                continue
            model = PlackettLuceModel(feature_names=feats).fit(train)
            preds = model.predict(test)
            m = match_level_metrics(preds)
            rows.append({"dataset": name, "test_season": test_season,
                         "n_train_seasons": train["season"].nunique(), **m})
            print(f"{name} / {test_season}: correct_3={m['correct_3_pct']:.3f} "
                  f"exact_321={m['exact_321_pct']:.3f} log_loss={m['log_loss']:.3f}")

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "advanced_vs_core_comparison.csv", index=False)

    summary = result.groupby("dataset").agg(
        mean_correct_3=("correct_3_pct", "mean"), mean_exact_321=("exact_321_pct", "mean"),
        mean_log_loss=("log_loss", "mean"), mean_rank_corr=("mean_rank_corr", "mean"),
    )
    print("\n=== ADVANCED vs CORE (same test seasons) ===")
    print(summary.to_string())
    return result, summary


if __name__ == "__main__":
    run()
