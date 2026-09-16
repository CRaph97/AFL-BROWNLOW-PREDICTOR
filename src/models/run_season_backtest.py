"""
Phase 4, sections I/J: season-level backtest and full historical pseudo-live
backtest, using Model 1 (Plackett-Luce, expanding window -- the primary
architecture per the headline comparison in run_backtest.py, confirmed by
docs/MODEL_COMPARISON.md).

For each held-out test season, this literally re-enacts "pretend it is
immediately before that season's Brownlow count": train only on seasons
strictly before it, predict every one of its matches, aggregate expected
votes to season totals per player, then reveal and compare against the real
totals (summed independently from the CORE table's own brownlow_votes column
-- never derived from the model).
"""
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import season_level_metrics
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
    + feature_sets.FAMILIES["lagged_form"] + feature_sets.FAMILIES["win_margin_interaction"]
)


def run():
    df = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    df = feature_sets.prepare_features(df, FEATURES, dropna=True)

    season_rows = []
    player_detail_rows = []

    for test_season in TEST_SEASONS:
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]
        if len(train) == 0 or len(test) == 0:
            continue

        model = PlackettLuceModel(feature_names=FEATURES).fit(train)
        preds = model.predict(test)

        actual_totals = test.groupby(["season", "player_id"])["brownlow_votes"].sum().reset_index()
        actual_totals = actual_totals.rename(columns={"brownlow_votes": "actual_total"})

        metrics, merged = season_level_metrics(preds.assign(season=test_season), actual_totals)
        season_rows.append({"test_season": test_season, "n_train_seasons": train["season"].nunique(), **metrics})

        merged["test_season"] = test_season
        merged["error"] = merged["predicted_total"] - merged["actual_total"]
        merged["actual_rank"] = merged["actual_total"].rank(ascending=False, method="min")
        merged["predicted_rank"] = merged["predicted_total"].rank(ascending=False, method="min")
        player_detail_rows.append(merged)

        print(f"{test_season} (train on {train['season'].nunique()} prior seasons): "
              f"MAE={metrics['season_mae']:.3f} bias={metrics['season_bias']:.3f} "
              f"spearman={metrics['season_spearman']:.3f} top10_incl={metrics['top10_inclusion']:.3f}")

    season_metrics = pd.DataFrame(season_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    season_metrics.to_csv(REPORTS_DIR / "season_prediction_metrics.csv", index=False)

    player_detail = pd.concat(player_detail_rows, ignore_index=True)
    player_detail.to_csv(REPORTS_DIR / "pseudo_live_backtest_detail.csv", index=False)

    print("\n=== Season-level backtest summary ===")
    print(season_metrics.to_string(index=False))

    print("\n=== Largest overpredictions / underpredictions (pooled across test seasons) ===")
    print("Overpredicted (model too high):")
    print(player_detail.nlargest(10, "error")[["test_season", "player_id", "actual_total", "predicted_total", "error"]])
    print("\nUnderpredicted (model too low):")
    print(player_detail.nsmallest(10, "error")[["test_season", "player_id", "actual_total", "predicted_total", "error"]])

    return season_metrics, player_detail


if __name__ == "__main__":
    run()
