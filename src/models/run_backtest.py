"""
Phase 4, sections C/D/E: the main model-comparison backtest.

Fits Model 0 (benchmark logistic), Model 1 (Plackett-Luce), Model 2a (GBM
utility + Plackett-Luce), Model 2b (GBM multiclass + renormalisation) across
walk-forward (rolling-origin) folds on the CORE dataset, for two training-
window strategies (expanding, recent-8-seasons), for test seasons 2015-2025.

Outputs:
  reports/model_metrics_by_season.csv  -- one row per (model, window, test season)
  reports/model_comparison.csv         -- aggregated across all test seasons, per (model, window)
  data/processed/oos_predictions_core.parquet -- pooled out-of-sample predictions
      (expanding window only, the primary strategy) for every model, for reuse by
      the calibration / season-level / error-analysis / stability scripts.
"""
import time
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import match_level_metrics
from src.evaluation.rolling_origin import generate_folds
from src.models.benchmark_model import BenchmarkModel
from src.models.gbm_model import GBMMulticlassModel, GBMUtilityModel
from src.models.plackett_luce import PlackettLuceModel
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_SEASONS = list(range(2015, 2026))
WINDOW_STRATEGIES = {"expanding": None, "recent8": 8}

FULL_FEATURES = (
    feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]
    + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]
    + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]
    + feature_sets.FAMILIES["lagged_form"] + feature_sets.FAMILIES["win_margin_interaction"]
)

MODEL_BUILDERS = {
    "Model0_Benchmark": lambda feats: BenchmarkModel(feats),
    "Model1_PlackettLuce": lambda feats: PlackettLuceModel(feature_names=feats),
    "Model2a_GBM_utility": lambda feats: GBMUtilityModel(feats),
    "Model2b_GBM_multiclass": lambda feats: GBMMulticlassModel(feats),
}


def load_core():
    df = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    df = feature_sets.prepare_features(df, FULL_FEATURES, dropna=True)
    return df


def run():
    df = load_core()
    all_seasons = sorted(df["season"].unique())
    folds = generate_folds(all_seasons, TEST_SEASONS, WINDOW_STRATEGIES)
    print(f"{len(folds)} folds x {len(MODEL_BUILDERS)} models = {len(folds) * len(MODEL_BUILDERS)} fit/predict cycles")

    season_rows = []
    oos_predictions = []
    t_start = time.time()

    for fold_i, fold in enumerate(folds):
        train_df = df[df["season"].isin(fold.train_seasons)]
        test_df = df[df["season"] == fold.test_season]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        for model_name, builder in MODEL_BUILDERS.items():
            model = builder(FULL_FEATURES)
            try:
                model.fit(train_df)
                preds = model.predict(test_df)
            except Exception as e:
                print(f"  SKIP {model_name} / {fold.window_name} / {fold.test_season}: {e}")
                continue
            metrics = match_level_metrics(preds)
            season_rows.append({
                "model": model_name, "window": fold.window_name, "test_season": fold.test_season,
                "n_train_seasons": len(fold.train_seasons), **metrics,
            })
            if fold.window_name == "expanding":
                keep_cols = ["season", "match_id", "player_id", "brownlow_votes", "p3", "p2", "p1", "p0", "expected_votes"]
                pred_slim = preds[keep_cols].copy()
                pred_slim["model"] = model_name
                oos_predictions.append(pred_slim)

        elapsed = time.time() - t_start
        print(f"[{fold_i+1}/{len(folds)}] {fold.window_name} train<{fold.test_season} "
              f"({len(fold.train_seasons)} seasons) -- {elapsed:.0f}s elapsed")

    season_metrics = pd.DataFrame(season_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    season_metrics.to_csv(REPORTS_DIR / "model_metrics_by_season.csv", index=False)

    comparison = season_metrics.groupby(["model", "window"]).agg(
        n_seasons=("test_season", "nunique"),
        mean_correct_3_pct=("correct_3_pct", "mean"),
        mean_exact_321_pct=("exact_321_pct", "mean"),
        mean_all_3_identified_pct=("all_3_identified_pct", "mean"),
        mean_top3_precision=("top3_precision", "mean"),
        mean_rank_corr=("mean_rank_corr", "mean"),
        mean_log_loss=("log_loss", "mean"),
        mean_brier=("brier_score", "mean"),
        mean_expected_vote_mae=("expected_vote_mae", "mean"),
    ).reset_index().sort_values("mean_correct_3_pct", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    if oos_predictions:
        pooled = pd.concat(oos_predictions, ignore_index=True)
        pooled.to_parquet(PROCESSED_DIR / "oos_predictions_core.parquet", index=False)
        print(f"Wrote pooled OOS predictions: {len(pooled):,} rows -> data/processed/oos_predictions_core.parquet")

    print("\n=== Model comparison (mean across all test seasons) ===")
    print(comparison.to_string(index=False))
    return season_metrics, comparison


if __name__ == "__main__":
    run()
