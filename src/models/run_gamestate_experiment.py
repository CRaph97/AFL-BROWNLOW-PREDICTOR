"""
Phase 4, Model 5 / section G8: the controlled experimental game-state test.

Compares the SAME base architecture (Plackett-Luce) WITHOUT vs WITH the
event-derived leverage features (build_experimental_features.py) on the
EXPERIMENTAL dataset (2021-2025, matches with complete 4-period chain
coverage only). This is the brief's explicit "does intra-match timing/
game-state information materially improve Brownlow vote prediction beyond
normal match-level statistics?" question -- answered honestly either way.

Test seasons: 2023, 2024, 2025 (train on all earlier available EXPERIMENTAL
seasons, expanding window) -- the EXPERIMENTAL dataset only spans 2021-2025,
so this is the longest feasible comparable sequence, per the brief's own
allowance for shorter datasets.
"""
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import match_level_metrics
from src.models.plackett_luce import PlackettLuceModel
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_SEASONS = [2023, 2024, 2025]

BASE_FEATURES = (
    feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]
    + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]
    + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]
    + feature_sets.FAMILIES["win_margin_interaction"]
)
GAMESTATE_FEATURES = ["leverage_weighted_disposals", "mean_leverage_disposals",
                       "leverage_weighted_scoring", "n_disposal_events", "n_scoring_events"]


def load_experimental():
    core = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    gamestate = pd.read_parquet(PROCESSED_DIR / "experimental_gamestate_features.parquet")
    df = core.merge(gamestate, on=["season", "match_id", "player_id"], how="inner")  # inner: EXPERIMENTAL scope only
    return df


def run():
    df = load_experimental()
    print(f"EXPERIMENTAL dataset: {len(df):,} rows, seasons {sorted(df['season'].unique())}")

    rows = []
    for variant_name, feats in [("without_gamestate", BASE_FEATURES),
                                 ("with_gamestate", BASE_FEATURES + GAMESTATE_FEATURES)]:
        d = feature_sets.prepare_features(df, feats, dropna=True)
        for test_season in TEST_SEASONS:
            train = d[d["season"] < test_season]
            test = d[d["season"] == test_season]
            if len(train) == 0 or len(test) == 0:
                print(f"  {variant_name}/{test_season}: insufficient data, skipped")
                continue
            model = PlackettLuceModel(feature_names=feats).fit(train)
            preds = model.predict(test)
            m = match_level_metrics(preds)
            rows.append({"variant": variant_name, "test_season": test_season,
                         "n_train": len(train), "n_test": len(test), **m})
            print(f"{variant_name} / {test_season}: n_train={len(train)} n_test={len(test)} "
                  f"correct_3={m['correct_3_pct']:.3f} exact_321={m['exact_321_pct']:.3f} "
                  f"log_loss={m['log_loss']:.3f}")

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "gamestate_experiment.csv", index=False)

    summary = result.groupby("variant").agg(
        mean_correct_3=("correct_3_pct", "mean"), mean_exact_321=("exact_321_pct", "mean"),
        mean_log_loss=("log_loss", "mean"), mean_rank_corr=("mean_rank_corr", "mean"),
    )
    print("\n=== Game-state experiment summary ===")
    print(summary.to_string())
    return result, summary


if __name__ == "__main__":
    run()
