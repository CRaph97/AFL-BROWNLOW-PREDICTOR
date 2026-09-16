"""
Produces reports/match_prediction_metrics.csv -- the per-match (not
season-aggregated) prediction outcome for the primary model (Model 1,
Plackett-Luce, expanding window), from the pooled out-of-sample predictions
already written by run_backtest.py. One row per match: whether the 3/2/1 were
correctly identified, rank correlation, etc. -- the match-level granularity
that reports/model_metrics_by_season.csv (already season-aggregated) doesn't
show.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def run(model_name: str = "Model1_PlackettLuce"):
    preds = pd.read_parquet(PROCESSED_DIR / "oos_predictions_core.parquet")
    preds = preds[preds["model"] == model_name]

    rows = []
    for match_id, g in preds.groupby("match_id"):
        actual_3 = g.loc[g["brownlow_votes"] == 3, "player_id"]
        actual_2 = g.loc[g["brownlow_votes"] == 2, "player_id"]
        actual_1 = g.loc[g["brownlow_votes"] == 1, "player_id"]
        if len(actual_3) != 1 or len(actual_2) != 1 or len(actual_1) != 1:
            continue
        actual_3, actual_2, actual_1 = actual_3.iloc[0], actual_2.iloc[0], actual_1.iloc[0]

        pred_3 = g.loc[g["p3"].idxmax(), "player_id"]
        remaining = g[g["player_id"] != pred_3]
        pred_2 = remaining.loc[remaining["p2"].idxmax(), "player_id"] if len(remaining) else None
        remaining2 = remaining[remaining["player_id"] != pred_2]
        pred_1 = remaining2.loc[remaining2["p1"].idxmax(), "player_id"] if len(remaining2) else None

        rank_corr = np.nan
        if g["expected_votes"].nunique() > 1:
            rank_corr, _ = spearmanr(g["expected_votes"], g["brownlow_votes"])

        rows.append({
            "match_id": match_id, "season": g["season"].iloc[0],
            "actual_3": actual_3, "actual_2": actual_2, "actual_1": actual_1,
            "predicted_3": pred_3, "predicted_2": pred_2, "predicted_1": pred_1,
            "correct_3": pred_3 == actual_3, "correct_2": pred_2 == actual_2, "correct_1": pred_1 == actual_1,
            "exact_321": (pred_3 == actual_3) and (pred_2 == actual_2) and (pred_1 == actual_1),
            "top3_precision": len(set(g.nlargest(3, "expected_votes")["player_id"]) & {actual_3, actual_2, actual_1}) / 3,
            "rank_corr": rank_corr,
            "p3_of_actual_3": g.loc[g["player_id"] == actual_3, "p3"].iloc[0],
        })

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "match_prediction_metrics.csv", index=False)
    print(f"Wrote {len(result):,} match rows -> reports/match_prediction_metrics.csv")
    return result


if __name__ == "__main__":
    run()
