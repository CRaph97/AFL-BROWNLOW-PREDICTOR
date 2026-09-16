"""
Phase 4, section H/I: match-level and season-level evaluation metrics.

Every function takes a predictions DataFrame with, at minimum, columns:
  match_id, player_id, season, brownlow_votes (actual, 0-3), p3, p2, p1, p0,
  expected_votes
as produced by any of the src/models/*.py model classes' .predict() methods.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from scipy.stats import spearmanr, kendalltau


def match_level_metrics(preds: pd.DataFrame) -> dict:
    """One row of metrics summarising performance across every match in `preds`."""
    results = []
    for match_id, g in preds.groupby("match_id"):
        actual_3 = g.loc[g["brownlow_votes"] == 3, "player_id"]
        actual_2 = g.loc[g["brownlow_votes"] == 2, "player_id"]
        actual_1 = g.loc[g["brownlow_votes"] == 1, "player_id"]
        if len(actual_3) != 1 or len(actual_2) != 1 or len(actual_1) != 1:
            continue  # malformed match (should not happen post Phase 2 validation) -- skip, don't guess
        actual_3, actual_2, actual_1 = actual_3.iloc[0], actual_2.iloc[0], actual_1.iloc[0]
        actual_set = {actual_3, actual_2, actual_1}

        pred_3 = g.loc[g["p3"].idxmax(), "player_id"]
        remaining = g[g["player_id"] != pred_3]
        pred_2 = remaining.loc[remaining["p2"].idxmax(), "player_id"] if len(remaining) else None
        remaining2 = remaining[remaining["player_id"] != pred_2]
        pred_1 = remaining2.loc[remaining2["p1"].idxmax(), "player_id"] if len(remaining2) else None
        pred_set = {pred_3, pred_2, pred_1}

        top3_by_expected = set(g.nlargest(3, "expected_votes")["player_id"])

        rank_corr, _ = spearmanr(g["expected_votes"], g["brownlow_votes"]) if g["expected_votes"].nunique() > 1 else (np.nan, None)

        results.append({
            "match_id": match_id,
            "correct_3": pred_3 == actual_3,
            "correct_2": pred_2 == actual_2,
            "correct_1": pred_1 == actual_1,
            "exact_321": (pred_3 == actual_3) and (pred_2 == actual_2) and (pred_1 == actual_1),
            "all_3_identified": pred_set == actual_set,
            "top3_precision": len(top3_by_expected & actual_set) / 3,
            "top3_recall": len(top3_by_expected & actual_set) / 3,  # equal here since both sets have size 3
            "rank_corr": rank_corr,
        })

    per_match = pd.DataFrame(results)

    y_true_class = preds["brownlow_votes"].to_numpy()
    prob_matrix = preds[["p0", "p1", "p2", "p3"]].to_numpy()
    prob_matrix = np.clip(prob_matrix, 1e-12, 1)
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    try:
        ll = log_loss(y_true_class, prob_matrix, labels=[0, 1, 2, 3])
    except Exception:
        ll = np.nan

    brier = np.mean(np.sum((prob_matrix - pd.get_dummies(y_true_class).reindex(columns=[0, 1, 2, 3], fill_value=0).to_numpy()) ** 2, axis=1))
    mae = np.mean(np.abs(preds["expected_votes"] - preds["brownlow_votes"]))

    return {
        "n_matches": len(per_match),
        "correct_3_pct": per_match["correct_3"].mean(),
        "correct_2_pct": per_match["correct_2"].mean(),
        "correct_1_pct": per_match["correct_1"].mean(),
        "exact_321_pct": per_match["exact_321"].mean(),
        "all_3_identified_pct": per_match["all_3_identified"].mean(),
        "top3_precision": per_match["top3_precision"].mean(),
        "top3_recall": per_match["top3_recall"].mean(),
        "mean_rank_corr": per_match["rank_corr"].mean(),
        "log_loss": ll,
        "brier_score": brier,
        "expected_vote_mae": mae,
    }


def season_level_metrics(preds: pd.DataFrame, actual_season_totals: pd.DataFrame) -> dict:
    """preds: match-level predictions with expected_votes. actual_season_totals: one row
    per (season, player_id) with the TRUE total votes that season (from the CORE table,
    summed independently -- never derived from the model)."""
    pred_totals = preds.groupby(["season", "player_id"])["expected_votes"].sum().reset_index()
    pred_totals = pred_totals.rename(columns={"expected_votes": "predicted_total"})

    merged = pred_totals.merge(actual_season_totals, on=["season", "player_id"], how="outer").fillna(0)

    mae = np.mean(np.abs(merged["predicted_total"] - merged["actual_total"]))
    rmse = np.sqrt(np.mean((merged["predicted_total"] - merged["actual_total"]) ** 2))
    bias = np.mean(merged["predicted_total"] - merged["actual_total"])
    spearman, _ = spearmanr(merged["predicted_total"], merged["actual_total"])
    kendall, _ = kendalltau(merged["predicted_total"], merged["actual_total"])

    def top_n_inclusion(n):
        actual_top = set(merged.nlargest(n, "actual_total")["player_id"])
        pred_top = set(merged.nlargest(n, "predicted_total")["player_id"])
        return len(actual_top & pred_top) / n

    return {
        "n_players": len(merged),
        "season_mae": mae,
        "season_rmse": rmse,
        "season_bias": bias,
        "season_spearman": spearman,
        "season_kendall": kendall,
        "top5_inclusion": top_n_inclusion(5),
        "top10_inclusion": top_n_inclusion(10),
    }, merged


def calibration_table(preds: pd.DataFrame, prob_col: str = "p3", outcome_col: str = "brownlow_votes",
                       outcome_value: int = 3, n_bins: int = 10) -> pd.DataFrame:
    df = preds[[prob_col, outcome_col]].copy()
    df["actual"] = (df[outcome_col] == outcome_value).astype(int)
    df["bin"] = pd.qcut(df[prob_col], q=n_bins, duplicates="drop")
    table = df.groupby("bin", observed=True).agg(
        n=("actual", "size"),
        mean_predicted=(prob_col, "mean"),
        mean_actual=("actual", "mean"),
    ).reset_index()
    table["gap"] = table["mean_predicted"] - table["mean_actual"]
    return table


def expected_calibration_error(preds: pd.DataFrame, prob_col: str = "p3", outcome_col: str = "brownlow_votes",
                                 outcome_value: int = 3, n_bins: int = 10) -> float:
    table = calibration_table(preds, prob_col, outcome_col, outcome_value, n_bins)
    total_n = table["n"].sum()
    return float(np.sum(table["n"] / total_n * np.abs(table["gap"])))
