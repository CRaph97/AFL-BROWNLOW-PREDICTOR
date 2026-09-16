"""
Phase 4, section N: error analysis by segment, using the pooled Model 1
out-of-sample predictions (data/processed/oos_predictions_core.parquet).

DESIGN NOTE / bug fixed during Phase 4: `match_level_metrics()` needs every
player in a match present to correctly determine the actual/predicted 3-2-1
(it looks up "the" 3-vote getter, "the" 2-vote getter, etc., and computes
predictions by ranking ALL players in that match). Filtering the input to
match_level_metrics() by a PLAYER-level attribute (role, disposals>=25, ...)
before grouping by match_id silently breaks nearly every match's completeness
(a match doesn't have one "role" -- its 22-44 players do), and the very
first version of this script did exactly that, crashing when a segment's
filtered rows produced zero valid matches (KeyError on an empty 'correct_3'
column).

Fixed approach: compute match-level correctness ONCE on the FULL (unfiltered)
prediction set, then attach that correctness back onto the row belonging to
the ACTUAL 3-vote getter (and separately the actual 2-vote and 1-vote
getters) in each match, and segment THOSE rows by role/season/margin/etc.
This answers a well-posed question -- e.g. "when the actual 3-vote winner was
a key defender, how often did the model correctly predict it was them" --
rather than a broken one.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def per_match_correctness(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per match: whether the model's top pick for 3/2/1 matched the
    actual outcome. Computed on the FULL, unfiltered set of players in each
    match (per the module docstring)."""
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

        rows.append({
            "match_id": match_id,
            "actual_3_player": actual_3, "actual_2_player": actual_2, "actual_1_player": actual_1,
            "correct_3": pred_3 == actual_3, "correct_2": pred_2 == actual_2, "correct_1": pred_1 == actual_1,
            "exact_321": (pred_3 == actual_3) and (pred_2 == actual_2) and (pred_1 == actual_1),
        })
    return pd.DataFrame(rows)


def segment_summary(df: pd.DataFrame, segment_col: str, correctness_col: str) -> pd.DataFrame:
    g = df.groupby(segment_col)
    out = g.agg(n=(correctness_col, "size"), accuracy=(correctness_col, "mean")).reset_index()
    return out[out["n"] >= 5]


def run():
    preds = pd.read_parquet(PROCESSED_DIR / "oos_predictions_core.parquet")
    pl_preds = preds[preds["model"] == "Model1_PlackettLuce"].copy()

    core = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    context_cols = ["match_id", "player_id", "role", "win_loss_draw", "absolute_margin", "disposals", "goals"]
    pl_preds = pl_preds.merge(core[context_cols], on=["match_id", "player_id"], how="left")

    match_correct = per_match_correctness(pl_preds)
    print(f"Match-level correctness computed for {len(match_correct):,} matches")

    # attach the ACTUAL 3-vote getter's own row attributes (role, margin, disposals, goals, season)
    winner_rows = pl_preds.merge(
        match_correct[["match_id", "actual_3_player", "correct_3", "correct_2", "correct_1", "exact_321"]],
        left_on=["match_id", "player_id"], right_on=["match_id", "actual_3_player"], how="inner",
    )
    winner_rows["margin_bucket"] = pd.cut(winner_rows["absolute_margin"], [0, 12, 24, 48, 100, 200],
                                           labels=["0-12", "12-24", "24-48", "48-100", "100+"])
    winner_rows["is_high_disposal"] = winner_rows["disposals"] >= 25
    winner_rows["is_high_goal"] = winner_rows["goals"] >= 3

    all_results = []
    for seg_col in ["season", "role", "win_loss_draw", "margin_bucket", "is_high_disposal", "is_high_goal"]:
        seg = segment_summary(winner_rows, seg_col, "correct_3")
        seg["segment_type"] = seg_col
        seg = seg.rename(columns={seg_col: "segment_value"})
        all_results.append(seg)

    result = pd.concat(all_results, ignore_index=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "error_analysis_by_segment.csv", index=False)

    # surprising misses: actual 3-vote players the model ranked very low in their own match
    pl_preds["match_rank_of_p3"] = pl_preds.groupby("match_id")["p3"].rank(ascending=False)
    actual_3_rows = pl_preds[pl_preds["brownlow_votes"] == 3].copy()
    surprising_misses = actual_3_rows[actual_3_rows["match_rank_of_p3"] > 3].sort_values(
        "match_rank_of_p3", ascending=False
    )
    surprising_misses.to_csv(REPORTS_DIR / "surprising_3vote_misses.csv", index=False)

    # systematically over/under-predicted players (pooled expected votes vs actual, by player)
    by_player = pl_preds.groupby("player_id").agg(
        n_matches=("match_id", "nunique"), mean_expected=("expected_votes", "mean"),
        mean_actual=("brownlow_votes", "mean"),
    )
    by_player = by_player[by_player["n_matches"] >= 20]
    by_player["bias"] = by_player["mean_expected"] - by_player["mean_actual"]
    by_player.sort_values("bias").to_csv(REPORTS_DIR / "player_level_bias.csv")

    print(f"Wrote error_analysis_by_segment.csv ({len(result)} rows), "
          f"surprising_3vote_misses.csv ({len(surprising_misses)} rows), "
          f"player_level_bias.csv ({len(by_player)} players)")
    print("\nRole segment (accuracy of predicting the ACTUAL 3-vote winner, given their role):")
    print(result[result["segment_type"] == "role"][["segment_value", "n", "accuracy"]].to_string(index=False))
    print("\nMargin bucket segment:")
    print(result[result["segment_type"] == "margin_bucket"][["segment_value", "n", "accuracy"]].to_string(index=False))
    return result


if __name__ == "__main__":
    run()
