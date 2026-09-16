"""
Phase 3, section I/Q8: how do Brownlow-vote relationships change over time?

For each season (1999-2025, the window with a full common stat set -- see
docs/DATA_COVERAGE.md), computes:
  - spearman correlation of disposals, contested_possessions, clearances,
    goals, and possession_impact_index with brownlow_votes
  - the winning-team polling-rate advantage: P(polled any | win) - P(polled any | loss)
  - the 3-vote rate for winners vs losers
  - ruck/non-midfielder polling share (role effects over time), where role is available
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TRACKED_STATS = ["disposals", "contested_possessions", "clearances", "goals", "possession_impact_index"]


def build() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "analytical_features_v1.parquet")
    df = df[df["season"] >= 1999]

    rows = []
    for season, grp in df.groupby("season"):
        row = {"season": season, "n_matches": grp["match_id"].nunique()}
        for stat in TRACKED_STATS:
            valid = grp[stat].notna()
            if valid.sum() > 50:
                corr, _ = spearmanr(grp.loc[valid, stat], grp.loc[valid, "brownlow_votes"])
                row[f"{stat}_spearman"] = round(corr, 4)
            else:
                row[f"{stat}_spearman"] = None

        win_poll = grp.loc[grp["is_win"] == 1, "brownlow_votes"].gt(0).mean()
        loss_poll = grp.loc[grp["is_win"] == 0, "brownlow_votes"].gt(0).mean()
        row["winner_polling_rate"] = round(win_poll, 4)
        row["loser_polling_rate"] = round(loss_poll, 4)
        row["winner_advantage_polling"] = round(win_poll - loss_poll, 4)

        win_3 = (grp.loc[grp["is_win"] == 1, "brownlow_votes"] == 3).mean()
        loss_3 = (grp.loc[grp["is_win"] == 0, "brownlow_votes"] == 3).mean()
        row["winner_advantage_3votes"] = round(win_3 - loss_3, 4)

        if grp["role"].notna().any():
            three_votes = grp[grp["brownlow_votes"] == 3]
            if len(three_votes) > 0:
                row["midfielder_share_of_3votes"] = round((three_votes["role"] == "MIDFIELDER").mean(), 4)
                row["ruck_or_defender_share_of_3votes"] = round(
                    three_votes["role"].isin(["RUCK", "KEY_DEFENDER", "MEDIUM_DEFENDER"]).mean(), 4
                )
        rows.append(row)

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "season_vote_relationships.csv", index=False)
    print(f"Wrote {len(result)} seasons -> reports/season_vote_relationships.csv")
    return result


if __name__ == "__main__":
    r = build()
    print(r.to_string(index=False))
