"""
Phase 3, section H/Q9: role/position effects on Brownlow voting.

Uses the `role` column from build_role_proxy.py (real official position for
2021-2025, a validated ~77%-accurate statistical proxy for 1999-2020 -- see
docs/ROLE_ANALYSIS.md for the validation detail). 1984-1998 has no role
classification (see build_role_proxy.py) and is excluded here.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def build() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "analytical_features_v1.parquet")
    df = df.dropna(subset=["role"])

    g = df.groupby("role")
    summary = pd.DataFrame({
        "n_player_matches": g.size(),
        "votes_per_game": g["brownlow_votes"].mean(),
        "pct_polling_any": g.apply(lambda x: (x["brownlow_votes"] > 0).mean(), include_groups=False),
        "pct_3_votes": g.apply(lambda x: (x["brownlow_votes"] == 3).mean(), include_groups=False),
        "pct_2_votes": g.apply(lambda x: (x["brownlow_votes"] == 2).mean(), include_groups=False),
        "pct_1_vote": g.apply(lambda x: (x["brownlow_votes"] == 1).mean(), include_groups=False),
        "mean_disposals": g["disposals"].mean(),
        "mean_contested_possessions": g["contested_possessions"].mean(),
        "mean_goals": g["goals"].mean(),
        "mean_clearances": g["clearances"].mean(),
        "mean_hitouts": g["hitouts"].mean(),
        "share_of_all_3_votes": g.apply(lambda x: (x["brownlow_votes"] == 3).sum(), include_groups=False) / (df["brownlow_votes"] == 3).sum(),
    }).sort_values("votes_per_game", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(REPORTS_DIR / "role_vote_summary.csv")
    print(f"Wrote role summary ({len(summary)} roles) -> reports/role_vote_summary.csv")
    return summary


if __name__ == "__main__":
    s = build()
    print(s.to_string())
