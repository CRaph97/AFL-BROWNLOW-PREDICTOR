"""
Phase 3, section E: teammate competition / "vote stealing" features.

Thresholds (25/30 disposals, 2/3 goals) are EXPLORATORY choices flagged in the
brief as not sacred -- they are round-number, commonly-cited benchmarks in AFL
media commentary, used here as a starting point for comparison against the
continuous alternatives (gap_best_team etc. from build_relative_features.py),
not as a claim that they are the "correct" cutoffs.
"""
import numpy as np
import pandas as pd

DISPOSAL_THRESHOLDS = [25, 30]
GOAL_THRESHOLDS = [2, 3]


def build(core: pd.DataFrame) -> pd.DataFrame:
    df = core[["match_id", "team_id", "disposals", "goals"]].copy()
    out = pd.DataFrame(index=df.index)

    grp_key = ["match_id", "team_id"]

    for t in DISPOSAL_THRESHOLDS:
        above = (df["disposals"] >= t).astype(int)
        team_count_incl_self = df.assign(_a=above).groupby(grp_key)["_a"].transform("sum")
        out[f"n_teammates_disposals_ge_{t}"] = team_count_incl_self - above  # excludes self

    for t in GOAL_THRESHOLDS:
        above = (df["goals"] >= t).astype(int)
        team_count_incl_self = df.assign(_a=above).groupby(grp_key)["_a"].transform("sum")
        out[f"n_teammates_goals_ge_{t}"] = team_count_incl_self - above

    # team concentration of disposal output (Herfindahl-style index of team_share^2, computed here
    # independently rather than depending on build_relative_features' team_share, to keep this
    # module standalone)
    team_total = df.groupby(grp_key)["disposals"].transform("sum")
    share = np.where(team_total > 0, df["disposals"] / team_total, np.nan)
    out["team_disposal_share_sq"] = share ** 2
    out["team_disposal_concentration"] = df.assign(_s2=out["team_disposal_share_sq"]).groupby(grp_key)["_s2"].transform("sum")
    # ^ higher = fewer players doing most of the damage (more "1-2 stars"); lower = evenly spread

    return out


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    core = pd.read_parquet(ROOT / "data" / "processed" / "player_match_core_1984_2025.parquet")
    feats = build(core)
    print(feats.describe().T)
