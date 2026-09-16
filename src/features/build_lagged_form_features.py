"""
Phase 4, section A3: strictly lagged player-history ("form") features.

Every feature here uses ONLY games strictly before the current match (shift(1)
before any rolling/expanding calculation) -- never the current match's own
stats. Where a player has insufficient history for a given window, the value
is left NaN (never zero-filled or backfilled with a league average), and a
companion `_n_games` column records exactly how many prior games fed into it,
so a thin (early-career or early-season) estimate is distinguishable from a
robust one downstream.

Windows: previous 3 / 5 / 10 matches (rolling, ACROSS seasons -- career form
does not reset at season boundaries, matching how umpires/media would actually
perceive "recent form"), and season-to-date (resets each season, expanding).
Career-to-date is deliberately NOT built as a separate window: season-to-date
plus the 10-game rolling window already jointly capture short- and
medium-term form without adding a third, highly-collinear long-window feature
that would mostly just track career averages -- an explicit, documented scope
decision per the Phase 3/4 instruction to avoid uncontrolled feature
explosion.

Stats covered: disposals, contested_possessions, clearances, goals (per the
brief's explicit list), plus disposals_match_rank (a match-relative measure,
per "rolling relative-rank measures") and brownlow_votes (rolling vote rate --
built here as a candidate feature, but its use is gated behind the dedicated
reputation experiment in docs/REPUTATION_EXPERIMENT.md, never included in a
model by default).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

ROLLING_WINDOWS = [3, 5, 10]
FORM_STATS = ["disposals", "contested_possessions", "clearances", "goals", "brownlow_votes"]


def build(core: pd.DataFrame, match_rank: pd.Series | None = None) -> pd.DataFrame:
    df = core[["season", "player_id", "date", "match_id"] + FORM_STATS].copy()
    if match_rank is not None:
        df["disposals_match_rank"] = match_rank.reindex(df.index).to_numpy()
        stats = FORM_STATS + ["disposals_match_rank"]
    else:
        stats = FORM_STATS

    df = df.sort_values(["player_id", "date"])
    result = pd.DataFrame(index=df.index)

    g = df.groupby("player_id", sort=False)
    for s in stats:
        for w in ROLLING_WINDOWS:
            result[f"{s}_prev{w}_mean"] = g[s].transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
        result[f"{s}_prevN_count"] = g[s].transform(lambda x: x.shift(1).expanding().count())

    # season-to-date (expanding, resets each season, excludes current match)
    g_season = df.groupby(["player_id", "season"], sort=False)
    for s in stats:
        result[f"{s}_season_to_date_mean"] = g_season[s].transform(lambda x: x.shift(1).expanding().mean())
    result["season_to_date_n_games"] = g_season.cumcount()

    result["match_id"] = df["match_id"]
    result["player_id"] = df["player_id"]
    return result.reindex(core.index)


if __name__ == "__main__":
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    feats = build(core)
    print(feats.shape)
    print(feats.describe().T[["count", "mean", "std"]].head(20))
