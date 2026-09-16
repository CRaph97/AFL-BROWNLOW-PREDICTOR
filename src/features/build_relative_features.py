"""
Phase 3, section C: match-relative and team-relative features.

Deliberately NOT applied to every available stat (would cause uncontrolled
feature explosion per the Phase 3 brief). Applied to a curated set of 8
statistics chosen for plausible Brownlow relevance and low redundancy with
each other: disposals, contested_possessions, clearances, tackles, goals,
inside_50s, contested_marks, marks. See docs/FEATURE_REGISTRY.md for the
full exclusion rationale (e.g. handballs/kicks excluded as largely redundant
with disposals; one_percenters/bounces excluded as low-salience and sparse;
hitouts handled separately in the role-proxy work since it's positionally
extreme rather than generally informative).

Grain preserved: one row per player-match. Every output column is prefixed
with the base stat name.

Definitions (per player-match row, for a given base stat `x`):
  {x}_match_rank       - rank of this player's x among all players in the match (1 = highest, ties averaged)
  {x}_match_pct        - percentile rank within the match, in (0, 1]
  {x}_match_z          - (x - match_mean) / match_std (match_std==0 -> 0)
  {x}_team_rank        - rank within the player's own team only
  {x}_team_share       - x / sum(x for the player's team), NaN if team total is 0
  {x}_gap_best_team    - (this player's x) - (best teammate's x, excluding self); 0 for the best teammate, negative for everyone else
  {x}_gap_2nd_team     - (this player's x) - (second-best teammate's x, excluding self)
  {x}_opp_best_gap     - (this player's x) - (best player's x on the OPPOSING team)

All match/team groupings use match_id / (match_id, team_id) from the canonical
CORE table. NaN inputs propagate as NaN (never zero-filled) so that
structurally-unavailable eras (e.g. clearances before 1998) stay honestly
missing rather than looking like real zeros.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

RELATIVE_STATS = [
    "disposals", "contested_possessions", "clearances", "tackles",
    "goals", "inside_50s", "contested_marks", "marks",
]


def _match_rank(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="average")


def _best_excl_self(s: pd.Series) -> pd.Series:
    """For each row, the max of the *other* rows in the group (NaN-safe, excludes self)."""
    arr = s.to_numpy(dtype=float)
    n = len(arr)
    if n <= 1:
        return pd.Series(np.nan, index=s.index)
    order = np.argsort(-np.nan_to_num(arr, nan=-np.inf))
    out = np.full(n, np.nan)
    top1_idx, top2_idx = order[0], order[1] if n > 1 else order[0]
    for i in range(n):
        if i == top1_idx:
            out[i] = arr[top2_idx] if not np.isnan(arr[top2_idx]) else np.nan
        else:
            out[i] = arr[top1_idx] if not np.isnan(arr[top1_idx]) else np.nan
    return pd.Series(out, index=s.index)


def _second_best_excl_self(s: pd.Series) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    n = len(arr)
    if n <= 2:
        return pd.Series(np.nan, index=s.index)
    order = np.argsort(-np.nan_to_num(arr, nan=-np.inf))
    out = np.full(n, np.nan)
    for i in range(n):
        others_order = [o for o in order if o != i]
        out[i] = arr[others_order[1]] if len(others_order) > 1 and not np.isnan(arr[others_order[1]]) else np.nan
    return pd.Series(out, index=s.index)


def build(core: pd.DataFrame) -> pd.DataFrame:
    df = core[["match_id", "team_id", "opponent_id"] + RELATIVE_STATS].copy()
    out = pd.DataFrame(index=df.index)

    for stat in RELATIVE_STATS:
        g_match = df.groupby("match_id")[stat]
        g_team = df.groupby(["match_id", "team_id"])[stat]

        out[f"{stat}_match_rank"] = g_match.transform(_match_rank)
        match_size = g_match.transform("size")
        out[f"{stat}_match_pct"] = 1 - (out[f"{stat}_match_rank"] - 1) / match_size.clip(lower=1)

        match_mean = g_match.transform("mean")
        match_std = g_match.transform("std").replace(0, np.nan)
        out[f"{stat}_match_z"] = (df[stat] - match_mean) / match_std

        out[f"{stat}_team_rank"] = g_team.transform(_match_rank)
        team_total = g_team.transform("sum")
        out[f"{stat}_team_share"] = np.where(team_total > 0, df[stat] / team_total, np.nan)

        best_team = g_team.transform(_best_excl_self)
        second_team = g_team.transform(_second_best_excl_self)
        out[f"{stat}_gap_best_team"] = df[stat] - best_team
        out[f"{stat}_gap_2nd_team"] = df[stat] - second_team

        out[f"{stat}_opp_best_gap"] = df[stat] - _opponent_best(df, stat)

    return out


def _opponent_best(df: pd.DataFrame, stat: str) -> pd.Series:
    """Best value of `stat` on the OTHER team in the same match, broadcast to every row.
    Vectorised: each match has exactly two teams, so a self-join of per-(match,team) maxima
    on match_id where team_id differs gives the opponent's max directly."""
    team_max = df.groupby(["match_id", "team_id"])[stat].max().rename("team_max").reset_index()
    self_join = team_max.merge(team_max, on="match_id", suffixes=("", "_other"))
    self_join = self_join[self_join["team_id"] != self_join["team_id_other"]]
    opp_lookup = self_join.set_index(["match_id", "team_id"])["team_max_other"]
    keys = pd.MultiIndex.from_arrays([df["match_id"], df["team_id"]])
    return opp_lookup.reindex(keys).to_numpy()


if __name__ == "__main__":
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    feats = build(core)
    print(feats.describe().T[["count", "mean", "std", "min", "max"]])
