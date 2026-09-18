"""
Wheelo Ratings -- primary external Brownlow benchmark.

Source: local CSV data/external/wheelo-brownlow-predictions.csv (9,522
player-match rows, 2026 home-and-away season). This is match-level EV-type
data (a `Votes` predicted-votes-equivalent field per player per match, plus a
`Votes3_Probability` 0-100 P(3)-equivalent), the richest external source
available, per the project brief's instruction to give Wheelo deeper
integration than the remote sources.

Round reconciliation: Wheelo's own `Round` column was independently verified
against this app's official normalized round (src/data/round_normalization.py)
for two real, distinct 2026 matches by team-pair + round number -- "Syd v
Carl" at Wheelo Round 0 == this app's official Round 0 (2026_R0_sydney_v_
carlton_2026-03-05); "GWS v Carl" at Wheelo Round 15 == this app's official
Round 15 (2026_R15_greater_western_sydney_v_carlton_2026-06-20). Wheelo's
Round column already uses the AFL's official numbering (Opening Round = 0),
matching this app's normalized convention directly -- NO shift is applied.
This is a verified finding from real data, not an assumption.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.external.identity import WHEELO_TEAM_MAP, resolve_external_match_players, resolve_external_players

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "external" / "wheelo-brownlow-predictions.csv"
PROCESSED_DIR = ROOT / "data" / "external" / "processed"


def load_wheelo_match_level() -> pd.DataFrame:
    """Match-level Wheelo predictions, identity-resolved to canonical
    player_id using per-round disambiguation (see
    src.external.identity.resolve_external_match_players)."""
    raw = pd.read_csv(RAW_CSV)
    raw["team_id"] = raw["Team"].map(WHEELO_TEAM_MAP)
    unmapped = raw[raw["team_id"].isna()]
    if not unmapped.empty:
        raise ValueError(f"Unmapped Wheelo team codes: {sorted(unmapped['Team'].unique())}")

    resolved = resolve_external_match_players(raw, name_col="Player", team_col="team_id", round_col="Round")
    resolved = resolved.rename(columns={
        "Round": "round", "Match": "match_label", "Player": "wheelo_player_name",
        "Votes": "wheelo_ev", "Votes3_Probability": "wheelo_p3_pct",
    })
    return resolved[[
        "round", "match_label", "wheelo_player_name", "team_id", "player_id", "match_status",
        "wheelo_ev", "wheelo_p3_pct", "Rank",
    ]].rename(columns={"Rank": "wheelo_match_rank"})


def load_wheelo_season() -> pd.DataFrame:
    """Season-total Wheelo EV/rank per resolved player_id. Season EV is the
    sum of the player's own match-level `wheelo_ev` values -- summed ONLY
    over rows that actually resolved to a player_id (an unresolved/ambiguous
    row contributes nothing to anyone's total rather than being silently
    dropped from an implicit denominator, since it was never attributed to a
    player in the first place)."""
    match_level = load_wheelo_match_level()
    resolved = match_level[match_level["match_status"] == "resolved"]
    season = (
        resolved.groupby("player_id", as_index=False)
        .agg(
            wheelo_ev=("wheelo_ev", "sum"),
            wheelo_player_name=("wheelo_player_name", "first"),
            team_id=("team_id", "first"),
            n_matches=("wheelo_ev", "size"),
        )
    )
    season["wheelo_rank"] = season["wheelo_ev"].rank(ascending=False, method="min")
    return season.sort_values("wheelo_rank").reset_index(drop=True)


def identity_status_summary() -> dict:
    """Provenance-report-friendly summary of match-level identity resolution
    quality -- how many rows resolved / were ambiguous / were unresolved, and
    which distinct external player names fell into each non-resolved bucket."""
    match_level = load_wheelo_match_level()
    counts = match_level["match_status"].value_counts().to_dict()
    unresolved_names = sorted(
        match_level.loc[match_level["match_status"] == "unresolved", "wheelo_player_name"].unique().tolist()
    )
    ambiguous_names = sorted(
        match_level.loc[match_level["match_status"] == "ambiguous", "wheelo_player_name"].unique().tolist()
    )
    return {
        "n_rows": len(match_level),
        "status_counts": counts,
        "n_unresolved_names": len(unresolved_names),
        "unresolved_names": unresolved_names,
        "n_ambiguous_names": len(ambiguous_names),
        "ambiguous_names": ambiguous_names,
    }
