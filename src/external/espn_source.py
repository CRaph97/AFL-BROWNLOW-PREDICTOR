"""
ESPN Brownlow predictor tracker -- remote source, snapshot-fetched.

The live page (see src/external/refresh.py) requires no JavaScript rendering
for its core table: a single HTML <table> with columns VOTES, PLAYER, TEAM,
OR, R1..R24 -- a real, structured, round-by-round predicted-votes table for a
partial leaderboard (28 players in the 2026-09-18 snapshot; ESPN publishes
only its own top-N, not the full competition). Verified directly against the
raw fetched HTML (not merely a WebFetch-tool summary, which was independently
found to hallucinate a table for a DIFFERENT source in this same session --
see docs/EXTERNAL_BENCHMARKS.md's provenance section for the full account of
why this project verifies every remote source against raw HTML before
trusting it).

"OR" = ESPN's own label for the Opening Round, matching this app's official
round-0 convention by name; R1..R24 are assumed to align 1:1 with official
rounds 1-24 (ESPN does not expose enough per-row context, e.g. an opponent
column per round, to independently verify each round the way Wheelo's Round
column was cross-checked against two real matches -- this is a documented,
reasonable inference from the explicit "OR" label, not an independently
re-verified mapping, and is disclosed as such).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.external.identity import ESPN_BETFAIR_TEAM_MAP, resolve_external_players

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "external" / "raw"

ROUND_COLUMNS = ["OR"] + [f"R{i}" for i in range(1, 25)]
ROUND_LABEL_TO_OFFICIAL = {"OR": 0, **{f"R{i}": i for i in range(1, 25)}}


def parse_espn_html(html: str) -> pd.DataFrame:
    tables = re.findall(r"<table.{0,80000}?</table>", html, re.S)
    if not tables:
        return pd.DataFrame(columns=["player_name", "team_code", "season_votes"] + ROUND_COLUMNS)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tables[0], re.S)
    records = []
    for r in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if not cells or cells[0] == "VOTES":
            continue
        if len(cells) < 3 + len(ROUND_COLUMNS):
            continue
        votes, player, team, *round_vals = cells
        try:
            season_votes = float(votes)
        except ValueError:
            continue
        records.append({
            "player_name": player.title(),
            "team_code": team,
            "season_votes": season_votes,
            **{col: (None if v == "-" else float(v)) for col, v in zip(ROUND_COLUMNS, round_vals)},
        })
    return pd.DataFrame(records)


def load_espn_snapshot(raw_html_path: Path | None = None) -> pd.DataFrame:
    if raw_html_path is None:
        candidates = sorted(RAW_DIR.glob("espn_*.html"))
        if not candidates:
            return pd.DataFrame()
        raw_html_path = candidates[-1]
    html = raw_html_path.read_text(errors="ignore")
    df = parse_espn_html(html)
    if df.empty:
        return df
    df["team_id"] = df["team_code"].map(ESPN_BETFAIR_TEAM_MAP)
    unmapped = df[df["team_id"].isna()]
    if not unmapped.empty:
        raise ValueError(f"Unmapped ESPN team codes: {sorted(unmapped['team_code'].unique())}")
    resolved = resolve_external_players(df, name_col="player_name", team_col="team_id")
    return resolved


def load_espn_round_level() -> pd.DataFrame:
    """Melt the wide ESPN snapshot into (player_id, round, espn_round_votes)
    long form, official round already applied via ROUND_LABEL_TO_OFFICIAL."""
    wide = load_espn_snapshot()
    if wide.empty:
        return pd.DataFrame(columns=["player_id", "round", "espn_round_votes"])
    long = wide.melt(
        id_vars=["player_id", "player_name", "team_id", "match_status"],
        value_vars=ROUND_COLUMNS, var_name="round_label", value_name="espn_round_votes",
    )
    long["round"] = long["round_label"].map(ROUND_LABEL_TO_OFFICIAL)
    long = long.dropna(subset=["espn_round_votes"])
    return long[["player_id", "player_name", "team_id", "match_status", "round", "espn_round_votes"]]
