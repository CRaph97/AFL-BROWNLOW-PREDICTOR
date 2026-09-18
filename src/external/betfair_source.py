"""
Betfair Brownlow predictor hub -- remote source, snapshot-fetched.

The live page's raw HTML contains 207 `<table>` elements in total, but only a
minority (22 in the 2026-09-18 snapshot) are genuine per-match vote tables (a
bold team-pair header row like "STK V GCS" followed by ~3-4 player-name/
predicted-vote rows, i.e. Betfair's own picks for that match's vote-getters)
-- the rest are unrelated page-layout tables. This reflects Betfair's own
editorial choice of which rounds/matches get a recap article on this hub
page, not a parsing limitation: coverage is genuinely PARTIAL (22 of 207
2026 matches in this snapshot), not near-complete.

DOCUMENTED LIMITATION: unlike Wheelo and ESPN, this page does not reliably
expose which official round each match table belongs to. "ROUND N" text
occurs in the surrounding prose, but a raw scan of the fetched HTML found 38
such mentions in NON-monotonic order (e.g. "23", then "14", then "22") --
consistent with round numbers being referenced conversationally within
article bodies (e.g. "his best game since Round 14") rather than only as
section headers, which makes nearest-heading attribution unreliable and,
per this project's explicit "never fabricate, document rather than invent"
principle, was judged too likely to misattribute a match to the wrong round
to use for round-level comparison. Betfair is therefore integrated at
SEASON level only (summed across all 207 match tables) -- not used on the
"Leader After Round" page.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.external.identity import ESPN_BETFAIR_TEAM_MAP, resolve_external_players

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "external" / "raw"

_TEAM_PAIR_RE = re.compile(r"<strong>([A-Z]{2,3})\s+V\s+([A-Z]{2,3})</strong>")


def parse_betfair_html(html: str) -> pd.DataFrame:
    tables = re.findall(r"<table[^>]*>.{0,3000}?</table>", html, re.S)
    records = []
    for t in tables:
        header = _TEAM_PAIR_RE.search(t)
        if not header:
            continue
        team_a, team_b = header.group(1), header.group(2)
        rows = re.findall(r"<tr>\s*<td[^>]*>(?:<strong>.*?</strong>)?</td>\s*<td[^>]*></td>\s*</tr>|<tr>\s*<td[^>]*>([A-Z][A-Z .'\-]+)</td>\s*<td[^>]*>([\d.]+)</td>\s*</tr>", t)
        for player, vote in rows:
            if not player:
                continue
            records.append({
                "player_name": player.strip().title(),
                "team_a": team_a, "team_b": team_b,
                "predicted_votes": float(vote),
            })
    return pd.DataFrame(records)


def _resolve_ambiguous_team(row, canonical) -> tuple:
    """A Betfair row gives two candidate team codes (the match's two sides,
    not which one the player is actually on) -- try resolving against each;
    accept only if exactly one side yields a match, otherwise leave
    unresolved rather than guessing which side the player belongs to."""
    from src.external.identity import _normalise_full_surname, _first_initial

    surname_key = _normalise_full_surname(row["player_name"])
    first_initial = _first_initial(row["player_name"])
    hits = canonical[(canonical["surname_key"] == surname_key) & (canonical["first_initial"] == first_initial)]
    team_a_id = ESPN_BETFAIR_TEAM_MAP.get(row["team_a"])
    team_b_id = ESPN_BETFAIR_TEAM_MAP.get(row["team_b"])
    candidates = hits[hits["team_id"].isin([team_a_id, team_b_id])]
    if len(candidates) == 1:
        return candidates.iloc[0]["player_id"], candidates.iloc[0]["team_id"], "resolved"
    if len(candidates) == 0:
        return pd.NA, pd.NA, "unresolved"
    return pd.NA, pd.NA, "ambiguous"


def load_betfair_season(raw_html_path: Path | None = None) -> pd.DataFrame:
    from src.external.identity import load_canonical_players

    if raw_html_path is None:
        candidates = sorted(RAW_DIR.glob("betfair_*.html"))
        if not candidates:
            return pd.DataFrame()
        raw_html_path = candidates[-1]
    html = raw_html_path.read_text(errors="ignore")
    raw = parse_betfair_html(html)
    if raw.empty:
        return raw

    canonical = load_canonical_players()
    resolved = raw.apply(lambda r: _resolve_ambiguous_team(r, canonical), axis=1, result_type="expand")
    raw["player_id"], raw["team_id"], raw["match_status"] = resolved[0], resolved[1], resolved[2]

    ok = raw[raw["match_status"] == "resolved"]
    season = ok.groupby("player_id", as_index=False).agg(
        betfair_ev=("predicted_votes", "sum"),
        betfair_player_name=("player_name", "first"),
        team_id=("team_id", "first"),
        n_matches=("predicted_votes", "size"),
    )
    season["betfair_rank"] = season["betfair_ev"].rank(ascending=False, method="min")
    return season.sort_values("betfair_rank").reset_index(drop=True)
