"""
Read-only loader + display-formatting layer for the Neds/PointsBet Brownlow
Betting Opportunities page. Mirrors this project's established pattern
(dashboard/betting_data.py, dashboard/external_data.py): pages never compute
a probability, edge, or classification themselves -- everything numeric here
is a load of an already-built file from scripts/refresh_brownlow_odds.py's
output (data/betting/processed/*), never a live scrape or a re-derived edge.

Everything in this module that builds a human-readable STRING (market_label,
bet_description, confidence badges, formatting helpers) is presentation only
-- it never changes a probability, EV, edge, or classification, only how an
already-computed row is described to a reader who doesn't know this
project's internal market_type/confidence vocabulary.

Never scrapes on import or on any function call -- the refresh pipeline
(scripts/refresh_brownlow_odds.py) is a separate, manually-run step.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import data as d

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "betting" / "processed"


@st.cache_data
def load_opportunities() -> pd.DataFrame:
    path = PROCESSED / "priced_opportunities.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_combinations() -> pd.DataFrame:
    path = PROCESSED / "combinations.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_price_comparison() -> pd.DataFrame:
    path = PROCESSED / "price_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_refresh_summary() -> dict:
    path = PROCESSED / "refresh_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def canonical_player_names() -> set[str]:
    """The only trustworthy source of "this string is a real player" for a
    selector -- built from the same canonical Production/Objective identity
    join used everywhere else in this dashboard, not from whatever strings
    happen to appear in the betting pipeline's own player_name column (which
    is sourced from bookmaker entrant text and can -- see
    src/betting/scraping.py's UNMODELLED-row fix -- occasionally still carry
    a name this project cannot otherwise validate)."""
    comparison = d.load_dual_model_comparison()
    return set(comparison["player_name"].dropna().unique())


# --------------------------------------------------------------------------
# Human-readable market/bet descriptions
#
# Internal market_type constants (WINNER, TOP_N, TEAM_VOTES_OU, ...) are used
# for filtering/logic only and must never be rendered raw in normal UI -- see
# MARKET_LABEL_TEMPLATES below, used by market_label()/bet_description().
# --------------------------------------------------------------------------
_ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th",
             8: "8th", 9: "9th", 10: "10th"}


def _ordinal(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    return _ORDINALS.get(n, f"{n}th")


def _fmt_line(line) -> str:
    if line is None or (isinstance(line, float) and pd.isna(line)):
        return "?"
    f = float(line)
    return f"{f:g}"


def market_label(row: pd.Series) -> str:
    """The MARKET itself (no team/player/side) -- e.g. "Top 5 Finish". Never
    returns a raw market_type constant; returns "Unmodelled Market" rather
    than guessing for anything this pipeline doesn't price."""
    mt = row.get("market_type")
    if mt == "WINNER":
        return "Brownlow Winner"
    if mt == "TOP_N":
        return f"Top {int(row['n'])} Finish" if pd.notna(row.get("n")) else "Top N Finish"
    if mt == "EXACT_POSITION":
        return f"Finish {_ordinal(row.get('position'))}"
    if mt == "TO_POLL_A_VOTE":
        return "To Poll a Vote"
    if mt == "PLAYER_VOTES_OU":
        return f"Over/Under {_fmt_line(row.get('line'))} Brownlow Votes"
    if mt == "X_PLUS_VOTES":
        return f"{int(row['threshold'])}+ Brownlow Votes" if pd.notna(row.get("threshold")) else "X+ Brownlow Votes"
    if mt == "PLAYER_H2H":
        opponent = row.get("_h2h_opponent")
        return f"H2H vs {opponent}" if isinstance(opponent, str) and opponent else "Head-to-Head"
    if mt == "TEAM_VOTES_OU":
        team = d._display_team(row.get("team_id")) if pd.notna(row.get("team_id")) else "Team"
        return f"{team} Total Votes Over/Under {_fmt_line(row.get('line'))}"
    return "Unmodelled Market"


_H2H_PATTERN = re.compile(r"\(([^)]+?)\s+vs\s+([^)]+?)\)", re.IGNORECASE)


def _h2h_opponent_from_market_name(market_name: str, own: str) -> str | None:
    """Fallback for a PLAYER_H2H row that, for whatever reason, isn't part of
    a clean 2-row (source, market_name) group -- Neds' market_name embeds
    both names directly ("Brownlow H2H (Player A vs Player B)"), so this can
    recover the opponent from text alone when the primary sibling-row lookup
    (_attach_h2h_opponents) can't."""
    m = _H2H_PATTERN.search(str(market_name or ""))
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    own = str(own or "").strip()
    if own and own in a:
        return b
    if own and own in b:
        return a
    return b if a == own else a


def _attach_h2h_opponents(df: pd.DataFrame) -> pd.DataFrame:
    """PointsBet's PLAYER_H2H market_name does NOT embed both player names
    (e.g. "Brownlow Head to Head A (Brownlow Medal Player Head to Head)") --
    only Neds' does. The reliable, bookmaker-agnostic way to find a row's
    opponent is the same one src/betting/pricing.py's _price_h2h_group()
    already uses for PRICING: the two rows of one H2H market share
    (source, market_name), so the opponent is simply the other row in that
    group. Falls back to text extraction (Neds' format) only if a group
    isn't exactly 2 distinct players."""
    df = df.copy()
    df["_h2h_opponent"] = None
    h2h_mask = df["market_type"] == "PLAYER_H2H"
    if not h2h_mask.any():
        return df
    for (_source, _market_name), group in df[h2h_mask].groupby(["source", "market_name"]):
        names = group["player_name"].dropna().unique().tolist()
        if len(names) == 2:
            for idx in group.index:
                own = df.at[idx, "player_name"]
                df.at[idx, "_h2h_opponent"] = names[1] if own == names[0] else names[0]
        else:
            for idx in group.index:
                df.at[idx, "_h2h_opponent"] = _h2h_opponent_from_market_name(
                    df.at[idx, "market_name"], df.at[idx, "player_name"]
                )
    return df


def bet_description(row: pd.Series) -> str:
    """The full bet: WHAT + WHO/WHICH TEAM + WHAT line, in one readable
    sentence fragment, e.g. "Nick Daicos -- Top 5 Finish" or "Western
    Bulldogs -- Over 67.5 Total Votes"."""
    mt = row.get("market_type")
    label = market_label(row)
    side = str(row.get("side")).capitalize() if pd.notna(row.get("side")) else None

    if mt == "TEAM_VOTES_OU":
        team = d._display_team(row.get("team_id")) if pd.notna(row.get("team_id")) else "Unknown team"
        if side:
            return f"{team} -- {side} {_fmt_line(row.get('line'))} Total Votes"
        return f"{team} -- {label}"
    if mt == "PLAYER_VOTES_OU":
        who = row.get("player_name") or "Unknown player"
        if side:
            return f"{who} -- {side} {_fmt_line(row.get('line'))} Brownlow Votes"
        return f"{who} -- {label}"

    who = row.get("player_name")
    if pd.notna(who):
        return f"{who} -- {label}"
    return label


REQUIRED_FIELDS_BY_MARKET = {
    "WINNER": ["player_name"],
    "TOP_N": ["player_name", "n"],
    "EXACT_POSITION": ["player_name", "position"],
    "TO_POLL_A_VOTE": ["player_name"],
    "PLAYER_VOTES_OU": ["player_name", "line", "side"],
    "X_PLUS_VOTES": ["player_name", "threshold"],
    "PLAYER_H2H": ["player_name"],
    "TEAM_VOTES_OU": ["team_id", "line", "side"],
}


def is_display_incomplete(row: pd.Series) -> bool:
    """True if a REQUIRED display field for this row's market_type is
    missing -- selection/team/line/bookmaker/odds must all be resolvable for
    a row to appear anywhere outside the raw Advanced/All Markets inventory,
    per the brief's DISPLAY_MARKET_INCOMPLETE rule."""
    required = REQUIRED_FIELDS_BY_MARKET.get(row.get("market_type"))
    if required is None:
        return True  # UNMODELLED or anything not in the map is never display-complete
    if pd.isna(row.get("odds")) or pd.isna(row.get("source")):
        return True
    return any(pd.isna(row.get(f)) for f in required)


FLAGGED_STATUSES = {"PRICE_SUSPECT", "STALE_PRICE", "IDENTITY_AMBIGUOUS",
                    "SETTLEMENT_REVIEW_REQUIRED", "UNMODELLED", "INSUFFICIENT_SIMULATION_SUPPORT"}


def has_flag(row_or_flags) -> bool:
    flags = row_or_flags.get("data_quality_flags") if isinstance(row_or_flags, (pd.Series, dict)) else row_or_flags
    if not isinstance(flags, str) or not flags:
        return False
    return any(f in flags for f in FLAGGED_STATUSES)


QUALIFYING_CONFIDENCE = [
    "HIGH_CONFIDENCE_WHEELO_CONFIRMED", "HIGH_CONFIDENCE_WHEELO_NEUTRAL",
    "MEDIUM_CONFIDENCE", "HIGH_RISK_HIGH_REWARD",
]
_EVIDENCE_RANK = {c: i for i, c in enumerate(QUALIFYING_CONFIDENCE)}

CONFIDENCE_BADGES = {
    "HIGH_CONFIDENCE_WHEELO_CONFIRMED": "\U0001F7E2 High Confidence + Wheelo Confirmed",
    "HIGH_CONFIDENCE_WHEELO_NEUTRAL": "\U0001F7E2 High Confidence",
    "MEDIUM_CONFIDENCE": "\U0001F7E1 Medium Confidence",
    "HIGH_RISK_HIGH_REWARD": "\U0001F7E0 High Risk / High Reward",
    "MODEL_DISAGREEMENT": "⚪ Model Disagreement",
    "NO_VALUE": "⚪ No Value",
}


def confidence_badge(confidence: str) -> str:
    return CONFIDENCE_BADGES.get(confidence, str(confidence))


def prepare_display(df: pd.DataFrame) -> pd.DataFrame:
    """Adds pure-presentation columns (never touches a probability/edge/EV
    value) used across every section of the page."""
    if df.empty:
        return df
    out = _attach_h2h_opponents(df)
    out["bet"] = out.apply(bet_description, axis=1)
    out["market_label"] = out.apply(market_label, axis=1)
    out["is_incomplete"] = out.apply(is_display_incomplete, axis=1)
    out["has_flag"] = out["data_quality_flags"].apply(has_flag)
    out["confidence_badge"] = out["confidence"].apply(confidence_badge)
    out["conservative_edge_pp"] = (out["conservative_internal_probability"] - out["implied_probability"]) * 100.0
    return out


def qualifying_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """The "genuine opportunity" set for Top Opportunities / headline
    sections: a real confidence category worth surfacing, no data-quality
    flag, and a complete human-readable bet description. Excludes NO_VALUE,
    MODEL_DISAGREEMENT, and UNMODELLED explicitly -- the previous page
    version filtered flags but never confidence, which is what let
    NO_VALUE/MODEL_DISAGREEMENT rows appear in "Best Opportunities"."""
    if df.empty:
        return df
    q = df[
        df["confidence"].isin(QUALIFYING_CONFIDENCE)
        & ~df["has_flag"]
        & ~df["is_incomplete"]
    ].copy()
    q["_evidence_rank"] = q["confidence"].map(_EVIDENCE_RANK)
    return q.sort_values(["_evidence_rank", "conservative_edge_pp"], ascending=[True, False]).drop(columns="_evidence_rank")


def format_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:.1f}%"


def format_pp(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.1f}pp"


def format_ev(x) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:+.1f}%"


def format_odds(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}"


def with_bookmaker_odds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Collapses per-source rows (neds / pointsbet_1 / pointsbet_2) that
    describe the SAME selection (identified by group_cols) into one row with
    neds_odds / pointsbet_odds / best_odds / best_bookmaker columns, keeping
    every other (non-source, non-odds) column from the first row in the
    group. Used by every section that shows one line per selection with
    both bookmakers' prices side by side, so this grouping logic exists in
    exactly one place."""
    if df.empty:
        return df
    work = df.copy()
    work["_book"] = work["source"].apply(lambda s: "PointsBet" if str(s).startswith("pointsbet") else "Neds")
    rows = []
    other_cols = [c for c in work.columns if c not in ("source", "odds", "selection_id", "_book")]
    for _, group in work.groupby(group_cols, dropna=False):
        base = group.iloc[0][other_cols].to_dict()
        neds = group[group["_book"] == "Neds"]["odds"]
        pb = group[group["_book"] == "PointsBet"]["odds"]
        neds_odds = neds.iloc[0] if not neds.empty else None
        pb_odds = pb.iloc[0] if not pb.empty else None
        candidates = [("Neds", neds_odds), ("PointsBet", pb_odds)]
        candidates = [(b, o) for b, o in candidates if o is not None and not pd.isna(o)]
        if candidates:
            best_book, best_odds = max(candidates, key=lambda t: t[1])
        else:
            best_book, best_odds = None, None
        base.update({"neds_odds": neds_odds, "pointsbet_odds": pb_odds,
                     "best_odds": best_odds, "best_bookmaker": best_book})
        rows.append(base)
    return pd.DataFrame(rows)
