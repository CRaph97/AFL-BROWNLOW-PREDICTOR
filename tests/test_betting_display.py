"""
Regression tests for the Betting Opportunities page redesign
(dashboard/betting_opportunities.py, pages/23_Brownlow_Betting_Opportunities.py).

Covers the display/mapping bugs fixed in that pass:
- raw CONSTANT_CASE market_type leaking into UI (market_label/bet_description)
- Top-N not showing which N
- team O/U rows not showing team+line
- non-player text (margin buckets, "Round N") leaking into player_name
  (src/betting/scraping.py's UNMODELLED-row fix)
- combination display completeness (no "? -- ?", no blank legs)
- Top Opportunities excluding NO_VALUE/MODEL_DISAGREEMENT/UNMODELLED, not
  just data-quality-flagged rows
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard import betting_opportunities as bo

ROOT = Path(__file__).resolve().parent.parent
OPPS_PATH = ROOT / "data" / "betting" / "processed" / "priced_opportunities.csv"
COMBOS_PATH = ROOT / "data" / "betting" / "processed" / "combinations.csv"

pytestmark = pytest.mark.skipif(not OPPS_PATH.exists(), reason="no priced betting opportunities in this environment")


@pytest.fixture(scope="module")
def opps():
    return pd.read_csv(OPPS_PATH)


@pytest.fixture(scope="module")
def display(opps):
    return bo.prepare_display(opps)


def test_market_label_never_leaks_raw_constant(display):
    known_constants = set(display["market_type"].unique())
    leaked = [label for label in display["market_label"].unique() if label in known_constants]
    assert not leaked, f"raw market_type constant(s) leaked into a human-readable label: {leaked}"
    assert not display["market_label"].str.fullmatch(r"[A-Z_]+").any(), (
        "a CONSTANT_CASE-looking label leaked into the UI"
    )


def test_top_n_label_shows_the_n(display):
    top_n = display[display["market_type"] == "TOP_N"]
    if top_n.empty:
        pytest.skip("no TOP_N rows in this run's data")
    assert (top_n["market_label"].str.contains(r"Top \d+ Finish")).all()


def test_team_votes_ou_bet_names_team_and_line(display):
    team_rows = display[display["market_type"] == "TEAM_VOTES_OU"]
    if team_rows.empty:
        pytest.skip("no TEAM_VOTES_OU rows in this run's data")
    assert not team_rows["bet"].str.contains("Unknown team|nan", case=False, na=False).any()
    # every team bet must mention a numeric line
    assert team_rows["bet"].str.contains(r"\d").all()


def test_no_non_player_text_in_player_name(opps):
    bad_patterns = ["votes", "Round "]
    names = opps["player_name"].dropna().unique()
    leaked = [n for n in names if any(p.lower() in str(n).lower() for p in bad_patterns)]
    assert not leaked, f"non-player text leaked into player_name: {leaked}"


def test_qualifying_opportunities_excludes_no_value_and_unmodelled(display):
    q = bo.qualifying_opportunities(display)
    assert not q["confidence"].isin(["NO_VALUE", "MODEL_DISAGREEMENT"]).any()
    assert not (q["market_type"] == "UNMODELLED").any()
    assert not q["has_flag"].any()
    assert not q["is_incomplete"].any()


def test_combinations_have_no_blank_or_placeholder_legs():
    if not COMBOS_PATH.exists():
        pytest.skip("no combinations file in this environment")
    combos = pd.read_csv(COMBOS_PATH)
    if combos.empty:
        pytest.skip("no combination rows in this run's data")
    assert combos["legs"].notna().all()
    assert not combos["legs"].astype(str).str.contains(r"\?", regex=True).any()
    assert not (combos["legs"].astype(str).str.strip() == "").any()


def test_bet_description_always_non_empty_for_modelled_rows(display):
    modelled = display[display["market_type"] != "UNMODELLED"]
    assert modelled["bet"].notna().all()
    assert not (modelled["bet"].astype(str).str.strip() == "").any()


def test_h2h_opponent_resolves_from_market_name(display):
    h2h = display[display["market_type"] == "PLAYER_H2H"]
    if h2h.empty:
        pytest.skip("no PLAYER_H2H rows in this run's data")
    assert h2h["market_label"].str.startswith("H2H vs ").all()
    assert not h2h["market_label"].str.contains("H2H vs None").any()
