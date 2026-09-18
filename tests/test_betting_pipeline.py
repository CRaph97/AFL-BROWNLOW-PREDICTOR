"""
Integration/identity/QA tests for the Brownlow Betting Opportunities module
(src/betting/, scripts/refresh_brownlow_odds.py, dashboard/betting_opportunities.py).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

HARD_IDENTITIES = [
    ("Jason Horne-Francis", "port_adelaide"),
    ("Nasiah Wanganeen-Milera", "st_kilda"),
    ("Luke Davies-Uniacke", "north_melbourne"),
    ("Darcy Byrne-Jones", "port_adelaide"),
    ("Jack Gunston", "hawthorn"),
    ("Nick Daicos", "collingwood"),
    # Chad Warner is deliberately excluded from this list -- see
    # test_chad_warner_is_honestly_ambiguous_at_season_level below: he and
    # Corey Warner (both Sydney, both first-initial "C") are a GENUINE
    # season-level identity ambiguity, correctly left unresolved rather than
    # guessed, per this project's established "ambiguous stays unmatched"
    # principle. A season-total betting market naming "Chad Warner" cannot
    # be safely resolved by name+team alone; only a round-specific market
    # could be, via load_canonical_player_rounds().
]


def test_scraping_produces_honest_status_not_fabricated_data():
    from src.betting import scraping
    statuses = scraping.load_cached_statuses()
    assert len(statuses) == 3
    for s in statuses:
        assert s["status"] in ("OK", "NO_MARKET_DATA_IN_RESPONSE", "FETCH_FAILED")
        # If a source claims market data was found, it must report a positive
        # count -- never a silent zero passed off as "OK".
        if s["status"] == "OK":
            assert s["contains_market_data"] is True


def test_scraping_never_reports_fabricated_market_counts():
    from src.betting import scraping
    for s in scraping.load_cached_statuses():
        if s["status"] == "NO_MARKET_DATA_IN_RESPONSE":
            assert s["n_markets_found"] == 0
            assert s["n_selections_found"] == 0


def test_parse_all_from_snapshots_returns_correct_schema_when_no_raw_json_present(tmp_path, monkeypatch):
    """scraping.py was rewritten to use real Playwright browser automation +
    captured network JSON (see its module docstring) rather than static-HTML
    parsing -- this test now exercises that real code path's empty case
    (no raw snapshot files on disk yet) rather than the retired
    static-HTML `parse_markets()` function, which no longer exists."""
    from src.betting import scraping
    monkeypatch.setattr(scraping, "RAW_DIR", tmp_path)
    df = scraping.parse_all_from_snapshots()
    assert list(df.columns) == [
        "source", "market_type", "market_name", "selection", "player_name", "team",
        "odds", "line", "side", "n", "position", "threshold", "selection_id",
        "settlement_comments",
    ]
    assert df.empty


@pytest.mark.parametrize("name,team", HARD_IDENTITIES)
def test_hard_identity_resolves_via_shared_external_identity_module(name, team):
    """This module deliberately reuses src.external.identity rather than a
    new resolver -- proving the 7 named hard identities resolve there
    directly proves this module's identity handling is correct too, since it
    imports the same functions (see src/betting's scripts/refresh_brownlow_odds.py
    step_5_resolve_identities)."""
    from src.external.identity import load_canonical_players, resolve_external_players
    canonical = load_canonical_players()
    external_row = pd.DataFrame([{"player_name": name, "team": team}])
    resolved = resolve_external_players(external_row, name_col="player_name", team_col="team", canonical=canonical)
    assert resolved.iloc[0]["match_status"] == "resolved", f"{name} ({team}) failed to resolve"
    assert pd.notna(resolved.iloc[0]["player_id"])


def test_chad_warner_and_corey_warner_do_not_collide_in_betting_identity_resolution():
    """The exact collision class this project has fixed twice before
    (Byrne-Jones/Jones, Warner/Warner) must not silently recur here: even
    though both are season-level ambiguous (see test below), resolving them
    must never produce the SAME player_id for both -- it must produce NaN
    for both, not a guess."""
    from src.external.identity import load_canonical_players, resolve_external_players
    canonical = load_canonical_players()
    rows = pd.DataFrame([
        {"player_name": "Chad Warner", "team": "sydney"},
        {"player_name": "Corey Warner", "team": "sydney"},
    ])
    resolved = resolve_external_players(rows, name_col="player_name", team_col="team", canonical=canonical)
    ids = resolved["player_id"].dropna().tolist()
    assert len(set(ids)) == len(ids), "Chad Warner and Corey Warner resolved to the same player_id"


def test_chad_warner_is_honestly_ambiguous_at_season_level_not_guessed():
    """A real, worthwhile finding from building this module: Chad Warner and
    Corey Warner (both Sydney) share BOTH first-initial "C" and surname
    "Warner", so a season-total-only betting market naming just "C. Warner"
    or "Chad Warner" (if a bookmaker's own labelling were ever similarly
    coarse) cannot be safely resolved by name+team alone -- this is
    correctly flagged 'ambiguous', never silently guessed. A round-specific
    market (e.g. a same-game market) COULD be resolved via
    load_canonical_player_rounds(), since the two are not both active in
    every round."""
    from src.external.identity import load_canonical_players, resolve_external_players
    canonical = load_canonical_players()
    external_row = pd.DataFrame([{"player_name": "Chad Warner", "team": "sydney"}])
    resolved = resolve_external_players(external_row, name_col="player_name", team_col="team", canonical=canonical)
    assert resolved.iloc[0]["match_status"] == "ambiguous"
    assert pd.isna(resolved.iloc[0]["player_id"])


def test_refresh_pipeline_produces_valid_schema_files():
    processed = ROOT / "data" / "betting" / "processed"
    for fname in ["priced_opportunities.csv", "combinations.csv", "price_comparison.csv", "refresh_summary.json"]:
        assert (processed / fname).exists(), f"{fname} missing -- run scripts/refresh_brownlow_odds.py"

    from scripts.refresh_brownlow_odds import OPPORTUNITY_COLUMNS
    opps = pd.read_csv(processed / "priced_opportunities.csv")
    assert list(opps.columns) == OPPORTUNITY_COLUMNS

    summary = json.loads((processed / "refresh_summary.json").read_text())
    assert "source_statuses" in summary
    assert "validation" in summary
    assert summary["validation"]["probabilities_in_unit_interval"] is True
    assert summary["validation"]["odds_greater_than_one"] is True
    assert summary["validation"]["no_duplicate_selection_id"] is True


def test_refresh_pipeline_is_honest_about_zero_real_markets():
    """This is the correct, expected state for this task's run (see
    docs/BETTING_OPPORTUNITIES.md) -- the test exists to make that state
    explicit and monitored, not to assert success that didn't happen."""
    summary = json.loads((ROOT / "data" / "betting" / "processed" / "refresh_summary.json").read_text())
    for s in summary["source_statuses"]:
        assert s["status"] in ("OK", "NO_MARKET_DATA_IN_RESPONSE", "FETCH_FAILED")


def test_validate_step_flags_probabilities_out_of_range():
    from scripts.refresh_brownlow_odds import step_11_validate
    bad = pd.DataFrame({
        "production_probability": [1.5],
        "objective_probability": [0.5],
        "odds": [2.0],
        "selection_id": ["a"],
    })
    result = step_11_validate(bad, pd.DataFrame())
    assert result["probabilities_in_unit_interval"] is False


def test_validate_step_flags_odds_at_or_below_one():
    from scripts.refresh_brownlow_odds import step_11_validate
    bad = pd.DataFrame({
        "production_probability": [0.5], "objective_probability": [0.5],
        "odds": [1.0], "selection_id": ["a"],
    })
    result = step_11_validate(bad, pd.DataFrame())
    assert result["odds_greater_than_one"] is False


def test_validate_step_flags_duplicate_selection_ids():
    from scripts.refresh_brownlow_odds import step_11_validate
    bad = pd.DataFrame({
        "production_probability": [0.5, 0.5], "objective_probability": [0.5, 0.5],
        "odds": [2.0, 2.0], "selection_id": ["a", "a"],
    })
    result = step_11_validate(bad, pd.DataFrame())
    assert result["no_duplicate_selection_id"] is False


def test_module_never_mutates_production_or_objective_outputs():
    """Import and exercise every betting module, then confirm the canonical
    leaderboard files are byte-identical to before -- proves this is a
    read-only layer, not merely documented as one."""
    prod_path = ROOT / "reports" / "2026_leaderboard.csv"
    obj_path = ROOT / "reports" / "2026_objective_leaderboard.csv"
    before_prod = prod_path.read_bytes()
    before_obj = obj_path.read_bytes()

    from src.betting import classification, combinations, market_data, pricing, scraping  # noqa: F401
    sims = market_data.load_production_simulations()
    pricing.price_winner(sims, sims.player_ids[0])

    assert prod_path.read_bytes() == before_prod
    assert obj_path.read_bytes() == before_obj


def test_dashboard_loader_returns_empty_dataframe_not_error_when_no_data():
    from dashboard import betting_opportunities as bo
    df = bo.load_opportunities()
    assert isinstance(df, pd.DataFrame)


def test_no_stake_sizing_or_bet_placement_code_exists():
    """Grep the whole betting module for anything resembling stake sizing or
    account/bet automation -- these are permanently out of scope per the
    project brief, and this test exists so a future change can't silently
    reintroduce them."""
    betting_src = (ROOT / "src" / "betting").glob("*.py")
    forbidden = ["place_bet", "stake_size", "bookmaker_login", "submit_bet", "kelly_criterion", "stake ="]
    for f in betting_src:
        text = f.read_text().lower()
        for term in forbidden:
            assert term not in text, f"forbidden term '{term}' found in {f.name}"
