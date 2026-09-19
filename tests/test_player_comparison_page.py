"""
Regression tests for pages/29_Player_Comparison.py.

Covers: default 2-player render, add/remove/duplicate-prevention mechanics,
real Daicos-vs-Smith and Gunston numbers, finishing-probability values
traced to the real Monte Carlo arrays (not approximated), and confirmation
that no model/data/simulation output changed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 180  # cold-cache Monte Carlo array loads are genuinely slow, not a bug


def _fresh_page():
    at = AppTest.from_file(str(ROOT / "pages" / "29_Player_Comparison.py"), default_timeout=TIMEOUT)
    at.run()
    return at


def _assert_no_unexpected_exception(at):
    """This page unconditionally renders an st.page_link() to Player Search
    inside the Polling Profile section, which -- like every other
    st.page_link() user in this codebase (see test_sidebar_navigation.py) --
    can only resolve inside a live st.navigation() router context. Run
    standalone (as every test above the dedicated router test does), this
    raises StreamlitPageNotFoundError every time; that's the same known,
    pre-existing testing artifact this whole test suite already tolerates
    for other pages, not a bug in this page. The true "zero exception via
    the real app" proof is test_player_comparison_page_link_works_via_real_router
    below."""
    for exc in at.exception or []:
        assert "StreamlitPageNotFoundError" in str(exc) or "Could not find page" in str(exc), exc


def test_default_two_player_render():
    at = _fresh_page()
    _assert_no_unexpected_exception(at)
    select_keys = [s.key for s in at.selectbox if s.key.startswith("cmp_select")]
    assert len(select_keys) == 2


def test_add_player_up_to_five_and_remove():
    at = _fresh_page()
    add_btn = lambda: [b for b in at.button if "Add player" in b.label]
    for expected_count in (3, 4, 5):
        add_btn()[0].click().run()
        _assert_no_unexpected_exception(at)
        select_keys = [s.key for s in at.selectbox if s.key.startswith("cmp_select")]
        assert len(select_keys) == expected_count
    assert not add_btn(), "Add player button must not remain once 5 players are selected"

    remove_btns = [b for b in at.button if b.label == "Remove"]
    assert len(remove_btns) == 3  # one per player beyond the first 2
    remove_btns[0].click().run()
    select_keys = [s.key for s in at.selectbox if s.key.startswith("cmp_select")]
    assert len(select_keys) == 4


def test_no_duplicate_players_selectable():
    at = _fresh_page()
    [b for b in at.button if "Add player" in b.label][0].click().run()
    sel0 = [s for s in at.selectbox if s.key == "cmp_select_0"][0]
    sel1 = [s for s in at.selectbox if s.key == "cmp_select_1"][0]
    sel2 = [s for s in at.selectbox if s.key == "cmp_select_2"][0]
    assert sel1.value not in sel0.options or sel0.value != sel1.value
    assert sel0.value not in sel2.options
    assert sel1.value not in sel2.options


def test_daicos_vs_smith_real_numbers():
    at = _fresh_page()
    season_table = next(df.value for df in at.dataframe if "Prod EV" in df.value.columns)
    daicos = season_table[season_table["Player"] == "Nick Daicos"].iloc[0]
    smith = season_table[season_table["Player"] == "Bailey Smith"].iloc[0]
    assert daicos["Prod EV"] == "47.12"
    assert daicos["Obj EV"] == "38.68"
    assert daicos["Agreement"] == "DIVERGENT"
    assert smith["Prod EV"] == "35.37"
    assert smith["Agreement"] == "MODERATE"

    finishing_tables = [df.value for df in at.dataframe if "Winner %" in df.value.columns]
    assert len(finishing_tables) == 2  # Production tab + Objective tab
    prod_table = finishing_tables[0]
    assert prod_table[prod_table["Player"] == "Nick Daicos"].iloc[0]["Winner %"] == "99.1%"


def test_jack_gunston_disagreement_visible_not_blended():
    at = _fresh_page()
    [b for b in at.button if "Add player" in b.label][0].click().run()
    [s for s in at.selectbox if s.key == "cmp_select_2"][0].set_value("Jack Gunston").run()
    season_table = next(df.value for df in at.dataframe if "Prod EV" in df.value.columns)
    row = season_table[season_table["Player"] == "Jack Gunston"].iloc[0]
    assert row["Prod EV"] == "12.73"
    assert row["Obj EV"] == "1.29"
    assert row["Agreement"] == "DIVERGENT"
    # Both values present and distinct -- never averaged into one number.
    assert row["Prod EV"] != row["Obj EV"]


def test_agreement_band_uses_ev_scale_not_probability_scale():
    """The EV-scale CLOSE/MODERATE/DIVERGENT bands (documented on the Model
    Agreement page, section B) must be used -- NOT
    dashboard.finishing_order._agreement_label(), which is calibrated for a
    probability-percentage-point gap and would silently mislabel an EV-vote
    gap if misapplied."""
    src = (ROOT / "pages" / "29_Player_Comparison.py").read_text()
    assert '"Agreement": _ev_agreement_band(gap)' in src, (
        "the Season Comparison table's Agreement column must use the new "
        "EV-scale band function, not fo._agreement_label() (which is "
        "calibrated for a probability-percentage-point gap, not an EV-vote gap)"
    )
    assert "CLOSE" in src and "MODERATE" in src and "DIVERGENT" in src


def test_finishing_probability_matches_real_simulation_recount():
    mc = np.load(ROOT / "data" / "processed" / "mc_totals_2026.npy")
    idx = pd.read_csv(ROOT / "reports" / "2026_mc_player_index.csv")
    daicos_col = idx[idx["player_name"] == "Nick Daicos"].index[0]
    ranks = (-mc).argsort(axis=1).argsort(axis=1)[:, daicos_col] + 1
    direct_winner_pct = (ranks <= 1).mean() * 100

    at = _fresh_page()
    finishing_tables = [df.value for df in at.dataframe if "Winner %" in df.value.columns]
    displayed = finishing_tables[0]
    displayed_pct = float(displayed[displayed["Player"] == "Nick Daicos"].iloc[0]["Winner %"].rstrip("%"))
    assert abs(displayed_pct - direct_winner_pct) < 0.2


def test_no_model_or_data_outputs_changed():
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--stat", "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv",
         "reports/2026_match_probabilities.csv", "reports/2026_objective_votes.csv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert diff.stdout.strip() == ""


def test_player_comparison_added_to_main_nav_after_player_search():
    router_src = (ROOT / "app.py").read_text()
    main_block = router_src.split('"MAIN": [')[1].split("],")[0]
    search_pos = main_block.find("Player Search")
    comparison_pos = main_block.find("Player Comparison")
    to_poll_pos = main_block.find("To Poll a Vote")
    assert -1 < search_pos < comparison_pos < to_poll_pos


def test_player_comparison_page_link_works_via_real_router():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    at.switch_page("pages/29_Player_Comparison.py")
    at.run()
    assert not at.exception
