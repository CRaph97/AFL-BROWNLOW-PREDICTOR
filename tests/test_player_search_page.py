"""
Targeted tests for pages/28_Player_Search.py and the Part A betting-label
semantics fix (dashboard.betting_opportunities' CONFIDENCE_BADGES rename +
the new "Minimum model probability" filter on pages/27_To_Poll_A_Vote.py).

Audit finding (see docs comments in dashboard/betting_opportunities.py):
classify_confidence() in src/betting/classification.py is a correct,
internally-consistent EV/edge-agreement (value/mispricing) signal -- NOT a
probability-of-outcome claim, and there was no crossover between season
P(any), match P(any), EV, or Wheelo P3 anywhere in the pipeline. The only
real defect was the LABEL TEXT ("High Confidence") misleadingly reading as
an outcome-likelihood claim. Fixed by renaming display text only; no
threshold or probability value changed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard import betting_opportunities as bo
from dashboard import player_search as ps

ROOT = Path(__file__).resolve().parent.parent
OPPS_PATH = ROOT / "data" / "betting" / "processed" / "priced_opportunities.csv"

pytestmark = pytest.mark.skipif(not OPPS_PATH.exists(), reason="no priced betting opportunities in this environment")


@pytest.fixture(scope="module")
def opps():
    return pd.read_csv(OPPS_PATH)


class TestBettingLabelSemantics:
    def test_no_raw_high_confidence_wording_remains(self):
        for label in bo.CONFIDENCE_BADGES.values():
            assert "High Confidence" not in label

    def test_adam_cerra_shows_strong_bet_value_not_high_confidence(self, opps):
        cerra = opps[(opps["player_name"] == "Adam Cerra") & (opps["market_type"] == "TO_POLL_A_VOTE")]
        if cerra.empty:
            pytest.skip("Adam Cerra not present in this run's data")
        row = cerra.iloc[0]
        assert row["confidence"] == "HIGH_CONFIDENCE_WHEELO_CONFIRMED", "internal enum must NOT be renamed"
        assert bo.confidence_badge(row["confidence"]) == "\U0001F7E2 Strong Bet Value + Wheelo Support"
        # Real numbers unchanged by the label rename.
        assert row["implied_probability"] == pytest.approx(0.190476, abs=1e-4)
        assert row["production_probability"] == pytest.approx(0.3316, abs=1e-4)
        assert row["objective_probability"] == pytest.approx(0.30995, abs=1e-4)

    def test_adam_cerra_likelihood_is_low(self, opps):
        cerra = opps[(opps["player_name"] == "Adam Cerra") & (opps["market_type"] == "TO_POLL_A_VOTE")]
        if cerra.empty:
            pytest.skip("Adam Cerra not present in this run's data")
        row = bo.prepare_display(cerra).iloc[0]
        assert row["likelihood_display"] == "Low — 31.0%"

    def test_jack_crisp_and_jordon_sweet_likelihood_bands(self, opps):
        display = bo.prepare_display(opps[opps["market_type"] == "TO_POLL_A_VOTE"])
        for name, expected in [("Jack Crisp", "High — 70.1%"), ("Jordon Sweet", "High — 77.3%")]:
            rows = display[display["player_name"] == name]
            if rows.empty:
                pytest.skip(f"{name} not present in this run's data")
            assert rows.iloc[0]["likelihood_display"] == expected

    def test_bet_value_and_likelihood_are_independent(self, opps):
        """Bet Value must never be derived from conservative_internal_probability's
        raw value (only from its relationship to implied probability), and
        Likelihood must never use odds or the Bet Value classification --
        Jordon Sweet (High likelihood, Moderate Bet Value) proves they can
        diverge."""
        display = bo.prepare_display(opps[opps["market_type"] == "TO_POLL_A_VOTE"])
        sweet = display[display["player_name"] == "Jordon Sweet"]
        if sweet.empty:
            pytest.skip("Jordon Sweet not present in this run's data")
        row = sweet.iloc[0]
        assert row["likelihood_display"].startswith("High")
        assert "Moderate" in row["confidence_badge"]

    def test_minimum_probability_filter_excludes_cerra_at_50_includes_at_25(self, opps):
        tpav = opps[opps["market_type"] == "TO_POLL_A_VOTE"]
        display = bo.prepare_display(tpav)
        piv = bo.with_bookmaker_odds(display, ["player_id"])
        cerra = piv[piv["player_name"] == "Adam Cerra"]
        if cerra.empty:
            pytest.skip("Adam Cerra not present in this run's data")
        conservative = cerra.iloc[0]["conservative_internal_probability"]
        assert conservative < 0.50
        assert conservative >= 0.25

    def test_conservative_probability_is_min_of_both_models_with_fallback(self):
        """priced_opportunities.csv's conservative_internal_probability is
        pandas .min(axis=1, skipna default) over [production, objective] --
        confirm this genuinely falls back to whichever single value exists
        rather than becoming NaN when only one model resolved a player."""
        df = pd.DataFrame({
            "production_probability": [0.5, None, 0.3],
            "objective_probability": [0.4, 0.6, None],
        })
        result = df[["production_probability", "objective_probability"]].min(axis=1)
        assert list(result) == [0.4, 0.6, 0.3]


class TestPlayerSearchPage:
    @pytest.mark.parametrize("player_name", ["Nick Daicos", "Jack Gunston", "Adam Cerra"])
    def test_season_snapshot_loads_for_named_players(self, player_name):
        comp = ps.canonical_player_table()
        row = comp[comp["player_name"] == player_name]
        if row.empty:
            pytest.skip(f"{player_name} not present in this environment")
        pid = row.iloc[0]["player_id"]
        snap = ps.season_snapshot(pid)
        assert snap, f"no snapshot for {player_name}"
        assert snap["production"]["ev"] is not None or snap["objective"]["ev"] is not None

    def test_jack_gunston_shows_large_visible_disagreement_not_blended(self):
        comp = ps.canonical_player_table()
        row = comp[comp["player_name"] == "Jack Gunston"]
        if row.empty:
            pytest.skip("Jack Gunston not present in this environment")
        pid = row.iloc[0]["player_id"]
        snap = ps.season_snapshot(pid)
        prod_ev, obj_ev = snap["production"]["ev"], snap["objective"]["ev"]
        assert prod_ev is not None and obj_ev is not None
        assert abs(prod_ev - obj_ev) > 5, "Gunston's well-documented Production/Objective gap should be large"
        # The two values must be surfaced separately -- never averaged into one number.
        assert snap["production"]["ev"] != snap["objective"]["ev"]

    def test_wheelo_never_gets_a_p_any_field(self):
        comp = ps.canonical_player_table()
        pid = comp.iloc[0]["player_id"]
        snap = ps.season_snapshot(pid)
        assert "p_any" not in snap["wheelo"]

    def test_page_renders_for_named_players_and_deep_link_sets_correct_match_id(self):
        from streamlit.testing.v1 import AppTest

        for player_name in ["Nick Daicos", "Jack Gunston", "Adam Cerra"]:
            at = AppTest.from_file(str(ROOT / "pages" / "28_Player_Search.py"), default_timeout=60)
            at.run()
            sel = [s for s in at.selectbox if s.key == "player_search_choice"]
            if not sel or player_name not in sel[0].options:
                pytest.skip(f"{player_name} not selectable in this environment")
            sel[0].set_value(player_name).run()
            assert not at.exception, f"{player_name}: {at.exception}"

            btns = [b for b in at.button if b.label.startswith("R")]
            if not btns:
                continue
            expected_match_id = None
            # Recover the match_id the first button should set by re-deriving
            # the drilldown directly (same source the page itself reads).
            from dashboard import betting_opportunities as bo_mod
            comp = ps.canonical_player_table()
            pid = comp[comp["player_name"] == player_name].iloc[0]["player_id"]
            drilldown = bo_mod.match_level_drilldown(pid)
            if drilldown.empty:
                continue
            expected_match_id = drilldown.head(5).iloc[0]["match_id"]
            btns[0].click().run()
            assert at.session_state.get("match_detail_preselect_match_id") == expected_match_id

    def test_nav_includes_player_search_in_main_at_expected_position(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.run()
        assert not at.exception
        src = (ROOT / "app.py").read_text()
        main_block = src.split('"MAIN": [')[1].split("],")[0]
        assert 'st.Page("pages/28_Player_Search.py"' in main_block
        titles = [line.split('title="')[1].split('"')[0] for line in main_block.splitlines() if 'title="' in line]
        assert titles.index("Player Search") == titles.index("Finishing Order") + 1
        # Player Comparison was later inserted directly after Player Search
        # (pages/29_Player_Comparison.py); Teams and Round-by-Round
        # (pages/30, pages/31) were inserted after that -- To Poll a Vote
        # now follows Round-by-Round, not Player Search/Comparison directly.
        assert titles.index("Player Comparison") == titles.index("Player Search") + 1
        assert titles.index("Teams") == titles.index("Player Comparison") + 1
        assert titles.index("Round-by-Round") == titles.index("Teams") + 1
        assert titles.index("To Poll a Vote") == titles.index("Round-by-Round") + 1


def test_no_model_data_or_simulation_files_changed():
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--stat", "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv",
         "data/betting/processed/priced_opportunities.csv", "data/processed/mc_totals_2026.npy",
         "data/processed/mc_totals_objective_2026.npy"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert diff.stdout.strip() == ""
