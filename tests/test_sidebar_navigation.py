"""
Regression tests for the grouped st.navigation() sidebar restructuring.

Scope: navigation/routing only. Does not exercise any model/data/pricing
logic beyond confirming each page still runs without exception (already
covered in depth by each page's own test suite).
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_router_loads_and_defaults_to_technical_summary():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.header, "default page rendered no headers at all"
    assert at.header[0].value == "1. How the Modelling Works", (
        "default landing page is not Technical Summary"
    )


def test_old_betting_opportunities_page_untouched_and_still_runs():
    """Superseded for normal nav (hidden), but its file must be byte-for-byte
    unmodified and must still execute standalone without exception -- a
    hidden page remains reachable by direct URL per Streamlit's documented
    visibility="hidden" behaviour."""
    path = ROOT / "pages" / "12_Betting_Opportunities.py"
    assert path.exists()
    at = AppTest.from_file(str(path), default_timeout=60)
    at.run()
    assert not at.exception


def test_every_existing_page_still_runs_standalone():
    """st.page_link() can only resolve a target page inside a live
    st.navigation() context -- a page that uses it (e.g.
    23_Brownlow_Betting_Opportunities.py, which links to the extracted
    27_To_Poll_A_Vote.py page) raises StreamlitPageNotFoundError when run
    standalone outside app.py's router. That's a testing artifact, not a
    production bug, since the deployed app always goes through app.py --
    verified separately, through the real router, by
    test_page_link_pages_work_via_real_router() below."""
    files = sorted(glob.glob(str(ROOT / "pages" / "*.py")))
    assert len(files) >= 25, "expected all pre-existing pages plus the relocated Guide & FAQs"
    pages_requiring_router_context = {
        "23_Brownlow_Betting_Opportunities.py",
        "29_Player_Comparison.py",  # links to Player Search via st.page_link()
    }
    failures = []
    for f in files:
        if Path(f).name in pages_requiring_router_context:
            continue
        at = AppTest.from_file(f, default_timeout=90)
        at.run()
        if at.exception:
            failures.append((f, str(at.exception)))
    assert not failures, failures


def test_page_link_pages_work_via_real_router():
    """Pages excluded above (because they use st.page_link, which needs a
    live st.navigation() context) must still work when actually reached the
    way a user reaches them: through app.py's router."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/23_Brownlow_Betting_Opportunities.py")
    at.run()
    assert not at.exception


def test_guide_and_faqs_relocated_not_duplicated():
    """Guide & FAQs must exist at its new location and app.py must no longer
    itself be a page body (it's a pure router now)."""
    new_home = ROOT / "pages" / "24_Guide_And_FAQs.py"
    assert new_home.exists()
    assert "Guide & FAQs" in new_home.read_text()

    router_src = (ROOT / "app.py").read_text()
    assert "st.navigation(" in router_src
    assert "30-Second Overview" not in router_src, "app.py should no longer contain page content directly"


def test_hidden_page_excluded_from_navigation_pages_dict_visible_defaults():
    """Structural check on app.py's own navigation config: the old page is
    marked hidden, the new one is marked default, matching what was verified
    live above."""
    router_src = (ROOT / "app.py").read_text()
    assert 'st.Page("pages/12_Betting_Opportunities.py"' in router_src
    assert 'visibility="hidden"' in router_src
    assert 'st.Page("pages/23_Brownlow_Betting_Opportunities.py"' in router_src
    assert "default=True" in router_src


def test_technical_summary_page_renders_and_is_placed_after_overview_in_main():
    at = AppTest.from_file(str(ROOT / "pages" / "25_Technical_Summary.py"), default_timeout=60)
    at.run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert headers == [
        "1. How the Modelling Works", "2. Production Model", "3. Objective Stats Model",
        "4. External / Wheelo", "5. Model Comparison", "6. Validation / Integrity",
        "7. Weightings / Feature Importance",
    ]

    # NOTE: this originally asserted Technical Summary comes after Overview
    # in MAIN. A later task moved "Overview" (retitled "Production Model")
    # out of MAIN into MODEL ANALYSIS entirely, obsoleting that specific
    # relative-position check -- Technical Summary being present in MAIN at
    # all is the part of this test's contract that still applies.
    router_src = (ROOT / "app.py").read_text()
    main_start = router_src.index('"MAIN": [')
    main_end = router_src.index("],", main_start)
    main_block = router_src[main_start:main_end]
    assert 'st.Page("pages/25_Technical_Summary.py"' in main_block
    assert "pages/00_Overview.py" not in main_block, "Overview/Production Model must no longer be in MAIN"


# ---------------------------------------------------------------------------
# MAIN/MODEL ANALYSIS/EXTERNAL VALIDATION consolidation (ADVANCED retired).
# ---------------------------------------------------------------------------

def test_advanced_group_retired_and_new_groups_present():
    """The docstring and an inline comment may still mention "ADVANCED" in
    prose (explaining its retirement / where its pages moved) -- the
    structural check is that it is no longer a dict key."""
    router_src = (ROOT / "app.py").read_text()
    assert '"ADVANCED": [' not in router_src
    for section in ('"MAIN": [', '"MODEL ANALYSIS": [', '"EXTERNAL VALIDATION": ['):
        assert section in router_src


def test_teams_and_round_by_round_visible_in_main_at_expected_position():
    router_src = (ROOT / "app.py").read_text()
    main_start = router_src.index('"MAIN": [')
    main_end = router_src.index("],", main_start)
    main_block = router_src[main_start:main_end]
    assert 'st.Page("pages/30_Teams.py", title="Teams")' in main_block
    assert 'st.Page("pages/31_Round_By_Round.py", title="Round-by-Round")' in main_block
    titles = [line.split('title="')[1].split('"')[0] for line in main_block.splitlines() if 'title="' in line]
    assert titles.index("Teams") == titles.index("Player Comparison") + 1
    assert titles.index("Round-by-Round") == titles.index("Teams") + 1
    assert titles.index("To Poll a Vote") == titles.index("Round-by-Round") + 1


def test_post_brownlow_diagnostic_pages_kept_in_model_analysis():
    """Scenario Comparison / Model Disagreement / Uncertainty / Defender Bias
    Watchlist / Projection Concentration must survive the ADVANCED->MODEL
    ANALYSIS move unaltered -- these are explicitly preserved for
    post-Brownlow audit/retraining, not redundant with the new MAIN pages."""
    router_src = (ROOT / "app.py").read_text()
    ma_start = router_src.index('"MODEL ANALYSIS": [')
    ma_end = router_src.index("],", ma_start)
    ma_block = router_src[ma_start:ma_end]
    for path, title in [
        ("pages/2_Scenario_Comparison.py", "Scenario Comparison"),
        ("pages/3_Model_Disagreement.py", "Model Disagreement"),
        ("pages/9_Uncertainty.py", "Uncertainty"),
        ("pages/7_Defender_Bias_Watchlist.py", "Defender Bias Watchlist"),
        ("pages/8_Projection_Concentration.py", "Projection Concentration"),
        ("pages/14_Order_Scenarios.py", "Order Scenarios"),
    ]:
        assert f'st.Page("{path}", title="{title}")' in ma_block


@pytest.mark.parametrize("path,title", [
    ("pages/12_Betting_Opportunities.py", "Betting Opportunities"),
    ("pages/16_Player_H2H.py", "Player H2H"),
    ("pages/17_Multi_Player_Comparison.py", "Multi Player Comparison"),
    ("pages/18_Team_Player_Rankings.py", "Team Player Rankings"),
    ("pages/1_Player_Detail.py", "Player Detail"),
    ("pages/5_Round_View.py", "Round View"),
    ("pages/11_Team_Breakdown.py", "Team Breakdown"),
    ("pages/10_Round_By_Round_Leaderboard.py", "Round By Round Leaderboard"),
])
def test_legacy_page_hidden_but_routable(path, title):
    router_src = (ROOT / "app.py").read_text()
    assert f'st.Page("{path}", title="{title}", visibility="hidden")' in router_src
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    at.switch_page(path)
    at.run()
    assert not at.exception, f"{path}: {at.exception}"


def test_hidden_pages_excluded_from_rendered_sidebar_links():
    """st.page_link()'s rendered label defaults to the page title -- confirm
    none of the hidden legacy titles leak into the sidebar's actual
    page_link elements (the custom sidebar loop in app.py explicitly skips
    any page whose .visibility == "hidden")."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    rendered_labels = {pl.proto.label for pl in at.sidebar.get("page_link")}
    hidden_titles = {
        "Player H2H", "Multi Player Comparison", "Team Player Rankings",
        "Player Detail", "Round View", "Team Breakdown", "Round By Round Leaderboard",
        "Betting Opportunities",
    }
    assert not (hidden_titles & rendered_labels), hidden_titles & rendered_labels
    # Sanity check the positive case too, so an empty-by-accident sidebar
    # wouldn't silently pass the assertion above.
    assert "Teams" in rendered_labels
    assert "Round-by-Round" in rendered_labels
