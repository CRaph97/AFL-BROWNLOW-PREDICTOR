"""
Regression tests for the "To Poll a Vote" page extraction and Match Detail
deep-link, plus the Match Detail round-sort fix.

Background: `round` is stored as a string, so a plain `sort_values(["round",
...])` in pages/6_Match_Detail.py's match selector produced lexicographic
order (R1, R10, R11, ..., R2, R20, ...) instead of numeric order. The
detailed "To Poll a Vote" workflow was also extracted out of
pages/23_Brownlow_Betting_Opportunities.py into its own page
(pages/27_To_Poll_A_Vote.py), which adds an "Open Match Detail" deep-link
(via st.session_state + st.switch_page) on each of a player's top-5 likely
polling rounds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_match_detail_round_options_are_numerically_sorted():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages" / "6_Match_Detail.py"), default_timeout=60)
    at.run()
    assert not at.exception
    sb = [s for s in at.selectbox if s.label == "Select match"][0]
    rounds_in_order = [int(opt.split(":")[0][1:]) for opt in sb.options]
    assert rounds_in_order == sorted(rounds_in_order), (
        f"round options are not numerically non-decreasing: {rounds_in_order[:20]}"
    )


def test_match_detail_works_normally_with_no_preselection():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/6_Match_Detail.py")
    at.run()
    assert not at.exception
    assert "match_detail_preselect_match_id" not in at.session_state


def test_to_poll_a_vote_page_shows_all_three_evidence_sources():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages" / "27_To_Poll_A_Vote.py"), default_timeout=60)
    at.run()
    assert not at.exception
    main_table = at.dataframe[0].value
    for col in ("Production %", "Objective %", "Wheelo support"):
        assert col in main_table.columns

    sel = [s for s in at.selectbox if s.key == "tpav_drilldown_player"]
    if not sel:
        pytest.skip("no To Poll a Vote selections available in this environment")
    sel[0].set_value("Tim English").run()
    assert not at.exception
    drill = [df.value for df in at.dataframe if "Production P3" in df.value.columns]
    assert drill, "drilldown table did not render"
    cols = set(drill[0].columns)
    assert {"Production P3", "Production P2", "Production P1", "Production EV"} <= cols
    assert {"Objective P3", "Objective P2", "Objective P1", "Objective EV"} <= cols
    assert {"Wheelo pred. votes", "Wheelo P3 (%)"} <= cols


def test_deep_link_bailey_dale_round13_opens_correct_hawthorn_match():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/27_To_Poll_A_Vote.py")
    at.run()
    sel = [s for s in at.selectbox if s.key == "tpav_drilldown_player"]
    if not sel or "Bailey Dale" not in sel[0].options:
        pytest.skip("Bailey Dale not present in this environment's To Poll a Vote data")
    sel[0].set_value("Bailey Dale").run()
    btns = [b for b in at.button if "R13" in (b.label or "")]
    assert btns, "no Round 13 deep-link button found for Bailey Dale"
    btns[0].click().run()
    assert not at.exception
    match_sb = [s for s in at.selectbox if s.label == "Select match"]
    assert match_sb, "did not navigate to Match Detail"
    assert "Hawthorn" in match_sb[0].value
    assert match_sb[0].value.startswith("R13:")


def test_betting_opportunities_page_link_works_via_real_router():
    """st.page_link only resolves inside a live st.navigation() context --
    verified via the real router (app.py + AppTest.switch_page), not by
    running the page file standalone (which raises StreamlitPageNotFoundError
    purely as a testing artifact, not a production bug, since the deployed
    app always goes through app.py)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("pages/23_Brownlow_Betting_Opportunities.py")
    at.run()
    assert not at.exception


def test_betting_opportunities_page_no_longer_contains_full_drilldown():
    src = (ROOT / "pages" / "23_Brownlow_Betting_Opportunities.py").read_text()
    assert "_render_polling_drilldown" not in src
    assert "tpav_drilldown_player" not in src
    assert 'st.page_link("pages/27_To_Poll_A_Vote.py"' in src


def test_no_model_or_data_outputs_changed():
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--stat", "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv",
         "data/betting/processed", "data/external/processed"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert diff.stdout.strip() == ""
