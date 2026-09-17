"""
Page-level smoke tests (Streamlit AppTest, headless) for the three new/
changed pages added alongside the dual-model work: Order Scenarios, Model
Agreement, and the revised Betting Opportunities (all three of its view
modes). Confirms each renders without raising -- does not assert on model
correctness, which is covered elsewhere (test_objective_montecarlo.py,
test_betting_dual_model.py, test_objective_stats_model.py).
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGES = Path(__file__).resolve().parent.parent / "pages"


def test_order_scenarios_page_renders():
    at = AppTest.from_file(str(PAGES / "14_Order_Scenarios.py"))
    at.run(timeout=30)
    assert not at.exception


def test_model_agreement_page_renders():
    at = AppTest.from_file(str(PAGES / "15_Model_Agreement.py"))
    at.run(timeout=30)
    assert not at.exception


def test_betting_opportunities_all_three_views_render():
    for view in [None, "Objective Stats Model", "Compare Both"]:
        at = AppTest.from_file(str(PAGES / "12_Betting_Opportunities.py"))
        at.run(timeout=30)
        if at.warning:
            continue  # markets repo/output not present on this machine -- graceful path, nothing to switch
        if view:
            at.radio[0].set_value(view).run(timeout=30)
        assert not at.exception, f"view={view!r}"


def test_guide_and_faqs_page_still_renders_after_new_sections():
    at = AppTest.from_file(str(PAGES.parent / "app.py"))
    at.run(timeout=30)
    assert not at.exception
