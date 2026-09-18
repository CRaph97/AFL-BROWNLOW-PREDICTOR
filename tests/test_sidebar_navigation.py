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


def test_router_loads_and_defaults_to_betting_opportunities():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.header, "default page rendered no headers at all"
    assert at.header[0].value == "1. Top Opportunities", (
        "default landing page is not Brownlow Betting Opportunities"
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
    files = sorted(glob.glob(str(ROOT / "pages" / "*.py")))
    assert len(files) >= 25, "expected all pre-existing pages plus the relocated Guide & FAQs"
    failures = []
    for f in files:
        at = AppTest.from_file(f, default_timeout=60)
        at.run()
        if at.exception:
            failures.append((f, str(at.exception)))
    assert not failures, failures


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
