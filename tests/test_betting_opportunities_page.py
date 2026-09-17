"""
Page-level smoke test for pages/12_Betting_Opportunities.py using Streamlit's
AppTest harness (headless, no browser needed). Confirms the page renders
without raising, regardless of whether AFL-BROWNLOW-MARKETS is checked out
on this machine (both branches - real data and the graceful "no data" path -
must not throw).
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard import betting_data as bd

PAGE_PATH = str(Path(__file__).resolve().parent.parent / "pages" / "12_Betting_Opportunities.py")


def test_page_renders_without_exception():
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_page_shows_data_or_graceful_warning_never_both_absent():
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    status = bd.data_source_status()
    if status["file_found"]:
        assert len(at.dataframe) >= 1
        assert len(at.warning) == 0
    else:
        assert len(at.warning) >= 1
