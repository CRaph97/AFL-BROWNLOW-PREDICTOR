"""
Regression tests for per-section sidebar accent styling
(dashboard.data.highlight_objective_stats_nav, renamed in purpose but not in
name for zero-touch compatibility with its 27 existing call sites).

Covers a real bug found via headless-browser investigation during this
change: a position-based CSS selector (matching the Nth section's DOM
wrapper) breaks when a page is loaded directly by URL, because this app's
sidebar sometimes renders as a flat, ungrouped list instead of the expected
4-section structure. Fixed by matching each page individually via an exact,
"/"-anchored href suffix instead, which is unaffected by that DOM quirk.
"""
from __future__ import annotations

import re

from dashboard.data import highlight_objective_stats_nav


def _css() -> str:
    """Extract the raw <style> CSS this function emits, without needing a
    live Streamlit render context."""
    import streamlit as st

    captured = {}

    def _fake_markdown(html, unsafe_allow_html=False):
        captured["html"] = html

    orig = st.markdown
    st.markdown = _fake_markdown
    try:
        highlight_objective_stats_nav()
    finally:
        st.markdown = orig
    return captured["html"]


def test_no_green_remains():
    css = _css()
    assert "1f6f43" not in css
    assert "175934" not in css
    assert "2f8f57" not in css


def test_every_real_page_href_is_anchored_not_a_bare_substring():
    """Every selector must anchor on a leading "/", never a bare
    `[href*="Name"]` substring match -- the exact defect class that made
    "/Overview" (Production Model) also match "/External_Overview", and
    would have made "/Betting_Opportunities" also match
    "/Brownlow_Betting_Opportunities"."""
    css = _css()
    href_selectors = re.findall(r'\[href[$*]="[^"]*"\]', css)
    assert href_selectors, "expected at least one href-based selector"
    for sel in href_selectors:
        assert sel.startswith('[href$="') and (sel.endswith('/"]') or sel[8] == "/"), (
            f"non-anchored or non-suffix href selector found: {sel}"
        )


def test_overview_and_external_overview_get_different_colours():
    css = _css()
    # Extract the colour used in the [aria-current="page"] rule for each.
    def _accent_for(href_suffix: str) -> str:
        m = re.search(
            re.escape(f'[href$="{href_suffix}"][aria-current="page"] {{ background-color: ') + r"([^;]+);",
            css,
        )
        assert m, f"no active-state rule found for {href_suffix}"
        return m.group(1)

    prod_colour = _accent_for("/Overview")
    external_colour = _accent_for("/External_Overview")
    assert prod_colour != external_colour


def test_section_membership_matches_app_py():
    """The hardcoded section_pages mapping inside the function must stay in
    sync with app.py's real MAIN/MODEL ANALYSIS/EXTERNAL VALIDATION/ADVANCED
    lists -- spot-check a representative page from each group is present."""
    css = _css()
    for href_suffix in ["/Technical_Summary", "/Objective_Stats_Model", "/External_Overview", "/Player_Detail"]:
        assert f'[href$="{href_suffix}"]' in css, f"{href_suffix} missing from sidebar styling"
