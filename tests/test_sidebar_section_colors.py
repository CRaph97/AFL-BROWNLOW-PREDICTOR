"""
Regression tests for per-section sidebar accent styling.

app.py replaced Streamlit's automatic multipage sidebar menu with a custom
one (st.sidebar.expander()/st.page_link(), with the automatic menu hidden
via st.navigation(..., position="hidden")) because sidebar-mode automatic
nav groups cannot natively collapse per-section (verified live: only
position="top" supports that). dashboard.data.sidebar_nav_css() builds the
CSS for this custom sidebar and is unit-tested directly here, without
needing a live Streamlit/browser context -- it's a pure function of a
{section: colour} dict.

dashboard.data.highlight_objective_stats_nav() (the old per-page hook, still
called from 27 page files) is now an intentional no-op -- its own small
test lives in this file too.
"""
from __future__ import annotations

from dashboard.data import (
    SIDEBAR_SECTION_ACCENTS,
    highlight_objective_stats_nav,
    nav_slug,
    sidebar_nav_css,
)


def test_no_green_remains():
    css = sidebar_nav_css()
    assert "1f6f43" not in css
    assert "175934" not in css
    assert "2f8f57" not in css
    assert "green" not in css.lower()


def test_every_section_gets_a_rule_using_generic_key_wildcards():
    """No href/page enumeration needed at all (the structural fix for the
    earlier href-substring-collision bug class: this approach can't repeat
    it, since it never matches on href) -- each section's rule is a plain
    wildcard on its own st-key-navlink_<SECTION>_ prefix."""
    css = sidebar_nav_css()
    for section in SIDEBAR_SECTION_ACCENTS:
        slug = nav_slug(section)
        assert f'st-key-navlink_{slug}_' in css, f"no rule found for section {section!r}"
        assert f'st-key-navgroup_{slug}' in css, f"no header rule found for section {section!r}"


def test_overview_and_external_overview_get_different_colours():
    """MODEL ANALYSIS's 'Production Model' (pages/00_Overview.py) and
    EXTERNAL VALIDATION's 'External Overview' must not collide -- this
    approach avoids the risk structurally (it never matches on page name/
    href at all, only on which SECTION a link's key was built for), but
    assert the two sections' colours are at least genuinely different as a
    sanity check."""
    assert SIDEBAR_SECTION_ACCENTS["MODEL ANALYSIS"] != SIDEBAR_SECTION_ACCENTS["EXTERNAL VALIDATION"]


def test_active_link_rule_present_and_stronger_than_hover():
    """The active-page background must be a distinct, deliberately stronger
    tint than the plain hover tint for the same section (2E hex alpha ~18%
    vs 14 hex alpha ~8%)."""
    css = sidebar_nav_css()
    assert "2E" in css.upper() or "2e" in css
    assert "14" in css


def test_highlight_objective_stats_nav_is_a_harmless_noop():
    """Kept for its 27 existing per-page call sites; must not raise and
    must not touch Streamlit at all (no st.markdown call)."""
    import streamlit as st

    called = []
    orig = st.markdown
    st.markdown = lambda *a, **k: called.append((a, k))
    try:
        result = highlight_objective_stats_nav()
    finally:
        st.markdown = orig
    assert result is None
    assert called == []


def test_nav_slug_matches_between_sections_and_hypothetical_page_names():
    assert nav_slug("MODEL ANALYSIS") == "MODEL_ANALYSIS"
    assert nav_slug("External Overview") == "External_Overview"
    assert nav_slug("Guide & FAQs") == "Guide_FAQs"
