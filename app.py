"""
Entrypoint / navigation router for the 2026 Brownlow dashboard.

This file used to CONTAIN the "Guide & FAQs" page's body directly. Streamlit
requires the entrypoint to become a pure router as soon as any session calls
st.navigation() -- the pages/ directory's filename-based auto-discovery is
then ignored entirely -- so Guide & FAQs' content was relocated, unchanged,
to pages/24_Guide_And_FAQs.py.

Sidebar structure: grouped into MAIN / MODEL ANALYSIS / EXTERNAL VALIDATION /
ADVANCED sections.

CUSTOM SIDEBAR, not Streamlit's automatic multipage menu -- built from
st.sidebar.expander() + st.page_link() instead, because native st.navigation
sidebar groups render as static, non-collapsible headers (verified against
this Streamlit version's real behaviour: per-section collapse only exists
for position="top", a full top-nav-bar layout, not the traditional left
sidebar this app keeps). st.navigation(pages, position="hidden") is still
used underneath as the actual router (it owns page registration/routing and
is what every st.page_link() below resolves against and what
st.switch_page() -- used by the existing Match Detail deep-link -- targets);
its own built-in visible menu is simply never shown, so there is exactly ONE
visible sidebar navigation, the custom one below.

Colour + collapse verified live with a headless browser during development:
- st.container(key=...) and st.expander(key=...) both apply a stable,
  predictable "st-key-<key>" CSS class -- unlike the automatic menu's
  auto-generated emotion-cache classes, this is safe to target directly.
- st.page_link() does NOT set aria-current (unlike the automatic menu's
  links), so "is this the active page" is decided in PYTHON instead (object
  identity against st.navigation()'s own return value, using the SAME
  st.Page objects passed to both calls -- confirmed reliable live), and
  encoded into each link's container key so CSS can still colour it.
- Each st.Page's own .visibility is read back and honoured here: the hidden
  legacy Betting Opportunities entry is skipped from the rendered sidebar
  (st.page_link() does not automatically respect a page's own hidden
  visibility the way the automatic menu does -- it would otherwise render a
  normal, visible link for it).

"Technical Summary" (pages/25_...) is the default landing page (default=True
below). The original Sportsbet-via-Markets-repo Betting Opportunities page
(pages/12_Betting_Opportunities.py) is superseded by the newer "Brownlow
Betting Opportunities" page for normal use -- its own visibility="hidden"
keeps it out of the rendered sidebar while its file and code remain
completely unmodified and it stays reachable by direct URL.

pages/1_Player_Detail.py wasn't named in any of the 4 requested groups --
placed under ADVANCED (closest in kind to Match Detail) so it stays
reachable, per "every existing page must remain accessible somewhere".

Every other existing page keeps its original filename -- st.navigation
references pages by path, not by relying on the pages/ directory's numeric-
prefix ordering, so there was no reason to rename anything else.
"""
import streamlit as st

from dashboard.data import SIDEBAR_SECTION_ACCENTS, nav_slug, sidebar_nav_css

pages = {
    "MAIN": [
        st.Page("pages/25_Technical_Summary.py", title="Technical Summary", default=True),
        st.Page("pages/24_Guide_And_FAQs.py", title="Guide & FAQs"),
        st.Page("pages/26_Finishing_Order.py", title="Finishing Order"),
        st.Page("pages/28_Player_Search.py", title="Player Search"),
        st.Page("pages/29_Player_Comparison.py", title="Player Comparison"),
        st.Page("pages/27_To_Poll_A_Vote.py", title="To Poll a Vote"),
        st.Page("pages/6_Match_Detail.py", title="Match Detail"),
        st.Page("pages/23_Brownlow_Betting_Opportunities.py", title="Brownlow Betting Opportunities"),
        st.Page("pages/4_Brownlow_Night_Tracker.py", title="Brownlow Night Tracker"),
        # visibility="hidden" keeps it out of the rendered sidebar (see the
        # module docstring) while its file and code remain fully unmodified
        # and it stays reachable by direct URL. Kept in MAIN's list, not a
        # separate section dict key, so no empty "HIDDEN" group can render.
        st.Page("pages/12_Betting_Opportunities.py", title="Betting Opportunities", visibility="hidden"),
    ],
    "MODEL ANALYSIS": [
        st.Page("pages/00_Overview.py", title="Production Model"),
        st.Page("pages/13_Objective_Stats_Model.py", title="Objective Stats Model"),
        st.Page("pages/15_Model_Agreement.py", title="Model Agreement"),
        st.Page("pages/14_Order_Scenarios.py", title="Order Scenarios"),
        st.Page("pages/16_Player_H2H.py", title="Player H2H"),
        st.Page("pages/17_Multi_Player_Comparison.py", title="Multi Player Comparison"),
        st.Page("pages/18_Team_Player_Rankings.py", title="Team Player Rankings"),
    ],
    "EXTERNAL VALIDATION": [
        st.Page("pages/19_External_Overview.py", title="External Overview"),
        st.Page("pages/20_External_Player_Comparison.py", title="External Player Comparison"),
        st.Page("pages/21_External_Winning_Order.py", title="External Winning Order"),
        st.Page("pages/22_External_Leader_After_Round.py", title="External Leader After Round"),
    ],
    "ADVANCED": [
        st.Page("pages/1_Player_Detail.py", title="Player Detail"),
        st.Page("pages/2_Scenario_Comparison.py", title="Scenario Comparison"),
        st.Page("pages/3_Model_Disagreement.py", title="Model Disagreement"),
        st.Page("pages/5_Round_View.py", title="Round View"),
        st.Page("pages/7_Defender_Bias_Watchlist.py", title="Defender Bias Watchlist"),
        st.Page("pages/8_Projection_Concentration.py", title="Projection Concentration"),
        st.Page("pages/9_Uncertainty.py", title="Uncertainty"),
        st.Page("pages/10_Round_By_Round_Leaderboard.py", title="Round By Round Leaderboard"),
        st.Page("pages/11_Team_Breakdown.py", title="Team Breakdown"),
    ],
}

# position="hidden": st.navigation still owns routing/registration (required
# for st.page_link() and st.switch_page() to resolve pages at all), but its
# own automatic menu is never displayed -- the custom sidebar built below is
# the only visible navigation.
current_page = st.navigation(pages, position="hidden")

SECTION_DEFAULT_EXPANDED = {
    "MAIN": True,
    "MODEL ANALYSIS": False,
    "EXTERNAL VALIDATION": False,
    "ADVANCED": False,
}

# One global stylesheet, injected exactly once here -- not per page. Every
# individual page file still calls dashboard.data.highlight_objective_stats_nav()
# (27 existing call sites, left untouched since page contents are out of
# scope for this change), but that function is now an intentional no-op: its
# old target elements (the automatic menu's stSidebarNavLink/
# stNavSectionHeader) no longer exist once the automatic menu is hidden, so
# nothing would happen even if it weren't a no-op -- see its docstring in
# dashboard/data.py. This avoids injecting the same styling twice. The CSS
# itself is built by the shared, unit-tested sidebar_nav_css() rather than
# inline here, so its correctness can be verified without a live browser.
st.markdown(f"<style>{sidebar_nav_css(SIDEBAR_SECTION_ACCENTS)}</style>", unsafe_allow_html=True)

with st.sidebar:
    for section, section_pages in pages.items():
        section_slug = nav_slug(section)
        with st.expander(
            section,
            expanded=SECTION_DEFAULT_EXPANDED.get(section, False),
            key=f"navgroup_{section_slug}",
        ):
            for p in section_pages:
                if p.visibility == "hidden":
                    continue
                is_active = p is current_page
                page_slug = nav_slug(p.title)
                link_key = (
                    f"navlink_{section_slug}_active_{page_slug}"
                    if is_active
                    else f"navlink_{section_slug}_{page_slug}"
                )
                with st.container(key=link_key):
                    st.page_link(p)

current_page.run()
