"""
Entrypoint / navigation router for the 2026 Brownlow dashboard.

This file used to CONTAIN the "Guide & FAQs" page's body directly. Streamlit
requires the entrypoint to become a pure router as soon as any session calls
st.navigation() -- the pages/ directory's filename-based auto-discovery is
then ignored entirely -- so Guide & FAQs' content was relocated, unchanged,
to pages/24_Guide_And_FAQs.py.

Sidebar structure: grouped into MAIN / MODEL ANALYSIS / EXTERNAL VALIDATION /
ADVANCED sections. NOTE on a real Streamlit platform limitation (verified
against the installed version, not assumed): st.navigation's per-section
labels are only individually collapsible/expandable when position="top". For
the traditional left sidebar (position="sidebar", kept here -- switching to
top nav is a much bigger visual change than a nav cleanup and was not asked
for), section headers render as static, non-collapsible group labels; the
only related control is `expanded`, which governs a single "View X more /
View less" truncation of the WHOLE menu, not per-section collapse. Set to
True here so nothing is hidden behind that click, since a materially large
"View more" list would defeat the point of grouping pages for readability.

"Technical Summary" (pages/25_...) is the default landing page (default=True
below). The original Sportsbet-via-Markets-repo Betting Opportunities page
(pages/12_Betting_Opportunities.py) is superseded by the newer "Brownlow
Betting Opportunities" page for normal use: visibility="hidden" removes it
from the visible nav menu while leaving its file and code completely
unmodified and still reachable by direct URL, per Streamlit's own documented
behaviour for that parameter.

pages/1_Player_Detail.py wasn't named in any of the 4 requested groups --
placed under ADVANCED (closest in kind to Match Detail) so it stays
reachable, per "every existing page must remain accessible somewhere".

Every other existing page keeps its original filename -- st.navigation
references pages by path, not by relying on the pages/ directory's numeric-
prefix ordering, so there was no reason to rename anything else.
"""
import streamlit as st

pages = {
    "MAIN": [
        st.Page("pages/25_Technical_Summary.py", title="Technical Summary", default=True),
        st.Page("pages/24_Guide_And_FAQs.py", title="Guide & FAQs"),
        st.Page("pages/26_Finishing_Order.py", title="Finishing Order"),
        st.Page("pages/28_Player_Search.py", title="Player Search"),
        st.Page("pages/27_To_Poll_A_Vote.py", title="To Poll a Vote"),
        st.Page("pages/6_Match_Detail.py", title="Match Detail"),
        st.Page("pages/23_Brownlow_Betting_Opportunities.py", title="Brownlow Betting Opportunities"),
        st.Page("pages/4_Brownlow_Night_Tracker.py", title="Brownlow Night Tracker"),
        # visibility="hidden" excludes it from the rendered nav menu while
        # keeping the page fully intact and reachable by direct URL (see
        # module docstring). Placed in MAIN's list, not a separate section
        # dict key, so no empty "HIDDEN" group label can ever render.
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

current_page = st.navigation(pages, expanded=True)
current_page.run()
