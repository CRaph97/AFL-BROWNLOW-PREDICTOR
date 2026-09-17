"""
2026 Brownlow Model -- Review Dashboard (landing page).

Read-only presentation layer over the frozen, audited Phase 4 / 2026
production outputs. Run with:

    streamlit run app.py

See dashboard/data.py for the loading/derivation layer and README.md for
full documentation.
"""
import pandas as pd
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="2026 Brownlow Model", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    /* Streamlit's default text colors assume a light page background (often a
       translucent dark grey), which reads as near-invisible once the page
       background is forced dark above. Override with concrete light colors
       sized for a near-black background, without switching themes. */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #f0f6fc !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp li {
        color: #c9d1d9;
    }
    .stApp [data-testid="stCaptionContainer"] {
        color: #9da7b3 !important;
    }

    .metric-card {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 14px 18px; margin-bottom: 8px;
    }
    .metric-card [data-testid="stMetricLabel"] {
        color: #9da7b3 !important;
    }
    .metric-card [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-weight: 600;
    }
    /* Metric delta ("green status chip") color is left to Streamlit's own
       semantic (positive/negative) styling -- not overridden here, only kept
       legible via the surrounding label/value/card contrast above. */
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("2026 BROWNLOW MODEL")
st.caption(
    "Final production forecast, frozen and audited (see docs/2026_FINAL_AUDIT.md). "
    "This dashboard only displays and re-derives presentational fields from that "
    "output -- it never retrains or re-weights the model."
)

lb = d.load_leaderboard()
qc = d.load_quality_checks()

top = lb.iloc[0]
most_sensitive = lb.loc[lb["structural_break_sensitivity"].idxmax()]
most_disagree = lb.loc[lb["model_disagreement_range"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("#1 Projected Player", top["player_name"], f"{top['FINAL_ENSEMBLE']:.1f} EV")
    st.caption(f"Median {top['sim_median_votes']:.0f} | 80% [{top['sim_p10']:.0f}, {top['sim_p90']:.0f}] | 95% [{top['sim_p2_5']:.0f}, {top['sim_p97_5']:.0f}]")
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Highest Structural-Break Sensitivity", most_sensitive["player_name"], f"{most_sensitive['structural_break_sensitivity']:.2f}")
    st.caption("Spread across historical / recent / stats-assisted scenario EVs")
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Highest Model Disagreement", most_disagree["player_name"], f"{most_disagree['model_disagreement_range']:.2f}")
    st.caption("Range across all scenario models for this player")
    st.markdown("</div>", unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    status = "PASS" if qc.get("all_matches_covered") else "FAIL"
    st.metric("Data Integrity", status, f"{qc['n_2026_matches_covered_by_ensemble']}/{qc['n_2026_matches_total']} matches")
    st.caption(f"Season total EV = {qc['actual_season_total_expected_votes']:.1f} (expect {6*qc['n_2026_matches_total']})")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.subheader("Top 20 Leaderboard")

display_cols = {
    "rank": "Rank",
    "player_name": "Player",
    "team_id": "Team",
    "FINAL_ENSEMBLE": "Expected Votes",
    "sim_median_votes": "Median",
    "sim_p10": "80% Low",
    "sim_p90": "80% High",
    "sim_p2_5": "95% Low",
    "sim_p97_5": "95% High",
    "projected_3_vote_games": "Projected 3s",
    "projected_2_vote_games": "Projected 2s",
    "projected_1_vote_games": "Projected 1s",
    "model_disagreement_range": "Model Disagreement",
    "structural_break_sensitivity": "Structural-Break Sensitivity",
}
top20 = lb.head(20).copy()
top20["team_id"] = top20["team_id"].str.replace("_", " ").str.title()
view = top20[list(display_cols.keys())].rename(columns=display_cols)

styled = (
    view.style
    .format(
        {
            "Expected Votes": "{:.1f}",
            "Median": "{:.0f}",
            "80% Low": "{:.0f}",
            "80% High": "{:.0f}",
            "95% Low": "{:.0f}",
            "95% High": "{:.0f}",
            "Model Disagreement": "{:.2f}",
            "Structural-Break Sensitivity": "{:.2f}",
        }
    )
    .apply(d.gradient_style, subset=["Expected Votes"])
    .apply(lambda s: d.gradient_style(s, rgb=(232, 163, 61)), subset=["Structural-Break Sensitivity"])
    .apply(lambda s: d.gradient_style(s, rgb=(232, 163, 61)), subset=["Model Disagreement"])
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=760)

st.download_button(
    "Export Top 20 Leaderboard CSV",
    top20.to_csv(index=False).encode("utf-8"),
    file_name="2026_leaderboard_top20.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Use the sidebar to open Player Detail, Scenario Comparison, Model Disagreement, "
    "Round View, Match Detail, Defender Bias Watchlist, Projection Concentration, "
    "Uncertainty, and the Brownlow Night Tracker."
)
