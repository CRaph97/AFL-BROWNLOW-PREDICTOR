"""
To Poll a Vote -- dedicated page for the "will this player poll at least one
vote" market and its manual match-level cross-check drill-down.

Moved out of Brownlow Betting Opportunities (which now links here via a
compact summary card) so this workflow has room to breathe. Reuses the exact
same loaders/pricing/classification output as the main betting page --
nothing here recomputes a probability, edge, or classification; this file
only formats and organises already-priced rows for display, and adds a
"Open Match Detail" deep-link into an existing match-level row.

Never places a bet, sizes a stake, or automates a bookmaker account. Never
scrapes on page load: everything here is read from
data/betting/processed/*, built by `python scripts/refresh_brownlow_odds.py`.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import betting_opportunities as bo
from dashboard import data as d

st.set_page_config(page_title="To Poll a Vote", layout="wide")
d.highlight_objective_stats_nav()
st.title("To Poll a Vote")
st.caption(
    "Neds / PointsBet 'to poll a vote' markets priced against Production and Objective's own "
    "simulations, with Wheelo as independent corroboration. Decision-support only."
)
st.caption(f"ℹ️ {bo.VALUE_SIGNAL_CAPTION}")

raw_opportunities = bo.load_opportunities()
if raw_opportunities.empty:
    st.info(
        "**No priced betting opportunities are currently available.** Run "
        "`python scripts/refresh_brownlow_odds.py` to fetch fresh markets."
    )
    st.stop()

opportunities = bo.prepare_display(raw_opportunities)


def _render_polling_drilldown(prow) -> None:
    """Render one player's "why this bet?" match-level evidence panel,
    including a deep-link into Match Detail for each of their top-5 rounds.

    This is optional evidence display, not core page content -- a failure
    here (e.g. a deployed environment missing an optional CORE stat column)
    degrades to a graceful message for that one player only, per the
    try/except at the call site below.
    """
    drilldown = bo.match_level_drilldown(prow["player_id"])
    if drilldown.empty:
        st.caption("No match-level data resolved for this player.")
        return
    summary = bo.drilldown_summary(drilldown)
    if "production" in summary:
        r, opp, p = summary["production"]
        st.markdown(f"- **Strongest Production polling match:** Round {int(r)} vs {opp} (P(any vote) {p * 100:.1f}%)")
    if "objective" in summary:
        r, opp, p = summary["objective"]
        st.markdown(f"- **Strongest Objective polling match:** Round {int(r)} vs {opp} (P(any vote) {p * 100:.1f}%)")
    if "wheelo" in summary:
        r, opp, p = summary["wheelo"]
        st.markdown(f"- **Strongest Wheelo-supported match:** Round {int(r)} vs {opp} (P3-equivalent {p:.1f}%)")
    if "models_agree" in summary:
        st.markdown(f"- **Production and Objective point at the same game:** {'Yes' if summary['models_agree'] else 'No'}")

    top5 = drilldown.head(5).copy()
    drill_table = bo.build_drill_table(top5)
    st.dataframe(drill_table, use_container_width=True, hide_index=True)
    st.caption(
        "Wheelo has no P2/P1 data -- \"P(any)\" is never estimated for Wheelo, only shown "
        "for Production/Objective, computed as P3+P2+P1 from their own real match probabilities."
    )

    # Match deep-link: one "Open Match Detail" button per polling-round row.
    # Sets the match to preselect in session_state, then hands off to the
    # real, single Match Detail page -- no Match Detail rendering logic is
    # duplicated here.
    st.caption("Open the full match view:")
    link_cols = st.columns(len(top5))
    for i, (_, row) in enumerate(top5.iterrows()):
        with link_cols[i]:
            if st.button(f"R{row['round']} vs {row['opponent_display']}", key=f"open_match_{prow['player_id']}_{row['match_id']}"):
                st.session_state["match_detail_preselect_match_id"] = row["match_id"]
                st.switch_page("pages/6_Match_Detail.py")

    # Only show key-stat columns that are genuinely present (non-missing for
    # every one of these 5 matches) -- a column match_level_drilldown() had to
    # fill with NA because it's absent from this environment's CORE table is
    # omitted here entirely, rather than shown as a column of blanks.
    key_stats, any_missing = bo.build_key_stats_table(top5)
    st.caption("Existing match stats (key evidence, where recorded):")
    st.dataframe(key_stats, use_container_width=True, hide_index=True)
    if any_missing:
        st.caption("Some match-level stats are unavailable in the deployed dataset.")


tpav = opportunities[opportunities["market_type"] == "TO_POLL_A_VOTE"]
if tpav.empty:
    st.caption("No 'To Poll a Vote' markets currently available.")
else:
    fc1, fc2 = st.columns([2, 1])
    search3 = fc1.text_input("Player search", key="tpav_search")
    min_prob_choice = fc2.selectbox(
        "Minimum model probability", ["Any", "25%", "50%", "60%", "70%", "80%"], key="tpav_min_prob",
        help="Filters on min(Production, Objective) season probability -- or whichever one exists "
             "if only one model resolved this player. Separate from Value Signal.",
    )
    piv = bo.with_bookmaker_odds(tpav, ["player_id"])
    if search3:
        piv = piv[piv["player_name"].str.contains(search3, case=False, na=False)]
    if min_prob_choice != "Any":
        threshold = int(min_prob_choice.rstrip("%")) / 100.0
        piv = piv[piv["conservative_internal_probability"] >= threshold]
    piv = piv.assign(_sort=piv["conservative_internal_probability"] - piv["implied_probability"]).sort_values("_sort", ascending=False)
    show = pd.DataFrame({
        "Player": piv["player_name"],
        "Neds odds": piv["neds_odds"].apply(bo.format_odds),
        "PointsBet odds": piv["pointsbet_odds"].apply(bo.format_odds),
        "Best odds": piv["best_odds"].apply(bo.format_odds),
        "Implied %": piv["implied_probability"].apply(bo.format_pct),
        "Production %": piv["production_probability"].apply(bo.format_pct),
        "Objective %": piv["objective_probability"].apply(bo.format_pct),
        "Wheelo support": piv["wheelo_support_label"],
        "Value Signal": piv["confidence_badge"],
    })
    st.dataframe(show, use_container_width=True, hide_index=True, height=500)

    st.caption(
        "Manual cross-check: pick a player below to see their top 5 real matches most likely "
        "to produce at least one vote, using only existing match-level data. This is evidence "
        "display only -- it never changes the season betting probability shown above."
    )
    if not piv.empty:
        drilldown_names = sorted(piv["player_name"].dropna().unique())
        drilldown_choice = st.selectbox("Check likely polling rounds", drilldown_names, key="tpav_drilldown_player")
        sel_row = piv[piv["player_name"] == drilldown_choice].iloc[0]
        try:
            _render_polling_drilldown(sel_row)
        except Exception:
            st.caption("Some match-level stats are unavailable in the deployed dataset.")
