"""
Player Search -- a clean, non-betting-first cross-model profile for ANY
canonical 2026 player (not only bookmaker-market players).

Read-only over already-computed outputs. Reuses, rather than duplicates:
- dashboard.player_search for the season snapshot / trajectory (which
  themselves reuse dashboard.data's dual-model comparison and
  dashboard.finishing_order's simulation loaders)
- dashboard.betting_opportunities.match_level_drilldown() for "Likely
  Polling Rounds" (the exact same drill-down as pages/27_To_Poll_A_Vote.py)
- the same st.session_state["match_detail_preselect_match_id"] +
  st.switch_page("pages/6_Match_Detail.py") deep-link mechanism
- the same betting-opportunities loader/display pipeline for "Available
  bookmaker markets"

No new modelling, no new feature engineering, no new scoring system.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import betting_opportunities as bo
from dashboard import data as d
from dashboard import player_search as ps

st.set_page_config(page_title="Player Search", layout="wide")
d.highlight_objective_stats_nav()
st.title("Player Search")
st.caption(
    "A cross-model profile for any 2026 player -- Production, Objective, and Wheelo shown "
    "side by side, never blended into one number. Analysis-first; bookmaker markets (if any) "
    "are shown last, as secondary context."
)

comparison = ps.canonical_player_table()
players = sorted(comparison["player_name"].dropna().unique())
choice = st.selectbox("Search for a player", players, key="player_search_choice")
row = comparison[comparison["player_name"] == choice].iloc[0]
player_id = row["player_id"]

st.divider()

# ==========================================================================
# Season Snapshot
# ==========================================================================
st.header("Season Snapshot")
snap = ps.season_snapshot(player_id)
if not snap:
    st.caption("No season data resolved for this player.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Production")
        p = snap["production"]
        st.metric("Season EV", f"{p['ev']:.2f}" if p["ev"] is not None else "N/A")
        st.caption(f"Overall rank: #{p['rank']}" if p["rank"] else "Overall rank: N/A")
        st.caption(f"Team rank: #{p['team_rank']}" if p["team_rank"] else "Team rank: N/A")
        st.caption(f"Season P(any vote): {bo.format_pct(p['p_any'])}" if p["p_any"] is not None else "Season P(any vote): N/A")
    with c2:
        st.subheader("Objective")
        o = snap["objective"]
        st.metric("Season EV", f"{o['ev']:.2f}" if o["ev"] is not None else "N/A")
        st.caption(f"Overall rank: #{o['rank']}" if o["rank"] else "Overall rank: N/A")
        st.caption(f"Team rank: #{o['team_rank']}" if o["team_rank"] else "Team rank: N/A")
        st.caption(f"Season P(any vote): {bo.format_pct(o['p_any'])}" if o["p_any"] is not None else "Season P(any vote): N/A")
    with c3:
        st.subheader("Wheelo")
        w = snap["wheelo"]
        st.metric("Season EV", f"{w['ev']:.2f}" if w["ev"] is not None else "N/A")
        st.caption(f"Overall rank: #{w['rank']}" if w["rank"] else "Overall rank: N/A")
        st.caption(f"Team rank: #{w['team_rank']}" if w["team_rank"] else "Team rank: N/A")
        st.caption("No P(any)/P2/P1 -- Wheelo has no season simulation or per-match P2/P1 data.")

    if snap.get("model_gap") is not None:
        st.caption(
            f"**Model gap (|Production EV − Objective EV|):** {snap['model_gap']:.2f} votes -- "
            "shown as evidence of disagreement, never resolved into one blended number."
        )

st.divider()

# ==========================================================================
# Likely Polling Rounds (reuses the exact To Poll a Vote drill-down)
# ==========================================================================
st.header("Likely Polling Rounds")
drilldown = bo.match_level_drilldown(player_id)
if drilldown.empty:
    st.caption("No match-level data resolved for this player.")
else:
    top5 = drilldown.head(5).copy()
    summary = bo.drilldown_summary(drilldown)
    if "production" in summary:
        r, opp, pr = summary["production"]
        st.markdown(f"- **Strongest Production polling match:** Round {int(r)} vs {opp} (P(any vote) {pr * 100:.1f}%)")
    if "objective" in summary:
        r, opp, pr = summary["objective"]
        st.markdown(f"- **Strongest Objective polling match:** Round {int(r)} vs {opp} (P(any vote) {pr * 100:.1f}%)")
    if "wheelo" in summary:
        r, opp, pr = summary["wheelo"]
        st.markdown(f"- **Strongest Wheelo-supported match:** Round {int(r)} vs {opp} (P3-equivalent {pr:.1f}%)")

    st.dataframe(bo.build_drill_table(top5), use_container_width=True, hide_index=True)
    st.caption(
        "Wheelo has no P2/P1 data -- \"P(any)\" is never estimated for Wheelo, only shown "
        "for Production/Objective, computed as P3+P2+P1 from their own real match probabilities."
    )

    st.caption("Open the full match view:")
    link_cols = st.columns(len(top5))
    for i, (_, r) in enumerate(top5.iterrows()):
        with link_cols[i]:
            if st.button(f"R{r['round']} vs {r['opponent_display']}", key=f"ps_open_match_{player_id}_{r['match_id']}"):
                st.session_state["match_detail_preselect_match_id"] = r["match_id"]
                st.switch_page("pages/6_Match_Detail.py")

    st.divider()

    # ----------------------------------------------------------------------
    # Selected Match Evidence -- one selector, one round's stats (unlike
    # pages/27_To_Poll_A_Vote.py, which shows all 5 at once).
    # ----------------------------------------------------------------------
    st.subheader("Selected Match Evidence")
    round_options = [f"R{r['round']} vs {r['opponent_display']}" for _, r in top5.iterrows()]
    round_choice = st.selectbox("Choose a round to inspect", round_options, key="ps_round_choice")
    selected_row = top5.iloc[round_options.index(round_choice)]
    key_stats, any_missing = bo.build_key_stats_table(selected_row.to_frame().T)
    st.dataframe(key_stats, use_container_width=True, hide_index=True)
    if any_missing:
        st.caption("Some match-level stats are unavailable in the deployed dataset.")

st.divider()

# ==========================================================================
# Season trajectory
# ==========================================================================
st.header("Season Trajectory")
trajectory = ps.season_trajectory(player_id)
if trajectory.empty:
    st.caption("No round-by-round data resolved for this player.")
else:
    chart_data = trajectory.set_index("round")[["production_cumulative", "objective_cumulative", "wheelo_cumulative"]]
    chart_data = chart_data.rename(columns={
        "production_cumulative": "Production", "objective_cumulative": "Objective", "wheelo_cumulative": "Wheelo",
    })
    st.line_chart(chart_data, height=320)
    st.caption("Cumulative expected votes (Production/Objective) and predicted votes (Wheelo), by round.")

st.divider()

# ==========================================================================
# Available bookmaker markets (secondary)
# ==========================================================================
with st.expander("Available bookmaker markets", expanded=False):
    raw_opportunities = bo.load_opportunities()
    if raw_opportunities.empty:
        st.caption("No priced betting opportunities are currently available.")
    else:
        display = bo.prepare_display(raw_opportunities)
        player_markets = ps.player_bookmaker_markets(player_id, choice, display)
        if player_markets.empty:
            st.caption("No resolved bookmaker markets for this player currently.")
        else:
            piv = bo.with_bookmaker_odds(player_markets, ["market_type", "line", "side", "n", "position", "threshold"])
            show = pd.DataFrame({
                "Market": piv["market_label"],
                "Bookmaker": piv["best_bookmaker"],
                "Odds": piv["best_odds"].apply(bo.format_odds),
                "Production %": piv["production_probability"].apply(bo.format_pct),
                "Objective %": piv["objective_probability"].apply(bo.format_pct),
                "Likelihood²": piv["likelihood_display"],
                "Bet Value¹": piv["confidence_badge"],
            })
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(f"ℹ️ {bo.VALUE_SIGNAL_CAPTION}")
            st.caption(f"ℹ️ {bo.LIKELIHOOD_CAPTION}")
