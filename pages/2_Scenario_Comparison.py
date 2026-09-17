import plotly.graph_objects as go
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Scenario Comparison", layout="wide")
st.title("Scenario Comparison")
st.caption(
    "Historical / recent-era / stats-assisted / structural-break variants / final ensemble, "
    "per contender. Source: reports/2026_scenario_comparison.csv (unmodified)."
)

sc = d.load_scenario_comparison().sort_values("FINAL_ENSEMBLE", ascending=False)
n = st.slider("Number of players to show", 5, len(sc), 20)
top = sc.head(n)

cols = {
    "player_name": "Player", "team_id": "Team",
    "A_historical": "Historical", "B_recent_era": "Recent-Era",
    "C_stats_assisted": "Stats-Assisted", "D_low": "Struct-Break Low",
    "D_medium": "Struct-Break Med", "D_high": "Struct-Break High",
    "FINAL_ENSEMBLE": "Final Ensemble",
    "structural_break_sensitivity": "Struct-Break Sensitivity",
    "model_disagreement_range": "Model Disagreement",
}
view = top.copy()
view["team_id"] = view["team_id"].str.replace("_", " ").str.title()
view = view[list(cols.keys())].rename(columns=cols)

sort_col = st.selectbox("Sort by", list(cols.values())[2:], index=len(cols) - 3)
view = view.sort_values(sort_col, ascending=False)

st.dataframe(
    view.style.format({c: "{:.1f}" for c in cols.values() if c not in ("Player", "Team")}),
    use_container_width=True, hide_index=True, height=650,
)
st.download_button("Export scenario comparison CSV", view.to_csv(index=False).encode("utf-8"),
                    file_name="2026_scenario_comparison_view.csv", mime="text/csv")

st.divider()
st.subheader("Scenario EV Comparison Chart")
chart_players = top.sort_values("FINAL_ENSEMBLE", ascending=False).head(12)
fig = go.Figure()
for scen, label in [
    ("A_historical", "Historical"), ("B_recent_era", "Recent-Era"),
    ("C_stats_assisted", "Stats-Assisted"), ("FINAL_ENSEMBLE", "Final Ensemble"),
]:
    fig.add_trace(go.Bar(name=label, x=chart_players["player_name"], y=chart_players[scen]))
fig.update_layout(
    barmode="group", template="plotly_dark", height=480,
    yaxis_title="Expected Votes", legend_title="Scenario",
    margin=dict(t=20, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Most Scenario-Sensitive Players (this view)")
most_sensitive = top.sort_values("structural_break_sensitivity", ascending=False).head(5)
st.caption("Largest spread between historical / recent / stats-assisted scenario EVs -- these projections depend most on the unresolved 2026 structural-break assumption.")
st.dataframe(
    most_sensitive[["player_name", "A_historical", "C_stats_assisted", "structural_break_sensitivity"]]
    .rename(columns={"player_name": "Player", "A_historical": "Historical EV", "C_stats_assisted": "Stats-Assisted EV", "structural_break_sensitivity": "Sensitivity"})
    .style.format({"Historical EV": "{:.1f}", "Stats-Assisted EV": "{:.1f}", "Sensitivity": "{:.2f}"}),
    use_container_width=True, hide_index=True,
)
