import pandas as pd
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Team Player Rankings", layout="wide")
d.highlight_objective_stats_nav()
st.title("Team Player Rankings")
st.caption(
    "All players for one team, ranked by total season votes under both models "
    "(team-relative ranks computed on full-precision EV via "
    "dashboard.data.dual_model_team_rankings())."
)

cmp = d.load_dual_model_comparison()
teams = sorted(cmp["team_id"].dropna().unique().tolist())
team_labels = {t: d._display_team(t) for t in teams}
selected_label = st.selectbox("Select team", [team_labels[t] for t in teams])
selected_team = [t for t, lab in team_labels.items() if lab == selected_label][0]

sub = d.dual_model_team_rankings(selected_team)


# Numeric columns stay float (NaN renders blank) rather than mixing a string
# into a numeric column, which breaks Streamlit's Arrow serialisation.
# Missing players are called out explicitly by name in the caption below.
display = pd.DataFrame({
    "Player": sub["player_name"],
    "Production EV": sub["production_ev"].round(2),
    "Production Team Rank": sub["production_team_rank"],
    "Objective EV": sub["objective_ev"].round(2),
    "Objective Team Rank": sub["objective_team_rank"],
    "Absolute Difference": sub["absolute_difference"].round(2),
    "Average EV": sub["average_ev"].round(2),
}).sort_values("Production Team Rank", na_position="last")

st.subheader(f"{selected_label} — all players")
st.caption("Blank = not available in that model (see note below).")
st.dataframe(display.reset_index(drop=True), use_container_width=True, hide_index=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Production team ranking**")
    po = sub.sort_values("production_team_rank")[["player_name", "production_ev", "production_team_rank"]].dropna(subset=["production_team_rank"])
    po.columns = ["Player", "Production EV", "Team Rank"]
    st.dataframe(po.reset_index(drop=True), use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Objective team ranking**")
    oo = sub.sort_values("objective_team_rank")[["player_name", "objective_ev", "objective_team_rank"]].dropna(subset=["objective_team_rank"])
    oo.columns = ["Player", "Objective EV", "Team Rank"]
    st.dataframe(oo.reset_index(drop=True), use_container_width=True, hide_index=True)

n_missing_prod = (~sub["in_production"]).sum()
n_missing_obj = (~sub["in_objective"]).sum()
if n_missing_prod or n_missing_obj:
    st.caption(
        f"{n_missing_prod} player(s) not available in Production, "
        f"{n_missing_obj} not available in Objective, for {selected_label}."
    )
