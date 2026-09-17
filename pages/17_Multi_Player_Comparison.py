import pandas as pd
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Multi-Player Comparison", layout="wide")
d.highlight_objective_stats_nav()
st.title("Multi-Player Comparison")
st.caption(
    "Build a list of 2-10 players to compare side-by-side under both models "
    "(dashboard.data.load_dual_model_comparison() -- same canonical join as "
    "every other dual-model page)."
)

cmp = d.load_dual_model_comparison()
names = sorted(cmp["player_name"].tolist())
default = [n for n in ["Nick Daicos", "Bailey Smith", "Marcus Bontempelli"] if n in names] or names[:2]

selected = st.multiselect(
    "Add / remove players (2-10)", names, default=default, max_selections=10,
)

if len(selected) < 2:
    st.warning("Select at least 2 players.")
    st.stop()

sub = cmp[cmp["player_name"].isin(selected)].copy()
sub["Team"] = sub["team_id"].apply(d._display_team)

# Numeric columns are left as float (NaN renders blank) rather than mixing in
# an "Not available" string -- Arrow (Streamlit's dataframe serialisation)
# cannot represent a str+float mixed column cleanly. Missing players are
# still called out explicitly, by name, in the captions below the tables --
# never silently shown as a blank/0 with no explanation.
display = pd.DataFrame({
    "Player": sub["player_name"],
    "Team": sub["Team"],
    "Production EV": sub["production_ev"].round(2),
    "Objective EV": sub["objective_ev"].round(2),
    "Production Rank": sub["production_rank"],
    "Objective Rank": sub["objective_rank"],
    "Model Difference": sub["model_difference"].round(2),
    "Absolute Difference": sub["absolute_difference"].round(2),
    "Average EV": sub["average_ev"].round(2),
})

st.subheader("Comparison Table")
st.caption("Sort by clicking any column header. Blank = not available in that model (see note below).")
st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Ordering Under Each Model")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**By Production rank**")
    prod_order = sub.sort_values("production_rank", na_position="last")[["player_name", "production_ev", "production_rank"]]
    prod_order.columns = ["Player", "Production EV", "Production Rank"]
    st.dataframe(prod_order.reset_index(drop=True), use_container_width=True, hide_index=True)
with c2:
    st.markdown("**By Objective rank**")
    obj_order = sub.sort_values("objective_rank", na_position="last")[["player_name", "objective_ev", "objective_rank"]]
    obj_order.columns = ["Player", "Objective EV", "Objective Rank"]
    st.dataframe(obj_order.reset_index(drop=True), use_container_width=True, hide_index=True)

missing_prod = sub[~sub["in_production"]]["player_name"].tolist()
missing_obj = sub[~sub["in_objective"]]["player_name"].tolist()
if missing_prod:
    st.caption(f"Not available in Production: {', '.join(missing_prod)}")
if missing_obj:
    st.caption(f"Not available in Objective: {', '.join(missing_obj)}")
