import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Player H2H", layout="wide")
d.highlight_objective_stats_nav()
st.title("Player H2H Comparison")
st.caption(
    "Compares any two players under both models, from the same canonical "
    "dashboard.data.load_dual_model_comparison() join used by every "
    "dual-model page -- no separate calculation logic."
)

cmp = d.load_dual_model_comparison()
names = sorted(cmp["player_name"].tolist())

c1, c2 = st.columns(2)
player_a = c1.selectbox("Player A", names, index=names.index("Nick Daicos") if "Nick Daicos" in names else 0)
player_b = c2.selectbox("Player B", names, index=1 if len(names) > 1 else 0)

if player_a == player_b:
    st.warning("Select two different players.")
    st.stop()


def _card(col, name):
    row = cmp[cmp["player_name"] == name].iloc[0]
    col.subheader(name)
    if row["in_production"]:
        col.metric("Production EV", f"{row['production_ev']:.2f}", f"Rank #{int(row['production_rank'])}")
    else:
        col.metric("Production EV", "Not available", "excluded from Production's ensemble")
    if row["in_objective"]:
        col.metric("Objective EV", f"{row['objective_ev']:.2f}", f"Rank #{int(row['objective_rank'])}")
    else:
        col.metric("Objective EV", "Not available")
    if row["in_production"] and row["in_objective"]:
        col.metric("Model Midpoint", f"{row['average_ev']:.2f}")
        col.metric("Absolute Model Disagreement", f"{row['absolute_difference']:.2f}")
    return row


c1, c2 = st.columns(2)
row_a = _card(c1, player_a)
row_b = _card(c2, player_b)

st.divider()
st.subheader("Comparison")

if row_a["in_production"] and row_b["in_production"]:
    prod_gap = row_a["production_ev"] - row_b["production_ev"]
    prod_favours = player_a if prod_gap > 0 else (player_b if prod_gap < 0 else "Tied")
    st.write(f"**Production** rates **{prod_favours}** higher — EV gap: {abs(prod_gap):.2f} votes.")
else:
    st.write("**Production**: comparison not available (one or both players not in Production's ensemble).")

if row_a["in_objective"] and row_b["in_objective"]:
    obj_gap = row_a["objective_ev"] - row_b["objective_ev"]
    obj_favours = player_a if obj_gap > 0 else (player_b if obj_gap < 0 else "Tied")
    st.write(f"**Objective** rates **{obj_favours}** higher — EV gap: {abs(obj_gap):.2f} votes.")
else:
    st.write("**Objective**: comparison not available (one or both players not in Objective's leaderboard).")

if row_a["in_production"] and row_b["in_production"] and row_a["in_objective"] and row_b["in_objective"]:
    if prod_favours != obj_favours and prod_favours != "Tied" and obj_favours != "Tied":
        st.info(
            f"The two models **disagree on who ranks higher**: Production favours {prod_favours}, "
            f"Objective favours {obj_favours}. This kind of disagreement does not imply either model "
            f"is wrong -- see the Model Agreement and Uncertainty pages for context."
        )
