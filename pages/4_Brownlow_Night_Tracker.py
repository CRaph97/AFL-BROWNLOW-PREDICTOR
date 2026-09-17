import pandas as pd
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Brownlow Night Tracker", layout="wide")
st.title("Brownlow Night Tracker")
st.caption(
    "Local, session-only. Enter each round's ACTUAL 3-2-1 votes as they are read out on "
    "count night to compare against the model's predicted cumulative votes. This never "
    "overwrites the model's predictions -- it only tracks a separate 'actual' column "
    "alongside them. Nothing here is saved between browser sessions."
)

lb = d.load_leaderboard()
names = lb.sort_values("FINAL_ENSEMBLE", ascending=False)["player_name"].tolist()

if "actual_votes" not in st.session_state:
    st.session_state.actual_votes = {}  # {(player_name, round): votes}

selected = st.selectbox("Player to track", names, index=0)
pid = int(lb[lb["player_name"] == selected].iloc[0]["player_id"])
rbr = d.build_player_round_by_round(pid)

if rbr.empty:
    st.warning("No round-by-round data resolved for this player.")
    st.stop()

st.subheader(f"Enter actual votes for {selected}")
rounds = sorted(rbr["round"].astype(int).unique().tolist())
cols = st.columns(6)
for i, rnd in enumerate(rounds):
    key = (selected, rnd)
    with cols[i % 6]:
        val = st.number_input(f"R{rnd}", min_value=0, max_value=3, step=1,
                               value=st.session_state.actual_votes.get(key, 0),
                               key=f"actual_{selected}_{rnd}")
        st.session_state.actual_votes[key] = val

predicted = rbr.set_index("round")["expected_votes"].reindex(rounds).fillna(0)
actual = pd.Series({r: st.session_state.actual_votes.get((selected, r), 0) for r in rounds})

compare = pd.DataFrame({
    "Round": rounds,
    "Predicted (round)": predicted.values,
    "Actual (round)": actual.values,
})
compare["Predicted Cumulative"] = compare["Predicted (round)"].cumsum()
compare["Actual Cumulative"] = compare["Actual (round)"].cumsum()
compare["Difference"] = compare["Actual Cumulative"] - compare["Predicted Cumulative"]

def status(diff):
    if diff > 1.0:
        return "AHEAD OF MODEL"
    if diff < -1.0:
        return "BEHIND MODEL"
    return "ON MODEL"

compare["Status"] = compare["Difference"].map(status)

st.divider()
st.dataframe(
    compare.style.format({
        "Predicted (round)": "{:.2f}", "Actual (round)": "{:.0f}",
        "Predicted Cumulative": "{:.2f}", "Actual Cumulative": "{:.0f}",
        "Difference": "{:+.2f}",
    }),
    use_container_width=True, hide_index=True,
)

st.subheader("Cumulative Predicted vs. Actual")
chart_df = compare.set_index("Round")[["Predicted Cumulative", "Actual Cumulative"]]
st.line_chart(chart_df, height=320)

final_diff = compare["Difference"].iloc[-1] if len(compare) else 0.0
st.metric("Final Status", status(final_diff), f"{final_diff:+.2f} votes vs. model")
