import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Player Detail", layout="wide")
d.highlight_objective_stats_nav()
st.title("Player Detail")

lb = d.load_leaderboard()
names = lb.sort_values("FINAL_ENSEMBLE", ascending=False)["player_name"].tolist()
selected = st.selectbox("Select player", names, index=0)
row = lb[lb["player_name"] == selected].iloc[0]
pid = int(row["player_id"])

st.subheader(f"{selected} ({row['team_id'].replace('_', ' ').title()})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Votes", f"{row['FINAL_ENSEMBLE']:.1f}")
c1.metric("Median", f"{row['sim_median_votes']:.0f}")
c2.metric("80% Range", f"[{row['sim_p10']:.0f}, {row['sim_p90']:.0f}]")
c2.metric("95% Range", f"[{row['sim_p2_5']:.0f}, {row['sim_p97_5']:.0f}]")
c3.metric("Projected 3s / 2s / 1s", f"{row['projected_3_vote_games']} / {row['projected_2_vote_games']} / {row['projected_1_vote_games']}")
c3.metric("Structural-Break Sensitivity", f"{row['structural_break_sensitivity']:.2f}")
c4.metric("Model Disagreement", f"{row['model_disagreement_range']:.2f}")
c4.metric("Reputation Effect", "VOID", help="Not valid for interpretation: the A_with_reputation sensitivity scenario is contaminated by same-season vote information (data/canonical/contaminated_artefacts.json).")

rbr = d.build_player_round_by_round(pid)

if rbr.empty:
    st.warning("No round-by-round data resolved for this player (identity-resolution gap).")
    st.stop()

st.divider()
st.subheader("Round-by-Round")

def confidence_label(p3):
    if p3 >= 0.55:
        return "HIGH"
    if p3 >= 0.30:
        return "MEDIUM"
    return "LOW"

table = rbr.copy()
table["most_likely_votes"] = table["deterministic_pick"]
table["confidence"] = table["p3"].map(confidence_label)
display = table[[
    "round", "opponent_display", "result", "expected_votes", "p3", "p2", "p1",
    "most_likely_votes", "confidence", "primary_drivers",
]].rename(columns={
    "round": "Round", "opponent_display": "Opponent", "result": "Result",
    "expected_votes": "Expected Votes", "p3": "P(3)", "p2": "P(2)", "p1": "P(1)",
    "most_likely_votes": "Most Likely", "confidence": "Confidence", "primary_drivers": "Key Drivers",
})
st.dataframe(
    display.style.format({"Expected Votes": "{:.2f}", "P(3)": "{:.0%}", "P(2)": "{:.0%}", "P(1)": "{:.0%}"})
    .apply(d.gradient_style, subset=["Expected Votes"]),
    use_container_width=True, hide_index=True, height=500,
)
st.download_button(
    "Export round-by-round CSV", display.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected.replace(' ', '_')}_round_by_round.csv", mime="text/csv",
)

st.divider()
st.subheader("Cumulative Expected Votes by Round")
cum = table.sort_values("round").set_index("round")["expected_votes"].cumsum()
st.line_chart(cum, height=280)

st.divider()
st.subheader("Significant Games")
sig = d.classify_significant_games(rbr)
labels = {
    "high_confidence_3": "HIGH-CONFIDENCE 3-VOTE GAMES  (P(3) >= 0.55)",
    "likely_2_3": "LIKELY 2-3 VOTE GAMES  (1.5 <= EV < 2.5, not already high-confidence)",
    "borderline": "BORDERLINE POLLING GAMES  (0.8 <= EV < 1.5)",
    "surprising": "SURPRISING POTENTIAL POLLING GAMES  (EV >= 1.0 but not model's deterministic pick)",
    "zero_despite_strong": "LIKELY ZERO-VOTE GAMES DESPITE STRONG RAW STATS  (top driver z >= 1.5, EV < 0.5)",
}
for key, label in labels.items():
    sub = sig[key]
    with st.expander(f"{label}  --  {len(sub)} game(s)"):
        if sub.empty:
            st.caption("None.")
        else:
            st.dataframe(
                sub[["round", "opponent_display", "result", "expected_votes", "p3", "primary_drivers"]]
                .rename(columns={"opponent_display": "opponent"}),
                use_container_width=True, hide_index=True,
            )

st.divider()
st.subheader("Projection Concentration")
st.caption("See the dedicated Projection Concentration page for the full generic breakdown.")
st.dataframe(d.projection_concentration(rbr), use_container_width=True, hide_index=True)
