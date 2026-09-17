import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Projection Concentration", layout="wide")
d.highlight_objective_stats_nav()
st.title("Projection Concentration")
st.caption(
    "Generic version of the Daicos-projection audit (docs/2026_FINAL_AUDIT.md, section 2) -- "
    "works for any selected player. Checks whether a high season total is broadly earned across "
    "many games (supported) or concentrated in a handful of outlier probabilities (a possible "
    "over-concentration artifact)."
)

lb = d.load_leaderboard()
names = lb.sort_values("FINAL_ENSEMBLE", ascending=False)["player_name"].tolist()
selected = st.selectbox("Select player", names, index=0)
row = lb[lb["player_name"] == selected].iloc[0]
pid = int(row["player_id"])

rbr = d.build_player_round_by_round(pid)
if rbr.empty:
    st.warning("No round-by-round data resolved for this player.")
    st.stop()

st.metric("Season Expected Votes", f"{row['FINAL_ENSEMBLE']:.1f}")

conc = d.projection_concentration(rbr)
conc["share_of_total_ev"] = conc["ev_contribution"] / rbr["expected_votes"].sum()

c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(
        conc.rename(columns={"bucket": "Bucket", "n_games": "Games", "ev_contribution": "EV Contribution", "share_of_total_ev": "Share of Total EV"})
        .style.format({"EV Contribution": "{:.2f}", "Share of Total EV": "{:.0%}"}),
        use_container_width=True, hide_index=True,
    )
with c2:
    st.bar_chart(conc.set_index("bucket")["n_games"], height=280)

max_single_p3 = rbr["p3"].max()
n_zero = (rbr["expected_votes"] < 0.05).sum()
n_games = len(rbr)

st.divider()
st.subheader("Over-Concentration Check")
st.write(
    f"- Max single-game P(3): **{max_single_p3:.0%}**\n"
    f"- Games with EV >= 1.0: **{conc.loc[conc['bucket'] == 'EV >= 1.0', 'n_games'].iloc[0]} of {n_games}**\n"
    f"- Zero/near-zero-vote games: **{n_zero} of {n_games}**\n\n"
    "A season total supported by broad accumulation across many EV>=1.0 games (rather than "
    "one or two extreme single-game probabilities) is not evidence of a probability-concentration "
    "artifact. A total driven mostly by a small number of games with very high single-game "
    "P(3) and otherwise mostly zero-vote games would warrant further scrutiny."
)
