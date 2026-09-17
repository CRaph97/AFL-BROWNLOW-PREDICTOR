import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Team Breakdown", layout="wide")
st.title("Team Breakdown")
st.caption(
    "Per-team view built entirely from the frozen reports/2026_leaderboard.csv and "
    "reports/2026_match_probabilities.csv -- no probability or vote is recomputed here."
)

teams = d.team_list()
selected_team = st.selectbox(
    "Select team", teams, format_func=lambda t: t.replace("_", " ").title()
)

team_df = d.team_breakdown(selected_team)
if team_df.empty:
    st.warning("No players found for this team in the leaderboard.")
    st.stop()

team_total_ev = team_df["FINAL_ENSEMBLE"].sum()
top_poller = team_df.iloc[0]
n_projected_to_poll = int(
    ((team_df["projected_3_vote_games"] + team_df["projected_2_vote_games"] + team_df["projected_1_vote_games"]) > 0).sum()
)
concentration = d.team_concentration(team_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Team Total Expected Votes", f"{team_total_ev:.1f}")
c2.metric("Top Projected Poller", top_poller["player_name"], f"{top_poller['FINAL_ENSEMBLE']:.1f} EV")
c3.metric("Players Projected To Poll", n_projected_to_poll)
c4.metric("Vote Concentration (HHI)", f"{concentration:.2f}")
st.caption(
    "Vote Concentration is a Herfindahl-style index: the sum of each player's (share of team "
    "expected votes)^2, over players in this team's leaderboard rows. Closer to 1/n = evenly "
    "spread across n pollers; closer to 1.0 = concentrated in one or two players."
)

st.divider()
st.subheader("Players")
display_cols = team_df[
    [
        "player_name",
        "FINAL_ENSEMBLE",
        "sim_median_votes",
        "projected_3_vote_games",
        "projected_2_vote_games",
        "projected_1_vote_games",
        "share_of_team_ev",
    ]
].rename(
    columns={
        "player_name": "Player",
        "FINAL_ENSEMBLE": "Expected Season Votes",
        "sim_median_votes": "Median",
        "projected_3_vote_games": "Projected 3s",
        "projected_2_vote_games": "Projected 2s",
        "projected_1_vote_games": "Projected 1s",
        "share_of_team_ev": "Share of Team EV",
    }
)
st.dataframe(
    display_cols.style.format(
        {"Expected Season Votes": "{:.1f}", "Median": "{:.1f}", "Share of Team EV": "{:.0%}"}
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Round-by-Round (Material Games Only)")
st.caption(
    f"Player-match rows for this team with expected votes >= {d.TEAM_ROUND_EV_FLOOR} "
    "(the same threshold classify_significant_games() uses for a 'borderline polling game')."
)
rbr = d.team_round_by_round(selected_team)
st.dataframe(
    rbr.rename(
        columns={
            "round": "Round",
            "opponent": "Opponent",
            "player_name": "Player",
            "expected_votes": "Expected Votes",
            "most_likely_votes": "Most Likely Votes",
            "p3": "P(3)",
            "p2": "P(2)",
            "p1": "P(1)",
        }
    ).style.format({"Expected Votes": "{:.2f}", "P(3)": "{:.0%}", "P(2)": "{:.0%}", "P(1)": "{:.0%}"}),
    use_container_width=True,
    hide_index=True,
)
