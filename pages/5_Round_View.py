import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Round View", layout="wide")
st.title("Round View")

meta = d.match_meta_table()
pv = d.load_predicted_votes()
mp = d.load_match_probabilities()

rounds = sorted(meta["round"].astype(int).unique().tolist())
selected_round = st.selectbox("Select round", rounds, index=0)

round_matches = meta[meta["round"].astype(int) == selected_round].sort_values("match_id")
st.caption(f"{len(round_matches)} match(es) in round {selected_round}")

for _, m in round_matches.iterrows():
    with st.container(border=True):
        st.markdown(f"### {m['result_label']}")
        st.caption(f"{m['venue']} -- {m['date']}")

        match_pv = pv[pv["match_id"] == m["match_id"]].sort_values("predicted_votes", ascending=False)
        cols = st.columns(3)
        for i, (_, r) in enumerate(match_pv.iterrows()):
            with cols[i]:
                st.metric(f"Predicted {int(r['predicted_votes'])} vote(s)", r["player_name"],
                          f"{r['team_id'].replace('_', ' ').title()}")
                st.caption(f"P(3)={r['p3']:.0%}  P(2)={r['p2']:.0%}  P(1)={r['p1']:.0%}  EV={r['expected_votes']:.2f}")

        match_mp = mp[mp["match_id"] == m["match_id"]].sort_values("expected_votes", ascending=False).head(8)
        with st.expander("Full leading-candidate probabilities for this match"):
            st.dataframe(
                match_mp[["player_name", "team_id", "p3", "p2", "p1", "p0", "expected_votes"]]
                .rename(columns={"player_name": "Player", "team_id": "Team", "p3": "P(3)", "p2": "P(2)", "p1": "P(1)", "p0": "P(0)", "expected_votes": "Expected Votes"})
                .style.format({"P(3)": "{:.0%}", "P(2)": "{:.0%}", "P(1)": "{:.0%}", "P(0)": "{:.0%}", "Expected Votes": "{:.2f}"}),
                use_container_width=True, hide_index=True,
            )
