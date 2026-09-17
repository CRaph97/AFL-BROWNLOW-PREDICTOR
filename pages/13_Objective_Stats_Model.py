"""
2026 Objective Stats Model -- Experimental. Read-only view over the standalone
objective-stats pipeline's own output files (src/models/objective_stats_model.py,
src/models/build_2026_objective_outputs.py). This page never recomputes a
probability or score -- it only displays what that pipeline already produced.
"""
import pandas as pd
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Objective Stats Model", layout="wide")
d.highlight_objective_stats_nav()
st.title("2026 Objective Stats Model — Experimental")

st.warning(
    """
**This is an independent, experimental comparison model -- not the production forecast.**

- It deliberately **ignores all historical Brownlow voting patterns** -- no historical votes,
  reputation, prior-season performance, or role bias enter it at all.
- It asks a different question: *"who looks most deserving from 2026 match statistics alone?"*
- It is **NOT trained or fitted against historical Brownlow votes** -- its weights are hand-specified
  football judgement calls, not statistically estimated coefficients.
- Its main value is **comparison with the production model** (see the Overview / Player Detail pages).
- Disagreement between the two models may identify players particularly affected by the 2026
  statistics-assisted voting process -- or may simply reflect a documented data gap (flagged where
  known). See `docs/2026_OBJECTIVE_MODEL.md` for full methodology, every weight's rationale, and a
  weight-sensitivity analysis.
"""
)

obj_lb = d.load_objective_leaderboard()
comp = d.load_objective_vs_production()

st.header("1. Objective Leaderboard")
n = st.slider("Players to show", 10, 50, 20, key="obj_lb_n")
view = obj_lb.head(n).merge(
    comp[["Player", "Production EV", "Production Rank"]],
    left_on="player_name", right_on="Player", how="left",
).drop(columns=["Player"])
view = view.rename(columns={
    "rank": "Objective Rank", "player_name": "Player", "team_id": "Team",
    "objective_ev": "Objective EV", "objective_3_games": "Objective 3s",
    "objective_2_games": "Objective 2s", "objective_1_games": "Objective 1s",
})
st.dataframe(
    view[["Objective Rank", "Player", "Team", "Objective EV", "Production EV",
          "Objective 3s", "Objective 2s", "Objective 1s", "Production Rank"]],
    use_container_width=True, hide_index=True,
)
st.download_button("Export Objective Leaderboard CSV", obj_lb.to_csv(index=False).encode("utf-8"),
                    file_name="2026_objective_leaderboard.csv", mime="text/csv")

comp_both = comp.dropna(subset=["Production EV", "Objective EV"]).copy()

col1, col2 = st.columns(2)
with col1:
    st.header("2. Objective Rates More Highly")
    st.caption("Players the objective model favours relative to the production model.")
    top_pos = comp_both.sort_values("Difference", ascending=False).head(10)
    st.dataframe(
        top_pos[["Player", "Team", "Production EV", "Objective EV", "Difference"]],
        use_container_width=True, hide_index=True,
    )
with col2:
    st.header("3. Production Rates More Highly")
    st.caption("Players the production (historical-pattern) model favours relative to the objective model.")
    top_neg = comp_both.sort_values("Difference").head(10)
    st.dataframe(
        top_neg[["Player", "Team", "Production EV", "Objective EV", "Difference"]],
        use_container_width=True, hide_index=True,
    )

st.caption(
    "Note: a small number of players show Production EV of exactly 0.0 -- a pre-existing, "
    "already-documented production-pipeline data gap (footywire join failure for those players), "
    "not a new finding from this model. See docs/2026_OBJECTIVE_MODEL.md section 11."
)

st.divider()
st.header("4. Player Round-by-Round: Objective vs. Production")
player_names = sorted(obj_lb["player_name"].unique())
default_idx = player_names.index(obj_lb.iloc[0]["player_name"]) if len(player_names) else 0
sel_name = st.selectbox("Player", player_names, index=default_idx)
sel_row = obj_lb[obj_lb["player_name"] == sel_name].iloc[0]

rbr = d.build_player_objective_round_by_round(sel_row["player_id"])
st.dataframe(
    rbr[["round", "opponent_id", "objective_ev", "production_ev", "ev_difference",
         "objective_pred_votes", "production_pred_votes"]].rename(columns={
        "round": "Round", "opponent_id": "Opponent",
        "objective_ev": "Objective EV", "production_ev": "Production EV",
        "ev_difference": "Difference",
        "objective_pred_votes": "Objective 3/2/1", "production_pred_votes": "Production 3/2/1",
    }),
    use_container_width=True, hide_index=True,
)

st.divider()
st.header("5. Match View")
match_scores = d.load_objective_match_scores()
player_matches = match_scores[match_scores["player_id"].astype(str) == str(sel_row["player_id"])]
if len(player_matches):
    match_label = st.selectbox(
        "Match", player_matches["match_id"].tolist(),
        format_func=lambda mid: f"Round {player_matches.loc[player_matches.match_id==mid,'round'].iloc[0]} "
                                 f"vs {player_matches.loc[player_matches.match_id==mid,'opponent_id'].iloc[0]}",
    )
    match_rows = match_scores[match_scores["match_id"] == match_label].sort_values("p3", ascending=False)
    top3 = match_rows.head(3)
    c1, c2, c3 = st.columns(3)
    labels = ["Objective 3", "Objective 2", "Objective 1"]
    for col, (_, row), label in zip([c1, c2, c3], top3.iterrows(), labels):
        col.metric(label, row["player_name"], f"{row['expected_votes']:.2f} EV")
        col.caption(row["primary_drivers"] if pd.notna(row["primary_drivers"]) and row["primary_drivers"] else "No standout driver")

    st.subheader("Full match probability table")
    st.dataframe(
        match_rows[["player_name", "team_id", "objective_score", "p3", "p2", "p1", "p0",
                    "expected_votes", "primary_drivers"]].rename(columns={
            "player_name": "Player", "team_id": "Team", "objective_score": "Objective Score",
        }),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No scored matches found for this player.")
