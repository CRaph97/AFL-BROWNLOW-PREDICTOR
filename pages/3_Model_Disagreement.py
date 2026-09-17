import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Model Disagreement", layout="wide")
d.highlight_objective_stats_nav()
st.title("Model Disagreement")

st.markdown(
    "**Season-level disagreement** (from reports/2026_leaderboard.csv, the audited "
    "production output) vs. **round-level scenario spread** (derived here directly from "
    "data/processed/scenario_predictions_2026.parquet, the real match-level detail behind "
    "the season aggregate). The round-level table shows only the three genuine underlying "
    "scenarios (Historical / Recent-Era / Stats-Assisted) -- the final ensemble EV is a "
    "probability-space blend computed only at season level, so it is not fabricated here at "
    "round granularity."
)

lb = d.load_leaderboard()

st.subheader("Season-Level Disagreement (top 20 by disagreement)")
season_view = lb.sort_values("model_disagreement_range", ascending=False).head(20)
st.dataframe(
    season_view[["player_name", "team_id", "A_historical", "B_recent_era", "C_stats_assisted", "FINAL_ENSEMBLE", "model_disagreement_range", "structural_break_sensitivity"]]
    .rename(columns={
        "player_name": "Player", "team_id": "Team", "A_historical": "Historical EV",
        "B_recent_era": "Recent EV", "C_stats_assisted": "Stats-Assisted EV",
        "FINAL_ENSEMBLE": "Final EV", "model_disagreement_range": "Model Disagreement",
        "structural_break_sensitivity": "Structural-Break Sensitivity",
    })
    .style.format({c: "{:.2f}" for c in ["Historical EV", "Recent EV", "Stats-Assisted EV", "Final EV", "Model Disagreement", "Structural-Break Sensitivity"]}),
    use_container_width=True, hide_index=True, height=650,
)

st.divider()
st.subheader("Round-Level Scenario Spread")
n_players = st.slider("Players to include (by season-level Final EV rank)", 5, 40, 20)
pids = tuple(lb.sort_values("FINAL_ENSEMBLE", ascending=False).head(n_players)["player_id"].astype(int))
rd = d.round_level_disagreement_table(pids)

st.dataframe(
    rd[["player_name", "round", "A_historical", "B_recent_era", "C_stats_assisted", "round_disagreement"]]
    .rename(columns={
        "player_name": "Player", "round": "Round", "A_historical": "Historical EV",
        "B_recent_era": "Recent EV", "C_stats_assisted": "Stats-Assisted EV",
        "round_disagreement": "Round Disagreement",
    })
    .style.format({c: "{:.2f}" for c in ["Historical EV", "Recent EV", "Stats-Assisted EV", "Round Disagreement"]}),
    use_container_width=True, hide_index=True, height=600,
)
st.caption("Sorted by round-level disagreement, highest first -- these are the individual matches where the three scenarios disagree most, and therefore where a single-match prediction is least robust.")
