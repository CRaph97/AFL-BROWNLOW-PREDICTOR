import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="Leader After Round", layout="wide")
d.highlight_objective_stats_nav()
st.title("Leader After Round")
st.caption(
    "Cumulative totals using match-level data only. Round is this app's official normalized "
    "round (src/data/round_normalization.py) -- Wheelo's own Round column was independently "
    "verified to already use this same convention (see src/external/wheelo_source.py). ESPN's "
    "round columns are included where available; Betfair's round attribution was not reliable "
    "enough to use here (see src/external/betfair_source.py) and is excluded from this page."
)

prod = pd.read_csv("reports/2026_predicted_votes.csv")[["round", "player_id", "player_name", "team_id", "expected_votes"]]
prod["player_id"] = prod["player_id"].astype(str)
obj = pd.read_csv("reports/2026_objective_votes.csv")[["round", "player_id", "player_name", "team_id", "expected_votes"]]
obj["player_id"] = obj["player_id"].astype(str)
obj = obj[~obj["player_id"].str.startswith("NOID")]
wheelo = ed.load_wheelo_match_level()

min_round, max_round = int(prod["round"].min()), int(prod["round"].max())
round_n = st.slider("Leader after round", min_round, max_round, max_round)
top_n = st.radio("Show top", [5, 10, 20], index=1, horizontal=True)

def _cumulative(df, value_col, name_col, team_col):
    sub = df[df["round"] <= round_n]
    agg = sub.groupby("player_id", as_index=False).agg(
        cum=(value_col, "sum"), player_name=(name_col, "first"), team_id=(team_col, "first"),
    )
    agg["rank"] = agg["cum"].rank(ascending=False, method="min")
    return agg

prod_cum = _cumulative(prod, "expected_votes", "player_name", "team_id").rename(
    columns={"cum": "production_cum", "rank": "production_rank_at_round"}
)
obj_cum = _cumulative(obj, "expected_votes", "player_name", "team_id").rename(
    columns={"cum": "objective_cum", "rank": "objective_rank_at_round"}
)
wheelo_resolved = wheelo[wheelo["match_status"] == "resolved"]
wheelo_cum = _cumulative(wheelo_resolved, "wheelo_ev", "wheelo_player_name", "team_id").rename(
    columns={"cum": "wheelo_cum", "rank": "wheelo_rank_at_round", "player_name": "player_name"}
)

merged = prod_cum[["player_id", "player_name", "team_id", "production_cum", "production_rank_at_round"]].merge(
    obj_cum[["player_id", "objective_cum", "objective_rank_at_round"]], on="player_id", how="outer"
).merge(
    wheelo_cum[["player_id", "wheelo_cum", "wheelo_rank_at_round"]], on="player_id", how="outer"
)
merged["player_name"] = merged["player_name"].fillna(
    merged["player_id"].map(wheelo_cum.set_index("player_id")["player_name"])
)
merged["three_way_consensus_cum"] = merged[["production_cum", "objective_cum", "wheelo_cum"]].mean(axis=1, skipna=True)
merged["consensus_rank"] = merged["three_way_consensus_cum"].rank(ascending=False, method="min")
merged["max_rank_spread"] = merged[["production_rank_at_round", "objective_rank_at_round", "wheelo_rank_at_round"]].max(axis=1) - \
    merged[["production_rank_at_round", "objective_rank_at_round", "wheelo_rank_at_round"]].min(axis=1)

display = merged.sort_values("consensus_rank", na_position="last").head(top_n)
st.dataframe(display, use_container_width=True, hide_index=True, height=500)

st.divider()
st.subheader(f"Cumulative trajectory -- top {min(top_n, 10)} by consensus")
leaders = display.head(min(top_n, 10))["player_id"].tolist()
traj_rows = []
for pid in leaders:
    p_series = prod[prod["player_id"] == pid].groupby("round")["expected_votes"].sum().cumsum()
    name = merged.loc[merged["player_id"] == pid, "player_name"].iloc[0]
    for r, v in p_series.items():
        traj_rows.append({"round": r, "player": name, "cumulative_production_ev": v})
if traj_rows:
    traj_df = pd.DataFrame(traj_rows).pivot(index="round", columns="player", values="cumulative_production_ev")
    st.line_chart(traj_df)
