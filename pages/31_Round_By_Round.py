"""
Round-by-Round -- cross-model cumulative leaderboard (Production / Objective
/ Wheelo).

Read-only over already-computed match-level outputs. Reuses, rather than
duplicates, the exact cumulative-round aggregation pattern already built and
verified on pages/22_External_Leader_After_Round.py (per-model groupby-sum-
then-rank, on the app's own official normalized round -- src/data/
round_normalization.py -- not an assumed R0-R24 range). Extends it with an
EV-spread column and a real entering/leaving-top-10 comparison against the
previous round (both computed only from already-loaded cumulative data, no
new modelling).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="Round-by-Round", layout="wide")
d.highlight_objective_stats_nav()
st.title("Round-by-Round")
st.caption(
    "Cumulative totals using match-level data only, on this app's official normalized round. "
    "Production, Objective, and Wheelo kept visibly separate -- no blended consensus score."
)

prod = pd.read_csv("reports/2026_predicted_votes.csv")[["round", "player_id", "player_name", "team_id", "expected_votes"]]
prod["player_id"] = prod["player_id"].astype(str)
obj = pd.read_csv("reports/2026_objective_votes.csv")[["round", "player_id", "player_name", "team_id", "expected_votes"]]
obj["player_id"] = obj["player_id"].astype(str)
obj = obj[~obj["player_id"].str.startswith("NOID")]
wheelo = ed.load_wheelo_match_level()
wheelo_resolved = wheelo[wheelo["match_status"] == "resolved"] if not wheelo.empty else wheelo

min_round, max_round = int(prod["round"].min()), int(prod["round"].max())
round_n = st.slider("Round", min_round, max_round, max_round, format="R%d")


def _cumulative(df: pd.DataFrame, value_col: str, name_col: str, team_col: str, upto: int) -> pd.DataFrame:
    sub = df[df["round"] <= upto]
    if sub.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "team_id", "cum", "rank"])
    agg = sub.groupby("player_id", as_index=False).agg(
        cum=(value_col, "sum"), player_name=(name_col, "first"), team_id=(team_col, "first"),
    )
    agg["rank"] = agg["cum"].rank(ascending=False, method="min")
    return agg


def _build_cumulative_table(upto: int) -> pd.DataFrame:
    prod_cum = _cumulative(prod, "expected_votes", "player_name", "team_id", upto).rename(
        columns={"cum": "production_cum", "rank": "production_rank"}
    )
    obj_cum = _cumulative(obj, "expected_votes", "player_name", "team_id", upto).rename(
        columns={"cum": "objective_cum", "rank": "objective_rank"}
    )
    wheelo_cum = _cumulative(wheelo_resolved, "wheelo_ev", "wheelo_player_name", "team_id", upto).rename(
        columns={"cum": "wheelo_cum", "rank": "wheelo_rank"}
    ) if not wheelo_resolved.empty else pd.DataFrame(columns=["player_id", "player_name", "team_id", "wheelo_cum", "wheelo_rank"])

    merged = prod_cum[["player_id", "player_name", "team_id", "production_cum", "production_rank"]].merge(
        obj_cum[["player_id", "objective_cum", "objective_rank"]], on="player_id", how="outer",
    ).merge(
        wheelo_cum[["player_id", "wheelo_cum", "wheelo_rank"]], on="player_id", how="outer",
    )
    if not wheelo_cum.empty:
        merged["player_name"] = merged["player_name"].fillna(
            merged["player_id"].map(wheelo_cum.set_index("player_id")["player_name"])
        )
    rank_cols = ["production_rank", "objective_rank", "wheelo_rank"]
    ev_cols = ["production_cum", "objective_cum", "wheelo_cum"]
    merged["rank_spread"] = merged[rank_cols].max(axis=1, skipna=True) - merged[rank_cols].min(axis=1, skipna=True)
    merged["ev_spread"] = merged[ev_cols].max(axis=1, skipna=True) - merged[ev_cols].min(axis=1, skipna=True)
    merged["consensus_ev"] = merged[ev_cols].mean(axis=1, skipna=True)
    merged["overall_rank"] = merged["consensus_ev"].rank(ascending=False, method="min")
    return merged.sort_values("overall_rank", na_position="last").reset_index(drop=True)


current = _build_cumulative_table(round_n)

st.subheader(f"Cumulative Leaderboard through R{round_n}")
display = current.head(20).copy()
table = pd.DataFrame({
    "Rank": display["overall_rank"].map(lambda v: f"#{int(v)}" if pd.notna(v) else "N/A"),
    "Player": display["player_name"],
    "Prod cum EV": display["production_cum"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
    "Prod rank": display["production_rank"].map(lambda v: f"#{int(v)}" if pd.notna(v) else "N/A"),
    "Obj cum EV": display["objective_cum"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
    "Obj rank": display["objective_rank"].map(lambda v: f"#{int(v)}" if pd.notna(v) else "N/A"),
    "Wheelo cum EV": display["wheelo_cum"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
    "Wheelo rank": display["wheelo_rank"].map(lambda v: f"#{int(v)}" if pd.notna(v) else "N/A"),
    "Rank spread": display["rank_spread"].map(lambda v: f"{int(v)}" if pd.notna(v) else "N/A"),
    "EV spread": display["ev_spread"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
})
st.dataframe(table, use_container_width=True, hide_index=True, height=560)

st.divider()

# --------------------------------------------------------------------------
# Deterministic summary -- numbers only.
# --------------------------------------------------------------------------
st.subheader("Summary")
bullets = []
top1 = current.iloc[0] if not current.empty else None
if top1 is not None:
    leaders = {
        top1.get("player_name") for col in ["production_rank", "objective_rank", "wheelo_rank"]
        if pd.notna(top1.get(col)) and top1.get(col) == 1
    }
    prod_top = current.loc[current["production_rank"] == 1, "player_name"] if current["production_rank"].notna().any() else pd.Series(dtype=object)
    obj_top = current.loc[current["objective_rank"] == 1, "player_name"] if current["objective_rank"].notna().any() else pd.Series(dtype=object)
    wheelo_top = current.loc[current["wheelo_rank"] == 1, "player_name"] if current["wheelo_rank"].notna().any() else pd.Series(dtype=object)
    names = {n.iloc[0] for n in (prod_top, obj_top, wheelo_top) if not n.empty}
    if len(names) == 1:
        bullets.append(f"All three sources agree the leader through R{round_n} is **{names.pop()}**.")
    elif names:
        bullets.append(
            f"Sources disagree on the R{round_n} leader: Production={prod_top.iloc[0] if not prod_top.empty else 'N/A'}, "
            f"Objective={obj_top.iloc[0] if not obj_top.empty else 'N/A'}, "
            f"Wheelo={wheelo_top.iloc[0] if not wheelo_top.empty else 'N/A'}."
        )

if current["rank_spread"].notna().any():
    biggest_rank = current.loc[current["rank_spread"].idxmax()]
    bullets.append(
        f"**{biggest_rank['player_name']}** has the biggest cross-model rank disagreement through R{round_n} "
        f"(spread of {int(biggest_rank['rank_spread'])} places)."
    )
if current["ev_spread"].notna().any():
    biggest_ev = current.loc[current["ev_spread"].idxmax()]
    bullets.append(
        f"**{biggest_ev['player_name']}** has the biggest cumulative EV disagreement through R{round_n} "
        f"({biggest_ev['ev_spread']:.2f} votes spread)."
    )

if round_n > min_round:
    previous = _build_cumulative_table(round_n - 1)
    current_top10 = set(current[current["overall_rank"] <= 10]["player_id"])
    previous_top10 = set(previous[previous["overall_rank"] <= 10]["player_id"]) if not previous.empty else set()
    entered = current_top10 - previous_top10
    left = previous_top10 - current_top10
    name_lookup = dict(zip(current["player_id"], current["player_name"]))
    prev_name_lookup = dict(zip(previous["player_id"], previous["player_name"])) if not previous.empty else {}
    if entered:
        names_in = ", ".join(sorted(name_lookup.get(pid, pid) for pid in entered))
        bullets.append(f"Entered the consensus top 10 this round: {names_in}.")
    if left:
        names_out = ", ".join(sorted(prev_name_lookup.get(pid, pid) for pid in left))
        bullets.append(f"Left the consensus top 10 this round: {names_out}.")

for b in bullets[:5]:
    st.markdown(f"- {b}")
if not bullets:
    st.caption("No summary bullets available for this round.")

st.divider()

# --------------------------------------------------------------------------
# Trajectory chart -- 1-5 players, models kept visually separate.
# --------------------------------------------------------------------------
st.subheader("Trajectory")
name_options = sorted(current["player_name"].dropna().unique())
default_players = current.head(3)["player_name"].tolist()
chosen = st.multiselect("Players (max 5)", name_options, default=default_players[:3], max_selections=5)

if chosen:
    id_lookup = dict(zip(current["player_name"], current["player_id"]))
    chart_cols = {}
    for name in chosen:
        pid = id_lookup.get(name)
        p_series = prod[prod["player_id"] == pid].groupby("round")["expected_votes"].sum().reindex(range(min_round, round_n + 1), fill_value=0).cumsum()
        o_series = obj[obj["player_id"] == pid].groupby("round")["expected_votes"].sum().reindex(range(min_round, round_n + 1), fill_value=0).cumsum()
        chart_cols[f"{name} (Prod)"] = p_series
        chart_cols[f"{name} (Obj)"] = o_series
        if not wheelo_resolved.empty:
            w_sub = wheelo_resolved[wheelo_resolved["player_id"] == pid]
            if not w_sub.empty:
                w_series = w_sub.groupby("round")["wheelo_ev"].sum().reindex(range(min_round, round_n + 1), fill_value=0).cumsum()
                chart_cols[f"{name} (Wheelo)"] = w_series
    chart_df = pd.DataFrame(chart_cols)
    chart_df.index.name = "Round"
    st.line_chart(chart_df)
else:
    st.caption("Select at least one player to see their cumulative trajectory.")
