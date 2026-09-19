"""
Teams -- cross-model team view (Production / Objective / Wheelo).

Read-only over already-computed outputs. Reuses, rather than duplicates:
- dashboard.data.dual_model_team_rankings() (the exact same team-relative
  Production/Objective EV+rank join already used by the Betting
  Opportunities page's own Team Explorer section)
- dashboard.external_data.load_external_overview() for Wheelo team EV
- the same EV-scale agreement-band thresholds already documented and used
  on pages/29_Player_Comparison.py (duplicated as a tiny pure function per
  this project's established convention for small, cross-module-dependency-
  avoiding helpers -- see dashboard.betting_opportunities._normalise_player_id's
  own docstring for the precedent)
- reports/2026_match_probabilities.csv / 2026_objective_votes.csv for the
  optional round-contribution breakdown

Never blends Production/Objective/Wheelo into one score.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="Teams", layout="wide")
d.highlight_objective_stats_nav()
st.title("Teams")
st.caption(
    "Cross-model team view -- Production, Objective, and Wheelo shown side by side, never "
    "blended into one score. Interpretation bullets describe numerical agreement/order only."
)

teams = d.team_list()
team_choice = st.selectbox("Select team", teams, format_func=d._display_team)

rankings = d.dual_model_team_rankings(team_choice)
external_overview = ed.load_external_overview()
wheelo_team = external_overview[external_overview["team_id"] == team_choice] if not external_overview.empty else pd.DataFrame()
if not wheelo_team.empty:
    wheelo_team = wheelo_team.copy()
    # external_overview's player_id is int64; dual_model_team_rankings'
    # (via load_dual_model_comparison) normalises to str -- match that
    # convention here rather than assume dtypes agree (they don't).
    wheelo_team["player_id"] = wheelo_team["player_id"].astype(str)
    wheelo_team["wheelo_team_rank"] = wheelo_team["wheelo_ev"].rank(ascending=False, method="min")

team_prod_total = rankings["production_ev"].sum()
team_obj_total = rankings["objective_ev"].sum()
team_wheelo_total = wheelo_team["wheelo_ev"].sum() if not wheelo_team.empty else float("nan")
totals = [t for t in (team_prod_total, team_obj_total, team_wheelo_total) if pd.notna(t)]
model_spread = (max(totals) - min(totals)) if len(totals) >= 2 else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Production team total EV", f"{team_prod_total:.1f}")
c2.metric("Objective team total EV", f"{team_obj_total:.1f}")
c3.metric("Wheelo team total EV", f"{team_wheelo_total:.1f}" if pd.notna(team_wheelo_total) else "N/A")
c4.metric("Model spread (max-min total)", f"{model_spread:.1f}" if pd.notna(model_spread) else "N/A")

st.divider()


def _ev_agreement_band(diff: float | None) -> str:
    """Same EV-scale thresholds already documented and used on
    pages/29_Player_Comparison.py -- reused verbatim, not a new/invented
    scheme. Not fo._agreement_label(), which is calibrated for a
    probability-PERCENTAGE-POINT gap, not an absolute EV-vote gap."""
    if diff is None or pd.isna(diff):
        return "Insufficient data"
    if diff <= 2.0:
        return "CLOSE"
    if diff <= 5.0:
        return "MODERATE"
    return "DIVERGENT"


st.subheader("Player Table")
merged = rankings.merge(
    wheelo_team[["player_id", "wheelo_ev", "wheelo_rank", "wheelo_team_rank"]] if not wheelo_team.empty
    else pd.DataFrame(columns=["player_id", "wheelo_ev", "wheelo_rank", "wheelo_team_rank"]),
    on="player_id", how="left",
)
merged["ev_gap"] = (merged["production_ev"] - merged["objective_ev"]).abs()
merged["agreement"] = merged["ev_gap"].apply(_ev_agreement_band)
merged = merged.sort_values("production_ev", ascending=False, na_position="last")

table = pd.DataFrame({
    "Player": merged["player_name"],
    "Prod EV": merged["production_ev"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
    "Prod team rank": merged["production_team_rank"].map(lambda v: f"#{int(v)}" if pd.notna(v) else "N/A"),
    "Obj EV": merged["objective_ev"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
    "Obj team rank": merged["objective_team_rank"].map(lambda v: f"#{int(v)}" if pd.notna(v) else "N/A"),
    "Wheelo EV": merged["wheelo_ev"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
    "Wheelo team rank": merged["wheelo_team_rank"].map(lambda v: f"#{int(v)}" if pd.notna(v) else "N/A"),
    "Prod-Obj gap": merged["ev_gap"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"),
    "Agreement": merged["agreement"],
})
st.dataframe(table, use_container_width=True, hide_index=True, height=460)

st.divider()

# --------------------------------------------------------------------------
# Deterministic interpretation bullets -- numbers only, never a "better"
# player claim.
# --------------------------------------------------------------------------
st.subheader("Interpretation")
bullets = []
prod_leader = merged.loc[merged["production_ev"].idxmax(), "player_name"] if merged["production_ev"].notna().any() else None
obj_leader = merged.loc[merged["objective_ev"].idxmax(), "player_name"] if merged["objective_ev"].notna().any() else None
wheelo_leader = merged.loc[merged["wheelo_ev"].idxmax(), "player_name"] if merged["wheelo_ev"].notna().any() else None
if prod_leader and prod_leader == obj_leader and prod_leader == wheelo_leader:
    bullets.append(f"All three sources agree **{prod_leader}** leads this team.")
elif prod_leader and prod_leader == obj_leader:
    bullets.append(f"Production and Objective both lead with **{prod_leader}**" + (
        f", while Wheelo's top EV is **{wheelo_leader}**." if wheelo_leader and wheelo_leader != prod_leader else "."
    ))
elif prod_leader and obj_leader:
    bullets.append(f"Production's team leader is **{prod_leader}**, Objective's is **{obj_leader}** -- they disagree on the top player.")

if merged["ev_gap"].notna().any():
    widest = merged.loc[merged["ev_gap"].idxmax()]
    bullets.append(
        f"**{widest['player_name']}** has the largest Production-vs-Objective gap on this team "
        f"({widest['production_ev']:.2f} vs {widest['objective_ev']:.2f}, {widest['ev_gap']:.2f} votes)."
    )

net_diff = merged["objective_ev"].sum(skipna=True) - merged["production_ev"].sum(skipna=True)
if abs(net_diff) > 1.0:
    direction = "higher" if net_diff > 0 else "lower"
    bullets.append(f"Objective rates this team's total output **{abs(net_diff):.1f} votes {direction}** than Production overall.")

if not wheelo_team.empty and merged["wheelo_ev"].notna().any():
    merged["_wheelo_prod_gap"] = (merged["wheelo_ev"] - merged["production_ev"]).abs()
    merged["_wheelo_obj_gap"] = (merged["wheelo_ev"] - merged["objective_ev"]).abs()
    closer_to_prod = (merged["_wheelo_prod_gap"] < merged["_wheelo_obj_gap"]).sum()
    closer_to_obj = (merged["_wheelo_obj_gap"] < merged["_wheelo_prod_gap"]).sum()
    if closer_to_prod != closer_to_obj:
        which = "Production" if closer_to_prod > closer_to_obj else "Objective"
        bullets.append(f"Wheelo's EVs align more closely with **{which}** for most players on this team.")

for b in bullets[:5]:
    st.markdown(f"- {b}")
if not bullets:
    st.caption("No interpretation bullets available -- insufficient overlapping data for this team.")

st.divider()

# --------------------------------------------------------------------------
# Optional round-contribution breakdown -- existing match-level data only.
# --------------------------------------------------------------------------
with st.expander("Round contribution (existing match-level data)", expanded=False):
    prod_mp = d.load_match_probabilities()
    prod_team = prod_mp[prod_mp["team_id"] == team_choice][["round", "player_name", "expected_votes"]]
    obj_votes = d.load_objective_votes()
    obj_team = obj_votes[obj_votes["team_id"] == team_choice][["round", "player_name", "expected_votes"]]
    if prod_team.empty and obj_team.empty:
        st.caption("No match-level rows resolved for this team.")
    else:
        prod_round = prod_team.groupby("round", as_index=False)["expected_votes"].sum().rename(columns={"expected_votes": "Production round EV"})
        obj_round = obj_team.groupby("round", as_index=False)["expected_votes"].sum().rename(columns={"expected_votes": "Objective round EV"})
        round_table = prod_round.merge(obj_round, on="round", how="outer").sort_values("round")
        st.dataframe(round_table, use_container_width=True, hide_index=True, height=320)
