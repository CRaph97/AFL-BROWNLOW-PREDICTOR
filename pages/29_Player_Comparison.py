"""
Player Comparison -- side-by-side cross-model comparison for 2-5 players.

Read-only over already-computed outputs. Reuses, rather than duplicates:
- dashboard.player_search.canonical_player_table()/season_snapshot() (the
  exact same season EV/rank/team-rank/P(any) assembly as pages/28_Player_Search.py)
- dashboard.finishing_order.topn_table() for real Winner/Top-N probabilities,
  computed directly from the Production/Objective Monte Carlo draw arrays --
  never approximated from EV or a static leaderboard rank
- dashboard.betting_opportunities.match_level_drilldown()/drilldown_summary()
  for each player's polling profile (the same drill-down as To Poll a Vote /
  Player Search)
- the same st.session_state["match_detail_preselect_match_id"] +
  st.switch_page("pages/6_Match_Detail.py") deep-link mechanism

No new modelling, no new feature engineering, no new scoring system. Never
blends Production/Objective/Wheelo into one number -- every section keeps
them visibly separate.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import betting_opportunities as bo
from dashboard import data as d
from dashboard import finishing_order as fo
from dashboard import player_search as ps

st.set_page_config(page_title="Player Comparison", layout="wide")
d.highlight_objective_stats_nav()
st.title("Player Comparison")
st.caption(
    "Compare 2-5 players side by side across Production, Objective, and Wheelo -- "
    "never blended into one forecast. Insights below describe numerical agreement/order "
    "only, never which player is \"better\"."
)

MIN_PLAYERS = 2
MAX_PLAYERS = 5

comparison = ps.canonical_player_table()
all_players = sorted(comparison["player_name"].dropna().unique())


def _first_unused(selected: list[str]) -> str:
    for name in all_players:
        if name not in selected:
            return name
    return all_players[0]


if "cmp_players" not in st.session_state:
    st.session_state.cmp_players = (
        comparison.sort_values("production_ev", ascending=False)["player_name"].head(MIN_PLAYERS).tolist()
    )

st.subheader("1. Players")
n = len(st.session_state.cmp_players)
cols = st.columns(n + 1)
updated = list(st.session_state.cmp_players)
for i in range(n):
    with cols[i]:
        current = st.session_state.cmp_players[i]
        others_selected = set(st.session_state.cmp_players) - {current}
        options = [p for p in all_players if p not in others_selected]
        idx = options.index(current) if current in options else 0
        chosen = st.selectbox(f"Player {i + 1}", options, index=idx, key=f"cmp_select_{i}")
        updated[i] = chosen
        if i >= MIN_PLAYERS:
            if st.button("Remove", key=f"cmp_remove_{i}"):
                updated.pop(i)
                st.session_state.cmp_players = updated
                st.rerun()
with cols[-1]:
    st.write("")
    st.write("")
    if len(st.session_state.cmp_players) < MAX_PLAYERS:
        if st.button("+ Add player"):
            updated.append(_first_unused(updated))
            st.session_state.cmp_players = updated
            st.rerun()
    else:
        st.caption(f"Maximum {MAX_PLAYERS} players.")
st.session_state.cmp_players = updated
selected_names = updated

id_by_name = dict(zip(comparison["player_name"], comparison["player_id"]))
selected_ids = [id_by_name[name] for name in selected_names]
snapshots = {name: ps.season_snapshot(pid) for name, pid in zip(selected_names, selected_ids)}

st.divider()


def _ev_agreement_band(diff: float | None) -> str:
    """Same EV-scale thresholds already documented and used on the Model
    Agreement page (section B, "Vote-Total Agreement") -- reused verbatim,
    not a new/invented scheme. fo._agreement_label() is deliberately NOT
    used here: it's calibrated for a probability-PERCENTAGE-POINT gap
    (e.g. Top-N% difference), not an absolute EV-vote gap, and applying it
    to the wrong kind of quantity would produce a plausible-looking but
    semantically wrong label."""
    if diff is None:
        return "Insufficient data"
    if diff <= 2.0:
        return "CLOSE"
    if diff <= 5.0:
        return "MODERATE"
    return "DIVERGENT"


# ==========================================================================
# 2. Season Comparison
# ==========================================================================
st.subheader("2. Season Comparison")
rows = []
for name, pid in zip(selected_names, selected_ids):
    snap = snapshots.get(name) or {}
    p, o, w = snap.get("production", {}), snap.get("objective", {}), snap.get("wheelo", {})
    gap = snap.get("model_gap")
    rows.append({
        "Player": name,
        "Team": d._display_team(snap.get("team_id")) if snap.get("team_id") else "N/A",
        "Prod EV": f"{p.get('ev'):.2f}" if p.get("ev") is not None else "N/A",
        "Prod rank": f"#{p['rank']}" if p.get("rank") else "N/A",
        "Prod team rank": f"#{p['team_rank']}" if p.get("team_rank") else "N/A",
        "Obj EV": f"{o.get('ev'):.2f}" if o.get("ev") is not None else "N/A",
        "Obj rank": f"#{o['rank']}" if o.get("rank") else "N/A",
        "Obj team rank": f"#{o['team_rank']}" if o.get("team_rank") else "N/A",
        "Wheelo EV": f"{w.get('ev'):.2f}" if w.get("ev") is not None else "N/A",
        "Wheelo rank": f"#{w['rank']}" if w.get("rank") else "N/A",
        "Wheelo team rank": f"#{w['team_rank']}" if w.get("team_rank") else "N/A",
        "Prod-Obj gap": f"{gap:.2f}" if gap is not None else "N/A",
        "Agreement": _ev_agreement_band(gap),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ==========================================================================
# 3. Finishing Probabilities
# ==========================================================================
st.subheader("3. Finishing Probabilities")
st.caption("From real Production/Objective season simulations -- never inferred from EV or a static rank.")
THRESHOLDS = [1, 3, 5, 10, 20]
THRESHOLD_LABELS = {1: "Winner %", 3: "Top 3 %", 5: "Top 5 %", 10: "Top 10 %", 20: "Top 20 %"}
tab_prod, tab_obj = st.tabs(["Production", "Objective"])
selected_ids_norm = [bo._normalise_player_id(pid) for pid in selected_ids]
for tab, col in [(tab_prod, "production_topn"), (tab_obj, "objective_topn")]:
    with tab:
        table_rows = {name: {} for name in selected_names}
        for n_thresh in THRESHOLDS:
            table = fo.topn_table(n_thresh)
            table["_pid"] = table["player_id"].apply(bo._normalise_player_id)
            for name, pid_norm in zip(selected_names, selected_ids_norm):
                match = table[table["_pid"] == pid_norm]
                val = float(match.iloc[0][col]) if not match.empty else None
                table_rows[name][THRESHOLD_LABELS[n_thresh]] = f"{val * 100:.1f}%" if val is not None else "N/A"
        display_df = pd.DataFrame(table_rows).T
        display_df.index.name = "Player"
        st.dataframe(display_df.reset_index(), use_container_width=True, hide_index=True)

st.divider()

# ==========================================================================
# 4. Polling Profile
# ==========================================================================
st.subheader("4. Polling Profile")
for name, pid in zip(selected_names, selected_ids):
    with st.expander(f"{name}", expanded=False):
        snap = snapshots.get(name) or {}
        p, o, w = snap.get("production", {}), snap.get("objective", {}), snap.get("wheelo", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Production season EV", f"{p.get('ev'):.2f}" if p.get("ev") is not None else "N/A")
        c2.metric("Objective season EV", f"{o.get('ev'):.2f}" if o.get("ev") is not None else "N/A")
        c3.metric("Wheelo season EV", f"{w.get('ev'):.2f}" if w.get("ev") is not None else "N/A")

        drilldown = bo.match_level_drilldown(pid)
        if drilldown.empty:
            st.caption("No match-level data resolved for this player.")
        else:
            summary = bo.drilldown_summary(drilldown)
            if "production" in summary:
                r, opp, pr = summary["production"]
                st.markdown(f"- **Strongest Production polling match:** Round {int(r)} vs {opp} (P(any vote) {pr * 100:.1f}%)")
            if "objective" in summary:
                r, opp, pr = summary["objective"]
                st.markdown(f"- **Strongest Objective polling match:** Round {int(r)} vs {opp} (P(any vote) {pr * 100:.1f}%)")
            if "wheelo" in summary:
                r, opp, pr = summary["wheelo"]
                st.markdown(f"- **Strongest Wheelo-supported match:** Round {int(r)} vs {opp} (P3-equivalent {pr:.1f}%)")

            top_row = drilldown.head(1).iloc[0]
            if st.button(f"Open Match Detail (R{top_row['round']} vs {top_row['opponent_display']})", key=f"cmp_open_match_{pid}"):
                st.session_state["match_detail_preselect_match_id"] = top_row["match_id"]
                st.switch_page("pages/6_Match_Detail.py")
        st.page_link("pages/28_Player_Search.py", label=f"Open Player Search for full {name} profile")

st.divider()

# ==========================================================================
# 5. Comparison Insights -- deterministic, from already-loaded numbers only.
# Never "better" -- numerical disagreement/order only.
# ==========================================================================
st.subheader("5. Comparison Insights")
insights: list[str] = []

gaps = {name: snapshots[name]["model_gap"] for name in selected_names if snapshots.get(name, {}).get("model_gap") is not None}
if gaps:
    closest = min(gaps, key=gaps.get)
    insights.append(f"Production and Objective agree most closely on **{closest}** (gap {gaps[closest]:.2f} votes).")
    widest = max(gaps, key=gaps.get)
    if widest != closest and len(gaps) > 1:
        insights.append(f"**{widest}** has the largest internal model disagreement (gap {gaps[widest]:.2f} votes).")

for i in range(len(selected_names)):
    for j in range(i + 1, len(selected_names)):
        a, b = selected_names[i], selected_names[j]
        pa, pb = snapshots[a]["production"].get("rank"), snapshots[b]["production"].get("rank")
        oa, ob = snapshots[a]["objective"].get("rank"), snapshots[b]["objective"].get("rank")
        if None not in (pa, pb, oa, ob) and len(insights) < 5:
            prod_a_above_b = pa < pb
            obj_a_above_b = oa < ob
            if prod_a_above_b != obj_a_above_b:
                higher, lower = (a, b) if prod_a_above_b else (b, a)
                insights.append(f"Production ranks **{higher}** above **{lower}**, while Objective reverses the order.")

for name in selected_names:
    if len(insights) >= 5:
        break
    snap = snapshots.get(name, {})
    wv, pv, ov = snap.get("wheelo", {}).get("ev"), snap.get("production", {}).get("ev"), snap.get("objective", {}).get("ev")
    if None not in (wv, pv, ov):
        closer = "Production" if abs(wv - pv) < abs(wv - ov) else "Objective"
        insights.append(f"Wheelo's EV aligns more closely with **{closer}** for **{name}**.")

if not insights:
    st.caption("Not enough overlapping data among selected players to generate insights.")
else:
    for bullet in insights[:5]:
        st.markdown(f"- {bullet}")

st.divider()

# ==========================================================================
# 6. Optional visual -- one compact grouped EV chart.
# ==========================================================================
st.subheader("6. EV Comparison Chart")
chart_rows = []
for name in selected_names:
    snap = snapshots.get(name, {})
    chart_rows.append({
        "Player": name,
        "Production": snap.get("production", {}).get("ev"),
        "Objective": snap.get("objective", {}).get("ev"),
        "Wheelo": snap.get("wheelo", {}).get("ev"),
    })
chart_df = pd.DataFrame(chart_rows).set_index("Player")
st.bar_chart(chart_df, height=320)
