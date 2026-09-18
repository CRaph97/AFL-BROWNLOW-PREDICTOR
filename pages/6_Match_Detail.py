import pandas as pd
import streamlit as st

from dashboard import betting_opportunities as bo
from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="Match Detail", layout="wide")
d.highlight_objective_stats_nav()
st.title("Match Detail")

meta = d.match_meta_table()
pv = d.load_predicted_votes()
mp = d.load_match_probabilities()
core = d.load_core_2026()
watchlist = d.load_defender_watchlist()
obj_votes = d.load_objective_votes()
wheelo_ml = ed.load_wheelo_match_level()

meta_sorted = meta.sort_values(["round", "match_id"])
options = meta_sorted.apply(lambda r: f"R{r['round']}: {r['result_label']}", axis=1).tolist()
choice = st.selectbox("Select match", options, index=0)
m = meta_sorted.iloc[options.index(choice)]
match_id = m["match_id"]

st.subheader(m["result_label"])
st.caption(f"Round {m['round']} -- {m['venue']} -- {m['date']}")

# Defender-bias warning banner: this match's round + either team appears in the
# audited watchlist (round/player_name/team_id/opponent_id from
# docs/2026_FINAL_AUDIT.md) -- flag only, never adjust the prediction.
flagged = watchlist[
    (watchlist["round"] == int(m["round"]))
    & (watchlist["team_id"].isin([m["team_a"], m["team_b"]]))
]
if not flagged.empty:
    for _, f in flagged.iterrows():
        st.warning(
            f"**Known model weakness: elite defender performance may be underweighted.** "
            f"{f['player_name']} ({f['team_id'].replace('_', ' ').title()}) recorded "
            f"{f['disposals']:g} disposals / {f['contested_marks']:g} contested marks / "
            f"{f['one_percenters']:g} one-percenters this round but the model gives an "
            f"expected-votes estimate of only {f['expected_votes']:.2f}. Not manually adjusted."
        )

st.divider()
st.subheader("Predicted 3-2-1")
match_pv = pv[pv["match_id"] == match_id].sort_values("predicted_votes", ascending=False)
cols = st.columns(3)
for i, (_, r) in enumerate(match_pv.iterrows()):
    with cols[i]:
        st.metric(f"{int(r['predicted_votes'])} votes", r["player_name"], r["team_id"].replace("_", " ").title())

st.divider()
st.subheader("Probability Distribution -- Leading Players (Production)")
match_mp = mp[mp["match_id"] == match_id].sort_values("expected_votes", ascending=False).head(10)
st.bar_chart(match_mp.set_index("player_name")[["p3", "p2", "p1"]], height=320)
st.dataframe(
    match_mp[["player_name", "team_id", "p3", "p2", "p1", "p0", "expected_votes"]]
    .rename(columns={"player_name": "Player", "team_id": "Team", "p3": "P(3)", "p2": "P(2)", "p1": "P(1)", "p0": "P(0)", "expected_votes": "Expected Votes"})
    .style.format({"P(3)": "{:.0%}", "P(2)": "{:.0%}", "P(1)": "{:.0%}", "P(0)": "{:.0%}", "Expected Votes": "{:.2f}"}),
    use_container_width=True, hide_index=True,
)

st.divider()
st.subheader("Production vs. Objective vs. Wheelo -- This Match")
st.caption(
    "All three models kept visibly separate -- never blended into one score. "
    "Wheelo has no P2/P1 data, so only its match Votes/EV and P(3) are shown, never a fabricated P(any)."
)
_prod = mp[mp["match_id"] == match_id][["player_id", "player_name", "team_id", "p3", "p2", "p1", "expected_votes"]].copy()
_prod["pid_norm"] = _prod["player_id"].apply(bo._normalise_player_id)
_prod = _prod.rename(columns={"p3": "production_p3", "p2": "production_p2", "p1": "production_p1", "expected_votes": "production_ev"})

_obj = obj_votes[obj_votes["match_id"] == match_id][["player_id", "p3", "p2", "p1", "expected_votes"]].copy()
_obj["pid_norm"] = _obj["player_id"].apply(bo._normalise_player_id)
_obj = _obj.rename(columns={"p3": "objective_p3", "p2": "objective_p2", "p1": "objective_p1", "expected_votes": "objective_ev"})

_wheelo = pd.DataFrame(columns=["pid_norm", "wheelo_match_ev", "wheelo_p3_pct"])
if not wheelo_ml.empty:
    _wheelo = wheelo_ml[(wheelo_ml["round"] == int(m["round"])) & (wheelo_ml["match_status"] == "resolved")][
        ["player_id", "wheelo_ev", "wheelo_p3_pct"]
    ].copy()
    _wheelo["pid_norm"] = _wheelo["player_id"].apply(bo._normalise_player_id)
    _wheelo = _wheelo.rename(columns={"wheelo_ev": "wheelo_match_ev"})

combined = _prod[["pid_norm", "player_id", "player_name", "team_id", "production_p3", "production_p2", "production_p1", "production_ev"]].merge(
    _obj[["pid_norm", "objective_p3", "objective_p2", "objective_p1", "objective_ev"]], on="pid_norm", how="outer",
).merge(
    _wheelo[["pid_norm", "wheelo_match_ev", "wheelo_p3_pct"]], on="pid_norm", how="left",
)
# Sort by COMBINED evidence (Production EV + Objective EV), not either model's
# own value alone -- sorting by one model's raw number repeats a known
# pitfall (see dashboard/finishing_order.py's identical rationale): it can
# bury a player one model rates highly but the other doesn't, purely because
# of tie/ordering artefacts in the dominant model.
combined["combined_ev"] = combined["production_ev"].fillna(0) + combined["objective_ev"].fillna(0)
combined = combined.sort_values("combined_ev", ascending=False).head(10)

match_table = pd.DataFrame({
    "Player": combined["player_name"],
    "Team": combined["team_id"].apply(d._display_team),
    "Prod EV": combined["production_ev"],
    "Prod P3": combined["production_p3"],
    "Prod P2": combined["production_p2"],
    "Prod P1": combined["production_p1"],
    "Obj EV": combined["objective_ev"],
    "Obj P3": combined["objective_p3"],
    "Obj P2": combined["objective_p2"],
    "Obj P1": combined["objective_p1"],
    "Wheelo Votes": combined["wheelo_match_ev"],
    "Wheelo P3%": combined["wheelo_p3_pct"],
})
st.dataframe(
    match_table.style.format({
        "Prod EV": "{:.2f}", "Prod P3": "{:.0%}", "Prod P2": "{:.0%}", "Prod P1": "{:.0%}",
        "Obj EV": "{:.2f}", "Obj P3": "{:.0%}", "Obj P2": "{:.0%}", "Obj P1": "{:.0%}",
        "Wheelo Votes": "{:.2f}", "Wheelo P3%": "{:.1f}%",
    }, na_rep="N/A"),
    use_container_width=True, hide_index=True,
)

st.divider()
st.subheader("Key Model Drivers, Relative Ranks & Teammate Competition")
leading_ids = match_mp["player_id"].astype("Int64").tolist()
match_core = core[core["match_id"] == match_id]
for pid in leading_ids[:5]:
    prow = match_core[match_core["player_id"] == pid]
    if prow.empty:
        continue
    prow = prow.iloc[0]
    with st.expander(f"{prow['player_name']} -- drivers & context"):
        st.write("**Key drivers:**", d.primary_drivers_string(prow))
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Relative stat ranks (within team):**")
            for stat in ("disposals", "contested_possessions", "clearances", "tackles", "goals"):
                rank = prow.get(f"{stat}_team_rank")
                share = prow.get(f"{stat}_team_share")
                if rank is not None:
                    st.caption(f"{stat}: team rank {int(rank) if rank == rank else 'n/a'}, share {share:.0%}" if share == share else f"{stat}: n/a")
        with c2:
            st.write("**Teammate competition indicators:**")
            st.caption(f"Teammates with 25+ disposals: {prow.get('n_teammates_disposals_ge_25')}")
            st.caption(f"Teammates with 30+ disposals: {prow.get('n_teammates_disposals_ge_30')}")
            st.caption(f"Teammates with 2+ goals: {prow.get('n_teammates_goals_ge_2')}")
            st.caption(f"Team disposal concentration: {prow.get('team_disposal_concentration'):.3f}" if prow.get("team_disposal_concentration") == prow.get("team_disposal_concentration") else "n/a")

st.divider()
st.subheader("Scenario Disagreement (this match)")
sp = d.load_scenario_predictions_2026()
match_sp = sp[(sp["match_id"] == match_id) & (sp["player_id"].isin(leading_ids))]
pivot = match_sp.pivot_table(index="player_name", columns="scenario", values="expected_votes")
scen_cols = [c for c in ["A_historical", "B_recent_era", "C_stats_assisted"] if c in pivot.columns]
if not pivot.empty:
    pivot["disagreement"] = pivot[scen_cols].max(axis=1) - pivot[scen_cols].min(axis=1)
    st.dataframe(pivot[scen_cols + ["disagreement"]].style.format("{:.2f}"), use_container_width=True)
else:
    st.caption("No scenario-level detail available for this match's leading players.")

st.divider()
st.subheader("Winner / Margin Context")
st.write(f"**{m['team_a'].replace('_',' ').title()}** {m['team_a_score']:g} -- **{m['team_b'].replace('_',' ').title()}** {m['team_b_score']:g}  (margin {m['margin']:g})")
