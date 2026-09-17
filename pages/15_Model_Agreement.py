"""
Model Agreement -- surfaces conclusions independently reached by BOTH the
Production and Objective models. Read-only aggregation over each model's
already-produced output files; never fits, blends, or creates a new
combined model. Agreement is a robustness signal, not proof of correctness:
both models can share the same blind spot (e.g. defenders), so this page
should be read alongside the Guide & FAQs and Uncertainty pages, not as a
final word.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import betting_data as bd

st.set_page_config(page_title="Model Agreement", layout="wide")
d.highlight_objective_stats_nav()
st.title("Model Agreement")
st.warning(
    "**Agreement between the two models does not imply either is correct.** Both are built from "
    "overlapping 2026 match statistics and share some of the same limitations (e.g. under-predicting "
    "elite defenders) -- shared agreement can reflect a shared blind spot, not just genuine signal."
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


@st.cache_data
def load_all():
    prod_lb = pd.read_csv(REPORTS_DIR / "2026_leaderboard.csv")
    obj_lb = pd.read_csv(REPORTS_DIR / "2026_objective_leaderboard.csv")
    prod_votes = pd.read_csv(REPORTS_DIR / "2026_predicted_votes.csv")
    obj_votes = pd.read_csv(REPORTS_DIR / "2026_objective_votes.csv")
    order_scenarios = pd.read_csv(REPORTS_DIR / "2026_order_scenarios.csv")
    return prod_lb, obj_lb, prod_votes, obj_votes, order_scenarios


prod_lb, obj_lb, prod_votes, obj_votes, order_scenarios = load_all()
prod_lb = prod_lb.copy()
obj_lb = obj_lb.copy()
prod_lb["player_id"] = prod_lb["player_id"].astype(str)
obj_lb["player_id"] = obj_lb["player_id"].astype(str)

merged = prod_lb[["player_id", "player_name", "team_id", "rank", "FINAL_ENSEMBLE"]].rename(
    columns={"rank": "Production Rank", "FINAL_ENSEMBLE": "Production EV"}
).merge(
    obj_lb[["player_id", "rank", "objective_ev"]].rename(
        columns={"rank": "Objective Rank", "objective_ev": "Objective EV"}),
    on="player_id", how="inner",
)
merged["Rank Difference"] = (merged["Production Rank"] - merged["Objective Rank"]).abs()
merged["EV Difference"] = (merged["Production EV"] - merged["Objective EV"]).abs()

st.divider()

# --------------------------------------------------------------------------
# A. Leaderboard agreement
# --------------------------------------------------------------------------
st.header("A. Leaderboard Agreement")
st.caption("Players both models place in the same broad tier, and how close their exact ranks are.")

a1, a2, a3 = st.columns(3)
both_top3 = merged[(merged["Production Rank"] <= 3) & (merged["Objective Rank"] <= 3)]
both_top5 = merged[(merged["Production Rank"] <= 5) & (merged["Objective Rank"] <= 5)]
both_top10 = merged[(merged["Production Rank"] <= 10) & (merged["Objective Rank"] <= 10)]
a1.metric("Both models' Top 3", len(both_top3))
a2.metric("Both models' Top 5", len(both_top5))
a3.metric("Both models' Top 10", len(both_top10))

st.markdown("**Strongest consensus** (smallest rank difference, restricted to each model's own Top 20):")
consensus = merged[(merged["Production Rank"] <= 20) & (merged["Objective Rank"] <= 20)] \
    .sort_values("Rank Difference").head(10)
st.dataframe(
    consensus[["player_name", "team_id", "Production Rank", "Objective Rank", "Rank Difference",
               "Production EV", "Objective EV"]].rename(columns={"player_name": "Player", "team_id": "Team"}),
    use_container_width=True, hide_index=True,
)

st.divider()

# --------------------------------------------------------------------------
# B. Vote-total agreement
# --------------------------------------------------------------------------
st.header("B. Vote-Total Agreement")
st.caption(
    "Transparent thresholds, not a blended score: CLOSE = EV difference <= 2.0 votes; "
    "MODERATE = 2.0-5.0; DIVERGENT = > 5.0. Restricted to each model's own Top 30 for relevance."
)

pool = merged[(merged["Production Rank"] <= 30) | (merged["Objective Rank"] <= 30)].copy()


def _band(diff: float) -> str:
    if diff <= 2.0:
        return "CLOSE"
    if diff <= 5.0:
        return "MODERATE"
    return "DIVERGENT"


pool["Agreement Band"] = pool["EV Difference"].apply(_band)
b1, b2, b3 = st.columns(3)
b1.metric("CLOSE (<=2.0 votes)", (pool["Agreement Band"] == "CLOSE").sum())
b2.metric("MODERATE (2-5 votes)", (pool["Agreement Band"] == "MODERATE").sum())
b3.metric("DIVERGENT (>5 votes)", (pool["Agreement Band"] == "DIVERGENT").sum())

st.dataframe(
    pool.sort_values("EV Difference")[["player_name", "team_id", "Production EV", "Objective EV",
                                         "EV Difference", "Agreement Band"]]
    .rename(columns={"player_name": "Player", "team_id": "Team"}),
    use_container_width=True, hide_index=True, height=350,
)

st.divider()

# --------------------------------------------------------------------------
# C. Round / match agreement
# --------------------------------------------------------------------------
st.header("C. Round / Match Agreement")
st.caption("Matches where both models' deterministic 3-2-1 picks agree, fully or partially.")

pv3 = prod_votes[prod_votes["predicted_votes"] == 3][["match_id", "round", "player_name"]].rename(columns={"player_name": "prod_3"})
pv2 = prod_votes[prod_votes["predicted_votes"] == 2][["match_id", "player_name"]].rename(columns={"player_name": "prod_2"})
pv1 = prod_votes[prod_votes["predicted_votes"] == 1][["match_id", "player_name"]].rename(columns={"player_name": "prod_1"})
ov3 = obj_votes[obj_votes["objective_pred_votes"] == 3][["match_id", "player_name"]].rename(columns={"player_name": "obj_3"})
ov2 = obj_votes[obj_votes["objective_pred_votes"] == 2][["match_id", "player_name"]].rename(columns={"player_name": "obj_2"})
ov1 = obj_votes[obj_votes["objective_pred_votes"] == 1][["match_id", "player_name"]].rename(columns={"player_name": "obj_1"})

match_cmp = pv3.merge(pv2, on="match_id").merge(pv1, on="match_id") \
    .merge(ov3, on="match_id").merge(ov2, on="match_id").merge(ov1, on="match_id")


def _agreement_type(r) -> str:
    if r["prod_3"] == r["obj_3"] and r["prod_2"] == r["obj_2"] and r["prod_1"] == r["obj_1"]:
        return "EXACT 3-2-1 MATCH"
    if r["prod_3"] == r["obj_3"] and r["prod_2"] == r["obj_2"]:
        return "3 AND 2 VOTE MATCH"
    if r["prod_3"] == r["obj_3"]:
        return "3-VOTE MATCH"
    return "NO MATCH"


match_cmp["Agreement Type"] = match_cmp.apply(_agreement_type, axis=1)
match_cmp["Production 3-2-1"] = match_cmp["prod_3"] + " / " + match_cmp["prod_2"] + " / " + match_cmp["prod_1"]
match_cmp["Objective 3-2-1"] = match_cmp["obj_3"] + " / " + match_cmp["obj_2"] + " / " + match_cmp["obj_1"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Exact 3-2-1 match", (match_cmp["Agreement Type"] == "EXACT 3-2-1 MATCH").sum())
c2.metric("3 and 2 vote match", (match_cmp["Agreement Type"] == "3 AND 2 VOTE MATCH").sum())
c3.metric("3-vote match only", (match_cmp["Agreement Type"] == "3-VOTE MATCH").sum())
c4.metric("No match", (match_cmp["Agreement Type"] == "NO MATCH").sum())

agree_filter = st.selectbox("Show", ["EXACT 3-2-1 MATCH", "3 AND 2 VOTE MATCH", "3-VOTE MATCH", "All matches"])
shown = match_cmp if agree_filter == "All matches" else match_cmp[match_cmp["Agreement Type"] == agree_filter]
st.dataframe(
    shown[["round", "match_id", "Production 3-2-1", "Objective 3-2-1", "Agreement Type"]]
    .rename(columns={"round": "Round", "match_id": "Match"}),
    use_container_width=True, hide_index=True, height=350,
)

st.divider()

# --------------------------------------------------------------------------
# D. Order-scenario agreement
# --------------------------------------------------------------------------
st.header("D. Order-Scenario Agreement")
st.caption("Players appearing in the same finishing position across both models' most probable exact orders.")

depth = st.radio("Depth", [5, 7, 10], format_func=lambda n: f"Top {n}", horizontal=True, key="agreement_depth")
pos_cols = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"][:depth]

prod_top1 = order_scenarios[(order_scenarios.model == "Production") & (order_scenarios.order_depth == depth)
                             & (order_scenarios.scenario_rank == 1)].iloc[0]
obj_top1 = order_scenarios[(order_scenarios.model == "Objective") & (order_scenarios.order_depth == depth)
                            & (order_scenarios.scenario_rank == 1)].iloc[0]

rows = []
for i, col in enumerate(pos_cols, start=1):
    p, o = prod_top1[col], obj_top1[col]
    rows.append({"Position": i, "Production (most likely order)": p, "Objective (most likely order)": o,
                 "Same Player": "Yes" if p == o else "No"})
same_count = sum(r["Same Player"] == "Yes" for r in rows)
st.metric(f"Positions matching in each model's #1 exact order (Top {depth})", f"{same_count} / {depth}")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------------------------------
# E. Betting agreement (prominent)
# --------------------------------------------------------------------------
st.subheader("E. Betting Agreement — Both Models See Value")
st.caption(
    "Verified Sportsbet markets where BOTH models exceed the bookmaker's implied probability, "
    "BOTH have positive expected value, market mapping is verified, and the row is not "
    "PRICE_SUSPECT / SETTLEMENT_UNCERTAIN / REVIEW_REQUIRED. This does not guarantee a winning bet."
)

bet_df = bd.load_verified_opportunities()
if bet_df.empty:
    st.caption(f"No betting data found ({bd.data_source_status()['path']}) -- see the Betting "
               "Opportunities page for how to refresh it.")
else:
    bet_df = bd.with_objective_columns(bet_df)
    agree = bd.both_models_agree_rows(bet_df)
    st.metric("Markets where both models agree on value", len(agree))
    if not agree.empty:
        st.dataframe(bd.for_display_compare(agree), use_container_width=True, hide_index=True)
