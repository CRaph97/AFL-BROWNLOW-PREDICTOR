"""
Order Scenarios -- exact ordered finishing scenarios for the Production and
Objective models, side by side. Read-only: reports/2026_order_scenarios.csv
is pre-computed by src/models/build_2026_order_scenarios.py from each
model's own raw Monte Carlo draws (production: 100,000 sims, persisted by
run_2026_montecarlo.py; objective: 20,000 sims, persisted by
run_2026_objective_montecarlo.py, since the Objective model doesn't
otherwise keep a season simulation). This page never recomputes a
probability -- it only displays what those scripts already produced.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data as d
from src.models.order_scenarios import POSITION_COLS, contender_probabilities

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

st.set_page_config(page_title="Order Scenarios", layout="wide")
d.highlight_objective_stats_nav()
st.title("Order Scenarios")
st.caption(
    "Exact ordered finishing scenarios (not just individual player rankings) for both models. "
    "The Objective model's simulation is a lightweight, clearly-separate 20,000-run Monte Carlo "
    "over its own match probabilities -- see the Guide & FAQs page for what the Objective model is."
)

@st.cache_data
def load_scenarios() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "2026_order_scenarios.csv")


@st.cache_data
def load_totals(model: str):
    if model == "Production":
        totals = np.load(PROCESSED_DIR / "mc_totals_2026.npy")
        players = pd.read_csv(REPORTS_DIR / "2026_mc_player_index.csv")
    else:
        totals = np.load(PROCESSED_DIR / "mc_totals_objective_2026.npy")
        players = pd.read_csv(REPORTS_DIR / "2026_objective_mc_player_index.csv")
    return totals, players


try:
    scenarios = load_scenarios()
except FileNotFoundError:
    st.warning("reports/2026_order_scenarios.csv not found -- run "
               "`python -m src.models.build_2026_order_scenarios` first.")
    st.stop()

depth = st.radio("Finishing depth", [5, 7, 10], format_func=lambda n: f"Top {n}", horizontal=True)

st.divider()

pos_cols = POSITION_COLS[:depth]
prod_col, obj_col = st.columns(2)

for col, model_label, header in [(prod_col, "Production", "PRODUCTION MODEL"),
                                   (obj_col, "Objective", "OBJECTIVE MODEL")]:
    with col:
        st.subheader(f"Top {depth} — {header}")
        rows = scenarios[(scenarios["model"] == model_label) & (scenarios["order_depth"] == depth)] \
            .sort_values("scenario_rank")
        cum = 0.0
        for _, r in rows.iterrows():
            cum += r["probability"]
            order_str = " → ".join(str(r[c]) for c in pos_cols)
            st.markdown(f"**{int(r['scenario_rank'])}.** {order_str}")
            st.caption(f"Probability: {r['probability'] * 100:.2f}%")
        st.info(f"Cumulative probability of these {len(rows)} scenarios: {cum * 100:.2f}%")

st.divider()
st.subheader("Contender Probabilities: Winner / Top 2 / Top 3 / Top 5 / Top 7 / Top 10")

c1, c2 = st.columns(2)
for col, model_label in [(c1, "Production"), (c2, "Objective")]:
    with col:
        st.markdown(f"**{model_label}**")
        totals, players = load_totals(model_label)
        contenders = contender_probabilities(totals, players).head(15)
        disp = contenders[["player_name", "prob_winner", "prob_top2", "prob_top3",
                            "prob_top5", "prob_top7", "prob_top10"]].copy()
        for c in disp.columns[1:]:
            disp[c] = (disp[c] * 100).round(1).astype(str) + "%"
        disp.columns = ["Player", "Winner", "Top 2", "Top 3", "Top 5", "Top 7", "Top 10"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
