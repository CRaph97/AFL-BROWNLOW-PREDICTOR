import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="External Winning Order", layout="wide")
d.highlight_objective_stats_nav()
st.title("Winning Order -- Cross-Source Comparison")
st.caption(
    "Only sources that genuinely support a full ranked ordering are shown. Exact-order "
    "simulation probabilities (Production/Objective Monte Carlo) are NOT available for "
    "external sources that don't run their own simulation -- shown separately, never blended."
)

df = ed.load_external_overview()
if df.empty:
    st.warning("No external overview data. Run `python scripts/refresh_external_benchmarks.py` first.")
    st.stop()

n = st.radio("Top N", [5, 7, 10, 15, 20], index=2, horizontal=True)

rank_cols = {
    "Production": "production_rank", "Objective": "objective_rank",
    "Wheelo": "wheelo_ev", "ESPN": "espn_ev", "Betfair": "betfair_ev",
    "External Consensus": "external_consensus_rank",
}
orderings = {}
for label, col in rank_cols.items():
    sub = df.dropna(subset=[col]).copy()
    if sub.empty:
        continue
    ascending = col.endswith("_rank")
    sub = sub.sort_values(col, ascending=ascending).head(n)
    orderings[label] = sub["player_name"].tolist()

st.subheader(f"Top {n} side-by-side")
max_len = max(len(v) for v in orderings.values())
table = {label: names + [""] * (max_len - len(names)) for label, names in orderings.items()}
st.dataframe(table, use_container_width=True, hide_index=True)

st.divider()
all_names = set()
for v in orderings.values():
    all_names.update(v)
in_every = sorted(n_ for n_ in all_names if all(n_ in v for v in orderings.values()))
unique_to_one = sorted(n_ for n_ in all_names if sum(n_ in v for v in orderings.values()) == 1)

c1, c2 = st.columns(2)
c1.markdown(f"**Players in every Top-{n} list ({len(in_every)})**")
c1.write(in_every or "None")
c2.markdown(f"**Players unique to one methodology ({len(unique_to_one)})**")
c2.write(unique_to_one or "None")

st.divider()
st.subheader("Rank spread across sources")
rank_only = {}
for label, col in rank_cols.items():
    if col.endswith("_rank") and col in df.columns:
        rank_only[label] = df.set_index("player_name")[col]
    elif col in df.columns:
        rank_only[label] = df.set_index("player_name")[col].rank(ascending=False, method="min")
rank_df = None
import pandas as pd
if rank_only:
    rank_df = pd.DataFrame(rank_only)
    rank_df["min_rank"] = rank_df.min(axis=1)
    rank_df["max_rank"] = rank_df.max(axis=1)
    rank_df["rank_spread"] = rank_df["max_rank"] - rank_df["min_rank"]
    rank_df = rank_df.loc[rank_df.index.isin(all_names)].sort_values("rank_spread", ascending=False)
    st.dataframe(rank_df, use_container_width=True)

st.divider()
st.subheader("Production / Objective exact-order simulation scenarios (reused, not blended with external)")
order_scenarios = ed.load_order_scenarios()
depth = n if n in (5, 7, 10) else 10
for model in ["Production", "Objective"]:
    st.markdown(f"**{model} -- Top {depth} exact-order scenarios**")
    sub = order_scenarios[(order_scenarios["model"] == model) & (order_scenarios["order_depth"] == depth)].head(5)
    if sub.empty:
        st.caption("No exact-order scenario data available at this depth.")
    else:
        st.dataframe(sub, use_container_width=True, hide_index=True)
