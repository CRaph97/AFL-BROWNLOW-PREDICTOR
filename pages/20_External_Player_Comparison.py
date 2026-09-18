import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="External Player Comparison", layout="wide")
d.highlight_objective_stats_nav()
st.title("External Player Comparison")
st.caption("Evidence, not a recommendation -- no subjective winner is declared.")

df = ed.load_external_overview()
if df.empty:
    st.warning("No external overview data. Run `python scripts/refresh_external_benchmarks.py` first.")
    st.stop()

names = sorted(df["player_name"].dropna().unique().tolist())
c1, c2 = st.columns(2)
a = c1.selectbox("Player A", names, index=names.index("Nick Daicos") if "Nick Daicos" in names else 0)
b = c2.selectbox("Player B", names, index=1 if len(names) > 1 else 0)

cols = ["player_name", "production_ev", "production_rank", "objective_ev", "objective_rank",
        "wheelo_ev", "espn_ev", "betfair_ev", "external_consensus_ev", "external_consensus_rank",
        "our_midpoint", "our_internal_gap", "external_range", "agreement_category"]
row_a = df[df["player_name"] == a][cols].iloc[0]
row_b = df[df["player_name"] == b][cols].iloc[0]

st.divider()
c1, c2 = st.columns(2)
for col, row, name in [(c1, row_a, a), (c2, row_b, b)]:
    with col:
        st.subheader(name)
        st.metric("Production EV", f"{row['production_ev']:.1f}" if row['production_ev']==row['production_ev'] else "N/A",
                   f"rank {int(row['production_rank'])}" if row['production_rank']==row['production_rank'] else None)
        st.metric("Objective EV", f"{row['objective_ev']:.1f}" if row['objective_ev']==row['objective_ev'] else "N/A",
                   f"rank {int(row['objective_rank'])}" if row['objective_rank']==row['objective_rank'] else None)
        st.metric("External Consensus", f"{row['external_consensus_ev']:.1f}" if row['external_consensus_ev']==row['external_consensus_ev'] else "N/A",
                   f"rank {int(row['external_consensus_rank'])}" if row['external_consensus_rank']==row['external_consensus_rank'] else None)
        st.caption(f"Agreement: **{row['agreement_category']}**")

st.divider()
st.subheader("Model/source EV gap between the two players")
gap_rows = []
for label, key in [("Production", "production_ev"), ("Objective", "objective_ev"),
                    ("External Consensus", "external_consensus_ev"), ("Our Midpoint", "our_midpoint")]:
    va, vb = row_a[key], row_b[key]
    gap_rows.append({"Source": label, a: va, b: vb, "Gap (A-B)": (va - vb) if va == va and vb == vb else None})
st.dataframe(gap_rows, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Round-by-round Wheelo comparison")
for name in (a, b):
    pid = df[df["player_name"] == name]["player_id"].iloc[0]
    rbr = ed.player_round_by_round_external(pid)
    st.markdown(f"**{name}**")
    if rbr.empty or rbr["wheelo_ev"].isna().all():
        st.caption("No Wheelo match-level data resolved for this player.")
    else:
        st.dataframe(rbr, use_container_width=True, hide_index=True, height=250)
        cum = rbr.copy()
        cum["production_cum"] = cum["production_ev"].fillna(0).cumsum()
        cum["wheelo_cum"] = cum["wheelo_ev"].fillna(0).cumsum()
        st.line_chart(cum.set_index("round")[["production_cum", "wheelo_cum"]])

st.divider()
st.subheader("Disagreement origin (evidence, not a verdict)")
for name in (a, b):
    pid = df[df["player_name"] == name]["player_id"].iloc[0]
    rbr = ed.player_round_by_round_external(pid)
    valid = rbr.dropna(subset=["production_ev", "wheelo_ev"])
    if valid.empty:
        continue
    per_round_gap = (valid["production_ev"] - valid["wheelo_ev"]).abs()
    n_large = (per_round_gap > per_round_gap.mean() + per_round_gap.std()).sum() if len(per_round_gap) > 2 else 0
    if n_large <= 1:
        origin = "isolated game(s) -- one or two rounds drive most of the gap"
    elif n_large / max(len(per_round_gap), 1) > 0.5:
        origin = "persistent, season-wide difference across most rounds"
    else:
        origin = "model-specific divergence spread across several rounds"
    st.markdown(f"**{name}**: {origin} ({n_large} of {len(per_round_gap)} rounds show an outsized Production-vs-Wheelo gap)")
