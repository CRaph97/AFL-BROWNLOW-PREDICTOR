import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="External Overview", layout="wide")
d.highlight_objective_stats_nav()
st.title("External Consensus / Overview")
st.caption(
    "Production and Objective compared against independent third-party predictors. "
    "External data is evaluation/context only -- it never feeds back into either model."
)

status = ed.load_source_status()
with st.expander("Source status / provenance", expanded=False):
    st.caption(f"Last refreshed: {status.get('retrieved_at', 'never')}")
    for name, info in status.get("sources", {}).items():
        icon = {"ok": "✅", "unavailable": "⚠️", "failed": "❌", "parsed_empty": "⚠️"}.get(info.get("status"), "❔")
        st.markdown(f"{icon} **{name}** -- {info.get('status')}")
        if info.get("reason"):
            st.caption(info["reason"])
        if info.get("note"):
            st.caption(info["note"])

df = ed.load_external_overview()
if df.empty:
    st.warning("No external overview data. Run `python scripts/refresh_external_benchmarks.py` first.")
    st.stop()

st.divider()
view = st.radio(
    "View",
    ["Top contenders", "Strongest convergence", "Biggest external disagreement",
     "Production outliers", "Objective outliers", "Wheelo outliers", "Source availability"],
    horizontal=True,
)

display_cols = [
    "player_name", "team_id", "production_ev", "objective_ev", "wheelo_ev", "espn_ev", "betfair_ev",
    "external_consensus_ev", "our_midpoint", "production_rank", "objective_rank", "external_consensus_rank",
    "our_external_gap", "our_internal_gap", "external_range", "agreement_category", "external_source_label",
]
display_cols = [c for c in display_cols if c in df.columns]

if view == "Top contenders":
    sub = df.sort_values("production_ev", ascending=False, na_position="last").head(30)
elif view == "Strongest convergence":
    sub = df[df["agreement_category"] == "STRONG CONVERGENCE"].sort_values("production_ev", ascending=False).head(30)
elif view == "Biggest external disagreement":
    sub = df.reindex(df["our_external_gap"].abs().sort_values(ascending=False, na_position="last").index).head(30)
elif view == "Production outliers":
    sub = df[df["agreement_category"] == "PRODUCTION + EXTERNAL AGREE"].sort_values("production_ev", ascending=False).head(30)
elif view == "Objective outliers":
    sub = df[df["agreement_category"] == "OBJECTIVE + EXTERNAL AGREE"].sort_values("objective_ev", ascending=False).head(30)
elif view == "Wheelo outliers":
    sub = df.reindex(df["production_ev"].sub(df["wheelo_ev"]).abs().sort_values(ascending=False, na_position="last").index).head(30)
else:
    sub = df["external_source_label"].value_counts().rename_axis("source combination").reset_index(name="n_players")
    st.dataframe(sub, use_container_width=True, hide_index=True)
    st.stop()

st.dataframe(sub[display_cols], use_container_width=True, hide_index=True, height=700)
st.caption(
    "Agreement categories use fixed, documented thresholds (±15% relative or ±2.0 votes absolute, "
    "whichever is looser) fixed before results were examined -- see src/external/aggregate.py."
)
