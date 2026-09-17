"""
Betting Opportunities -- read-only view of Sportsbet Brownlow prices compared
against this model's probabilities. All ingestion, mapping, and value-audit
logic lives in the separate AFL-BROWNLOW-MARKETS repo; this page only reads
its finalized, already-audited output file. See dashboard/betting_data.py's
docstring and AFL-BROWNLOW-MARKETS/docs/VALUE_ENGINE.md for the full pipeline.

This page never changes a model prediction and is not a dependency of any
other page. If the markets repo isn't present or hasn't been refreshed yet,
every section below fails gracefully with an explanatory message.
"""
import streamlit as st

from dashboard import betting_data as bd

st.set_page_config(page_title="Betting Opportunities", layout="wide")
st.title("Betting Opportunities")
st.caption(
    "Sportsbet Brownlow prices compared against this model's probabilities. Read-only: "
    "sourced from the separate AFL-BROWNLOW-MARKETS repo, refreshed manually there."
)

status = bd.data_source_status()
df = bd.load_verified_opportunities()

if df.empty:
    st.warning(
        "No betting-opportunity data found.\n\n"
        f"- Looked for: `{status['path']}`\n"
        f"- AFL-BROWNLOW-MARKETS checkout found: {status['markets_repo_found']}\n"
        f"- Output file found: {status['file_found']}\n\n"
        "Run `python -m src.ingestion.refresh_sportsbet` in AFL-BROWNLOW-MARKETS, or set the "
        "`AFL_BROWNLOW_MARKETS_PATH` environment variable to point at that checkout."
    )
    st.stop()

st.caption(f"Data last refreshed: {df['timestamp'].max()}  ·  source: `{status['path']}`")

st.divider()

# --------------------------------------------------------------------------
# D. Filters
# --------------------------------------------------------------------------
st.subheader("Filters")
f1, f2, f3, f4, f5 = st.columns(5)
market_types = sorted(df["market_type"].dropna().unique())
selected_types = f1.multiselect("Market type", market_types, default=market_types)
min_odds = f2.number_input("Minimum odds", min_value=1.0, value=1.0, step=0.5)
min_edge = f3.number_input("Minimum edge (pp)", value=-100.0, step=1.0)
min_ev = f4.number_input("Minimum expected value", value=-1.0, step=0.05)
quality_labels = sorted(df["quality_label"].dropna().unique())
selected_labels = f5.multiselect("Quality label", quality_labels, default=quality_labels)

filtered = df[
    df["market_type"].isin(selected_types)
    & (df["odds"] >= min_odds)
    & (df["probability_edge_pp"] >= min_edge)
    & (df["expected_value"] >= min_ev)
    & df["quality_label"].isin(selected_labels)
].copy()

st.caption(f"{len(filtered)} of {len(df)} priced selections match the current filters.")

st.divider()


def _section(title: str, label: str, help_text: str) -> None:
    st.subheader(title)
    st.caption(help_text)
    rows = filtered[filtered["quality_label"] == label].sort_values("probability_edge_pp", ascending=False)
    if rows.empty:
        st.caption("No selections in this category match the current filters.")
    else:
        st.dataframe(bd.for_display(rows), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# A. Best verified value
# --------------------------------------------------------------------------
_section(
    "A. Best Verified Value",
    "HIGH-CONFIDENCE VALUE",
    "Passed every mapping/settlement/wording check, meaningful positive edge and EV, "
    "and low/moderate model disagreement and structural-break sensitivity.",
)

st.divider()

# --------------------------------------------------------------------------
# B. Model-sensitive value
# --------------------------------------------------------------------------
_section(
    "B. Model-Sensitive Value",
    "MODEL-SENSITIVE VALUE",
    "Same edge/EV bar as above, but this player's projection has elevated model "
    "disagreement or structural-break sensitivity - the edge is real relative to the "
    "model, but the model itself is less sure here. Treat with more caution.",
)

st.divider()

# --------------------------------------------------------------------------
# C. Speculative upside
# --------------------------------------------------------------------------
_section(
    "C. Speculative Upside",
    "SPECULATIVE UPSIDE",
    "Longer-priced (odds >= 15), positive expected value, lower absolute probability - "
    "still model-supported, but a longshot by nature.",
)

st.divider()

# --------------------------------------------------------------------------
# E. Market detail
# --------------------------------------------------------------------------
st.subheader("E. Market Detail")
if filtered.empty:
    st.caption("No selections match the current filters.")
else:
    options = (filtered["market_name"] + " — " + filtered["selection"]).tolist()
    picked = st.selectbox("Select an opportunity", options)
    row = filtered.iloc[options.index(picked)]

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"**Bookmaker market wording:** {row['market_name']}")
        st.markdown(f"**Selection:** {row['selection']}")
        if isinstance(row.get("player"), str) and row["player"]:
            st.markdown(f"**Player:** {row['player']}")
        if isinstance(row.get("team"), str) and row["team"]:
            st.markdown(f"**Team:** {row['team']}")
        st.markdown(f"**Odds:** {row['odds']:.2f}")
    with d2:
        st.markdown(f"**Model probability:** {row['model_probability'] * 100:.1f}%")
        st.markdown(f"**Implied probability:** {row['implied_probability'] * 100:.1f}%")
        st.markdown(f"**Probability edge:** {row['probability_edge_pp']:.1f}pp")
        st.markdown(f"**Expected value:** {row['expected_value']:.3f}")
    with d3:
        st.markdown(f"**Model disagreement:** {row['model_disagreement']:.2f}")
        st.markdown(f"**Structural-break sensitivity:** {row['structural_break_sensitivity']:.2f}")
        st.markdown(f"**Audit status:** {row['audit_status']}")
        st.markdown(f"**Last updated:** {row['timestamp']}")

    if isinstance(row.get("audit_notes"), str) and row["audit_notes"]:
        st.info(f"Audit note: {row['audit_notes']}")
    if isinstance(row.get("source_url"), str) and row["source_url"]:
        st.markdown(f"[View on Sportsbet]({row['source_url']})")

st.divider()

# --------------------------------------------------------------------------
# F. Interpretation
# --------------------------------------------------------------------------
st.subheader("F. Interpretation")
st.markdown(
    """
- **Model probabilities are estimates**, derived from 100,000 simulated Brownlow seasons under this
  model's assumptions - not certainties.
- **Prices can change.** This page shows a manually-refreshed snapshot, not a live feed - check the
  "Last refreshed" timestamp above before acting on anything here.
- **Positive expected value does not guarantee a winning outcome** on any single bet - it describes
  an average across many repeated, similarly-priced situations.
- **Some 2026 projections are sensitive to the 2026 umpire-statistics regime change** (see this
  dashboard's Uncertainty and Model Disagreement pages) - that sensitivity is exactly what the
  "Model-Sensitive Value" category above is flagging, not a separate concern.
- **`TEAM_VOTES_OU` edges are correlated, not independent** - Sportsbet prices that market at a flat
  1.87/1.87 across every team, varying only the line, so when our simulation disagrees with several
  teams' lines at once it's most likely one shared cause (a line-setting process that lags the
  season), not several unrelated findings.

This page does not size stakes and never will - it compares probabilities and prices, nothing more.
"""
)
