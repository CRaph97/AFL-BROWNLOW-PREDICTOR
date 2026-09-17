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
import pandas as pd
import streamlit as st

from dashboard import betting_data as bd

from dashboard import data as d

st.set_page_config(page_title="Betting Opportunities", layout="wide")
d.highlight_objective_stats_nav()
st.title("Betting Opportunities")
st.caption(
    "Sportsbet Brownlow prices compared against this model's probabilities. Read-only: "
    "sourced from the separate AFL-BROWNLOW-MARKETS repo, refreshed manually there."
)

status = bd.data_source_status()
df = bd.load_verified_opportunities()

if df.empty:
    st.warning(
        "No betting-opportunity data found (neither a live AFL-BROWNLOW-MARKETS checkout nor "
        "the bundled deployment snapshot).\n\n"
        f"- Live checkout looked for: `{status['live_path']}` (found: {status['markets_repo_found']})\n"
        f"- Bundled snapshot looked for: `{status['snapshot_path']}` (found: {status['snapshot_available']})\n\n"
        "Run `python -m src.ingestion.refresh_sportsbet` in AFL-BROWNLOW-MARKETS, set the "
        "`AFL_BROWNLOW_MARKETS_PATH` environment variable to point at that checkout, or restore "
        "`data/deployment/sportsbet_verified_value_opportunities.csv`."
    )
    st.stop()

_source_label = {
    "live_markets_repo": "live AFL-BROWNLOW-MARKETS checkout",
    "bundled_deployment_snapshot": "bundled static deployment snapshot (this repo's copy, refreshed manually)",
}.get(status["source_kind"], status["source_kind"])
st.caption(f"Data last refreshed: {df['timestamp'].max()}  ·  source: {_source_label}")
if status["source_kind"] == "bundled_deployment_snapshot":
    st.info(
        "Running from the bundled static snapshot -- this deployment has no access to the "
        "AFL-BROWNLOW-MARKETS repo (expected when hosted, e.g. on Streamlit Community Cloud). "
        "Prices will not update until the snapshot is refreshed and redeployed; see README.md's "
        "Deployment section."
    )

df = bd.with_objective_columns(df)

view = st.radio(
    "Model view", ["Production Model", "Objective Stats Model", "Compare Both"],
    horizontal=True,
    help="Production = the validated, historically-trained ensemble (this page's original view). "
         "Objective = the standalone, experimental, no-historical-votes model. Compare Both shows "
         "both side-by-side and where they agree or disagree.",
)

st.divider()

if view == "Compare Both":
    st.subheader("Compare Both Models")
    st.caption(
        "Objective Prob / EV show \"NOT AVAILABLE\" where this market type or player/team isn't "
        "reliably resolvable from the Objective model's own outputs -- never a fabricated number. "
        "Rows already excluded by the markets repo's own audit (PRICE_SUSPECT / SETTLEMENT_UNCERTAIN "
        "/ REVIEW_REQUIRED) are held out of every section below, including this raw table."
    )
    clean = bd.exclude_audited_rows(df)
    st.dataframe(bd.for_display_compare(clean), use_container_width=True, hide_index=True, height=420)

    st.divider()
    st.subheader("Production Model Value")
    st.caption("Production model sees positive edge and positive EV (audit exclusions applied).")
    st.dataframe(bd.for_display_compare(bd.production_value_rows(df)), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Objective Model Value")
    st.caption("Objective model sees positive edge and positive EV (audit exclusions applied).")
    st.dataframe(bd.for_display_compare(bd.objective_value_rows(df)), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Both Models Agree")
    st.caption(
        "Both models exceed the bookmaker's implied probability AND both have positive expected "
        "value, with verified market mapping. Agreement is a robustness signal, not proof of "
        "correctness -- both models can independently share the same blind spot."
    )
    st.dataframe(bd.for_display_compare(bd.both_models_agree_rows(df)), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Models Disagree")
    st.caption(
        "One model sees value and the other doesn't, or the two probabilities diverge materially "
        "(>=10 percentage points) even when their value conclusion happens to match."
    )
    st.dataframe(bd.for_display_compare(bd.models_disagree_rows(df)), use_container_width=True, hide_index=True)
    st.stop()

if view == "Objective Stats Model":
    st.info(
        "**Objective Stats Model view -- experimental.** Probabilities come from a lightweight "
        "Monte Carlo over the standalone, no-historical-votes model (see the *Objective Stats "
        "Model* and *Guide & FAQs* pages), not the validated production ensemble."
    )
    obj_only = df[df["objective_probability"].notna()].copy()
    obj_only["objective_edge_pp"] = obj_only["objective_edge_pp"].astype(float)
    obj_only["objective_expected_value"] = obj_only["objective_expected_value"].astype(float)
    obj_only = bd.exclude_audited_rows(obj_only)

    def _obj_tier(r) -> str:
        if r["objective_edge_pp"] <= 0:
            return "NO EDGE"
        if r["odds"] >= 15.0 and r["objective_edge_pp"] >= 10.0:
            return "OBJECTIVE SPECULATIVE UPSIDE"
        if r["objective_edge_pp"] >= 5.0 and r["objective_expected_value"] > 0:
            return "OBJECTIVE VALUE"
        return "MARGINAL EDGE"

    obj_only["objective_tier"] = obj_only.apply(_obj_tier, axis=1)

    ff1, ff2 = st.columns(2)
    obj_market_types = sorted(obj_only["market_type"].dropna().unique())
    obj_selected_types = ff1.multiselect("Market type", obj_market_types, default=obj_market_types)
    obj_min_edge = ff2.number_input("Minimum edge (pp)", value=-100.0, step=1.0, key="obj_min_edge")
    obj_filtered = obj_only[obj_only["market_type"].isin(obj_selected_types)
                             & (obj_only["objective_edge_pp"] >= obj_min_edge)]
    st.caption(f"{len(obj_filtered)} of {len(obj_only)} objective-resolvable selections match the current filters.")

    def _obj_section(title: str, tier: str, help_text: str) -> None:
        st.subheader(title)
        st.caption(help_text)
        rows = obj_filtered[obj_filtered["objective_tier"] == tier].sort_values("objective_edge_pp", ascending=False)
        if rows.empty:
            st.caption("No selections in this category match the current filters.")
        else:
            disp = pd.DataFrame({
                "Market": rows["market_name"], "Selection": rows["selection"], "Odds": rows["odds"],
                "Objective Probability": (rows["objective_probability"] * 100).round(1).astype(str) + "%",
                "Implied Probability": (1.0 / rows["odds"] * 100).round(1).astype(str) + "%",
                "Edge (pp)": rows["objective_edge_pp"].round(1),
                "Expected Value": rows["objective_expected_value"].round(3),
                "Audit Status": rows["audit_status"],
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)

    st.divider()
    _obj_section("Objective Value", "OBJECTIVE VALUE",
                 "Objective model shows a real edge (>=5pp) and positive EV against this price. "
                 "Independent of, and not validated the way, the production model's value tiers are.")
    st.divider()
    _obj_section("Objective Speculative Upside", "OBJECTIVE SPECULATIVE UPSIDE",
                 "Longer-priced (odds >= 15), positive-EV, objective-model-supported longshot.")
    st.divider()
    st.subheader("Interpretation")
    st.markdown(
        "- The Objective model is **experimental** and has no historical Brownlow-vote validation "
        "-- treat its edges as a second, differently-motivated opinion, not a superior one.\n"
        "- Rows with no reliable objective probability (e.g. `TRIFECTA`/`EXACTA`/`QUINELLA`/`GROUP_H2H`, "
        "or unresolved player/team names) are excluded from this view entirely, not shown as zero edge.\n"
        "- See the *Model Agreement* page for where this model and the production model independently "
        "reach the same conclusion."
    )
    st.stop()

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
