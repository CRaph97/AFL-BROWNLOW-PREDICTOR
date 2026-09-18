"""
Brownlow Betting Opportunities (Neds / PointsBet) -- decision-support only.

Never places a bet, sizes a stake, or automates a bookmaker account. Never
scrapes on page load: everything here is read from
data/betting/processed/*, built by `python scripts/refresh_brownlow_odds.py`.
See docs/BETTING_OPPORTUNITIES.md for the full methodology and the current
live-scraping result (0 real markets obtained this run -- documented
honestly, not hidden).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import betting_opportunities as bo
from dashboard import data as d

st.set_page_config(page_title="Brownlow Betting Opportunities", layout="wide")
d.highlight_objective_stats_nav()
st.title("Brownlow Betting Opportunities")
st.caption(
    "Neds / PointsBet markets priced against Production and Objective's own simulations, "
    "with Wheelo as independent quantitative corroboration. Decision-support only -- this "
    "page never places a bet, sizes a stake, or automates a bookmaker account, and never "
    "changes any model's predictions."
)

summary = bo.load_refresh_summary()
opportunities = bo.load_opportunities()
combinations = bo.load_combinations()
price_comparison = bo.load_price_comparison()

# --------------------------------------------------------------------------
# Top summary
# --------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Odds refresh", summary.get("refresh_completed_at", "never")[:19] if summary.get("refresh_completed_at") else "never")
c2.metric("Markets scraped", summary.get("markets_scraped", 0))
c3.metric("Markets modelled", summary.get("markets_modelled", 0))
c4.metric("High Confidence", summary.get("high_confidence_count", 0))
c5.metric("Medium / Speculative", f"{summary.get('medium_confidence_count', 0)} / {summary.get('speculative_count', 0)}")

with st.expander("Source status / provenance", expanded=opportunities.empty):
    if not summary:
        st.warning("No refresh has been run yet. Run `python scripts/refresh_brownlow_odds.py`.")
    for s in summary.get("source_statuses", []):
        icon = "✅" if s.get("status") == "OK" else "⚠️"
        st.markdown(f"{icon} **{s.get('source')}** -- {s.get('status')}  \n*{s.get('detail', '')}*")
    st.caption("Full methodology and honest accounting of this run's scraping result: docs/BETTING_OPPORTUNITIES.md")

if opportunities.empty:
    st.info(
        "**No priced betting opportunities are currently available.** "
        "The most recent refresh obtained 0 usable markets from Neds/PointsBet "
        "(see Source status above and docs/BETTING_OPPORTUNITIES.md for the full, honest "
        "explanation -- both bookmakers deliver their real odds via client-side JavaScript "
        "that this pipeline does not render, per the project's no-anti-bot-bypass constraint). "
        "The pricing engine, confidence classification, and combination logic below are fully "
        "built and tested (see tests/test_betting_pricing.py, tests/test_betting_classification.py) "
        "and will populate this page automatically the next time a real market row is available."
    )
    st.stop()

st.divider()

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
st.subheader("Filters")
f1, f2, f3, f4, f5, f6 = st.columns(6)
bookmaker_filter = f1.selectbox("Bookmaker", ["Best Odds", "Neds", "PointsBet"])
confidence_options = sorted(opportunities["confidence"].dropna().unique().tolist())
confidence_filter = f2.multiselect("Confidence", confidence_options, default=confidence_options)
market_options = sorted(opportunities["market_type"].dropna().unique().tolist())
market_filter = f3.multiselect("Market type", market_options, default=market_options)
search = f4.text_input("Player / team search")
min_odds = f5.number_input("Minimum odds", min_value=1.0, value=1.0, step=0.5)
min_edge = f6.number_input("Minimum internal edge (pp)", value=-100.0, step=1.0)

filtered = opportunities[
    opportunities["confidence"].isin(confidence_filter)
    & opportunities["market_type"].isin(market_filter)
    & (opportunities["odds"] >= min_odds)
]
if search:
    filtered = filtered[
        filtered["player_name"].str.contains(search, case=False, na=False)
        | filtered.get("team_id", pd.Series(dtype=object)).astype(str).str.contains(search, case=False, na=False)
    ]

sort_option = st.selectbox(
    "Sort by",
    ["Best valid opportunities (default)", "Conservative Edge", "Production Edge",
     "Objective Edge", "Smallest Model Gap", "Odds", "External Agreement"],
)
sort_map = {
    "Conservative Edge": "conservative_internal_probability",
    "Production Edge": "production_edge_pp",
    "Objective Edge": "objective_edge_pp",
    "Smallest Model Gap": "internal_gap_pp",
    "Odds": "odds",
}
if sort_option in sort_map and sort_map[sort_option] in filtered.columns:
    ascending = sort_option == "Smallest Model Gap"
    filtered = filtered.sort_values(sort_map[sort_option], ascending=ascending)

DISPLAY_COLS = [
    "selection", "market_type", "source", "odds", "implied_probability",
    "production_probability", "objective_probability", "internal_gap_pp",
    "production_ev", "objective_ev", "wheelo_support", "external_support",
    "confidence", "data_quality_flags",
]


def _render_table(df: pd.DataFrame, empty_message: str) -> None:
    if df.empty:
        st.caption(empty_message)
        return
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    for _, row in df.iterrows():
        with st.expander(f"{row.get('selection', '?')} -- {row.get('market_type', '?')}"):
            st.markdown(f"**Why classified this way:** {row.get('confidence', '?')}")
            st.markdown(f"- Production probability: {row.get('production_probability')}")
            st.markdown(f"- Objective probability: {row.get('objective_probability')}")
            st.markdown(f"- Implied probability: {row.get('implied_probability')}")
            st.markdown(f"- Wheelo evidence: {row.get('wheelo_ev')} (rank {row.get('wheelo_rank')})")
            st.markdown(f"- External evidence: {row.get('external_support')}")
            st.markdown(f"- Data quality: {row.get('data_quality_flags')}")

FLAGGED_STATUSES = {"PRICE_SUSPECT", "STALE_PRICE", "IDENTITY_AMBIGUOUS",
                    "SETTLEMENT_REVIEW_REQUIRED", "UNMODELLED", "INSUFFICIENT_SIMULATION_SUPPORT"}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Flagged rows must never appear in headline sections."""
    if "data_quality_flags" not in df.columns:
        return df
    return df[~df["data_quality_flags"].fillna("").apply(
        lambda flags: any(f in str(flags) for f in FLAGGED_STATUSES)
    )]


clean_filtered = _clean(filtered)

st.divider()
st.header("1. Best Opportunities")
_render_table(clean_filtered.head(20), "No opportunities currently pass the active filters.")

st.header("2. High Confidence + Wheelo Confirmed")
_render_table(clean_filtered[clean_filtered["confidence"] == "HIGH_CONFIDENCE_WHEELO_CONFIRMED"],
              "None currently.")

st.header("3. High Confidence / Wheelo Neutral")
_render_table(clean_filtered[clean_filtered["confidence"] == "HIGH_CONFIDENCE_WHEELO_NEUTRAL"], "None currently.")

st.header("4. Medium Confidence")
_render_table(clean_filtered[clean_filtered["confidence"] == "MEDIUM_CONFIDENCE"], "None currently.")

st.header("5. High Risk / High Reward")
_render_table(clean_filtered[clean_filtered["confidence"] == "HIGH_RISK_HIGH_REWARD"], "None currently.")

st.header("6. Model Disagreement")
_render_table(clean_filtered[clean_filtered["confidence"] == "MODEL_DISAGREEMENT"], "None currently.")

st.header("7. Best Bookmaker Price")
if price_comparison.empty:
    st.caption("No equivalent Neds/PointsBet market pairs currently available to compare.")
else:
    st.dataframe(price_comparison, use_container_width=True, hide_index=True)

st.header("8. Player Market View")
if "player_name" in opportunities.columns and not opportunities["player_name"].dropna().empty:
    player = st.selectbox("Player", sorted(opportunities["player_name"].dropna().unique()))
    _render_table(clean_filtered[clean_filtered["player_name"] == player], "No markets for this player.")
else:
    st.caption("No player-level markets currently available.")

st.header("9. All Markets")
_render_table(filtered, "No markets currently available.")

st.divider()
st.header("Suggested Combinations")
st.caption(
    "2/3/4/5-leg candidates, joint probabilities computed from real Production/Objective "
    "simulation draws (never a product of marginals). Shows 'candidate combination -- verify "
    "price with bookmaker' instead of a fabricated combined price whenever a real combined "
    "price isn't retrievable."
)
if combinations.empty:
    st.caption("No combination candidates currently available (requires priced single legs, of which there are none this run).")
else:
    for n_legs, label in [(2, "2-leg -- Lower variance"), (3, "3-leg -- Balanced"),
                           (4, "4-leg -- Higher return"), (5, "5-leg -- Speculative")]:
        subset = combinations[combinations["n_legs"] == n_legs]
        st.subheader(label)
        _render_table(subset, "None currently.")
