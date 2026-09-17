"""
Read-only bridge into the separate AFL-BROWNLOW-MARKETS repo's finalized
Sportsbet value-scan output.

STRICT RULE: this module never computes a probability, an edge, an expected
value, or a quality label itself - it only loads the CSV that repo's own
pipeline already finalized and does light presentational reshaping (column
renames for display, filtering). All of that math, and the audit that gates
it, lives in AFL-BROWNLOW-MARKETS (see its docs/VALUE_ENGINE.md and
docs/SPORTSBET_VALUE_AUDIT.md) - this repo's Brownlow model is never touched
by, or a dependency of, this module.

Path is configured via the AFL_BROWNLOW_MARKETS_PATH environment variable,
defaulting to a sibling checkout at ~/code/AFL-BROWNLOW-MARKETS (see that
repo's docs/INTEGRATION_PLAN.md for why a direct file read was chosen over an
export/copy step). If the repo or its output file isn't present, every loader
here returns an empty DataFrame rather than raising, so the Streamlit page
can fail gracefully.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_MARKETS_REPO = Path.home() / "code" / "AFL-BROWNLOW-MARKETS"
MARKETS_REPO = Path(os.environ.get("AFL_BROWNLOW_MARKETS_PATH", str(DEFAULT_MARKETS_REPO)))
VERIFIED_CSV = MARKETS_REPO / "reports" / "sportsbet_verified_value_opportunities.csv"

DISPLAY_COLUMNS = {
    "market_name": "Market",
    "selection": "Selection",
    "odds": "Odds",
    "model_probability": "Model Probability",
    "implied_probability": "Implied Probability",
    "probability_edge_pp": "Edge (pp)",
    "expected_value": "Expected Value",
    "model_disagreement": "Model Disagreement",
    "structural_break_sensitivity": "Structural-Break Sensitivity",
    "quality_label": "Quality Label",
    "timestamp": "Last Updated",
}

MAIN_TABLE_COLUMNS = list(DISPLAY_COLUMNS.keys())


def data_source_status() -> dict:
    """Presentational only - used by the page to explain why a table might be empty."""
    return {
        "path": str(VERIFIED_CSV),
        "markets_repo_found": MARKETS_REPO.exists(),
        "file_found": VERIFIED_CSV.exists(),
    }


def _read_verified_csv(path: Path) -> pd.DataFrame:
    """Uncached core logic, so tests can exercise the missing-file path directly
    without fighting st.cache_data's cross-test caching."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("odds", "model_probability", "implied_probability", "probability_edge_pp",
                "expected_value", "model_disagreement", "structural_break_sensitivity",
                "simulation_uncertainty", "line"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_verified_opportunities() -> pd.DataFrame:
    return _read_verified_csv(VERIFIED_CSV)


def for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Renamed/rounded copy for the summary tables (sections A/B/C). Presentational only."""
    out = df[MAIN_TABLE_COLUMNS].copy()
    out["model_probability"] = (out["model_probability"] * 100).round(1)
    out["implied_probability"] = (out["implied_probability"] * 100).round(1)
    out["probability_edge_pp"] = out["probability_edge_pp"].round(1)
    out["expected_value"] = out["expected_value"].round(3)
    out["model_disagreement"] = out["model_disagreement"].round(2)
    out["structural_break_sensitivity"] = out["structural_break_sensitivity"].round(2)
    out = out.rename(columns=DISPLAY_COLUMNS)
    out["Model Probability"] = out["Model Probability"].astype(str) + "%"
    out["Implied Probability"] = out["Implied Probability"].astype(str) + "%"
    return out
