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

import numpy as np
import pandas as pd
import streamlit as st

from src.models.objective_market_probability import load_objective_market_model, objective_probability_for_row

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


# Audit statuses that must never appear in a headline value section, in either
# the Production-only, Objective-only, or Compare-Both views. Mirrors the
# vocabulary already produced by AFL-BROWNLOW-MARKETS' src/audit/value_audit.py
# (read, never modified, by this repo).
EXCLUDED_AUDIT_STATUSES = {"PRICE_SUSPECT", "SETTLEMENT_UNCERTAIN", "REVIEW_REQUIRED"}


def _implied_probability(odds: float) -> float:
    return 1.0 / odds


def _probability_edge_pp(probability: float, odds: float) -> float:
    """Same formula as AFL-BROWNLOW-MARKETS' src/value/valuation.py
    (probability_edge_pp) -- duplicated here, not imported, per the repo
    boundary: this is a two-line formula, not ingestion/scraping/mapping
    logic, and keeping the Objective-side numbers comparable to the
    already-computed Production-side numbers requires using the same one."""
    return (probability - _implied_probability(odds)) * 100.0


def _expected_value(probability: float, odds: float) -> float:
    """Same formula as AFL-BROWNLOW-MARKETS' src/value/valuation.py
    (expected_value), for the same reason as _probability_edge_pp above."""
    return probability * odds - 1.0


@st.cache_resource
def _objective_market_model():
    return load_objective_market_model()


def with_objective_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds production_probability/edge/EV (renamed copies of the existing,
    already-computed model_probability/probability_edge_pp/expected_value --
    never recomputed), plus objective_probability/edge/EV computed fresh from
    this repo's own Objective Stats Model simulation, plus
    probability_difference. Rows where the objective side can't be reliably
    derived get pd.NA (rendered as "NOT AVAILABLE" by for_display_compare),
    never a fabricated number.
    """
    out = df.copy()
    out["production_probability"] = out["model_probability"]
    out["production_edge_pp"] = out["probability_edge_pp"]
    out["production_expected_value"] = out["expected_value"]

    model = _objective_market_model()
    if model is None:
        out["objective_probability"] = pd.NA
        out["objective_edge_pp"] = pd.NA
        out["objective_expected_value"] = pd.NA
        out["objective_unavailable_reason"] = "Objective simulation outputs not found"
        out["probability_difference"] = pd.NA
        return out

    probs, reasons = [], []
    for _, row in out.iterrows():
        p, reason = objective_probability_for_row(model, row)
        probs.append(p)
        reasons.append(reason)
    out["objective_probability"] = probs
    out["objective_unavailable_reason"] = reasons

    has_p = out["objective_probability"].notna()
    out["objective_edge_pp"] = pd.NA
    out["objective_expected_value"] = pd.NA
    out.loc[has_p, "objective_edge_pp"] = out.loc[has_p].apply(
        lambda r: _probability_edge_pp(r["objective_probability"], r["odds"]), axis=1)
    out.loc[has_p, "objective_expected_value"] = out.loc[has_p].apply(
        lambda r: _expected_value(r["objective_probability"], r["odds"]), axis=1)

    out["probability_difference"] = pd.NA
    out.loc[has_p, "probability_difference"] = (
        out.loc[has_p, "objective_probability"] - out.loc[has_p, "production_probability"]
    )
    return out


def exclude_audited_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows whose audit_status is in EXCLUDED_AUDIT_STATUSES. Use this
    before building any headline section (Production Value / Objective Value
    / Both Models Agree / Models Disagree) -- never after."""
    return df[~df["audit_status"].isin(EXCLUDED_AUDIT_STATUSES)]


def production_value_rows(df: pd.DataFrame) -> pd.DataFrame:
    clean = exclude_audited_rows(df)
    return clean[(clean["production_probability"] > _implied_probability(clean["odds"]))
                 & (clean["production_expected_value"] > 0)]


def objective_value_rows(df: pd.DataFrame) -> pd.DataFrame:
    clean = exclude_audited_rows(df)
    has_p = clean["objective_probability"].notna()
    clean = clean[has_p]
    return clean[(clean["objective_probability"] > _implied_probability(clean["odds"]))
                 & (clean["objective_expected_value"] > 0)]


def both_models_agree_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Both positive EV, both above implied probability, verified mapping
    (audit exclusions already applied)."""
    clean = exclude_audited_rows(df)
    has_p = clean["objective_probability"].notna()
    clean = clean[has_p]
    implied = _implied_probability(clean["odds"])
    both_above_implied = (clean["production_probability"] > implied) & (clean["objective_probability"] > implied)
    both_positive_ev = (clean["production_expected_value"] > 0) & (clean["objective_expected_value"] > 0)
    return clean[both_above_implied & both_positive_ev]


def models_disagree_rows(df: pd.DataFrame, material_diff_pp: float = 10.0) -> pd.DataFrame:
    """One model sees value and the other doesn't, OR the two probabilities
    diverge by more than `material_diff_pp` percentage points."""
    clean = exclude_audited_rows(df)
    has_p = clean["objective_probability"].notna()
    clean = clean[has_p]
    implied = _implied_probability(clean["odds"])
    prod_sees_value = (clean["production_probability"] > implied) & (clean["production_expected_value"] > 0)
    obj_sees_value = (clean["objective_probability"] > implied) & (clean["objective_expected_value"] > 0)
    one_sided = prod_sees_value != obj_sees_value
    material_divergence = (clean["objective_probability"] - clean["production_probability"]).abs() * 100 >= material_diff_pp
    return clean[one_sided | material_divergence]


def for_display_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Renamed/rounded copy for the Compare-Both table, with the exact
    columns the brief specifies. 'NOT AVAILABLE' replaces a missing
    objective-side number rather than leaving it blank/ambiguous."""
    out = pd.DataFrame({
        "Market": df["market_name"],
        "Selection": df["selection"],
        "Odds": df["odds"].round(2),
        "Implied Prob": (_implied_probability(df["odds"]) * 100).round(1).astype(str) + "%",
        "Production Prob": (df["production_probability"] * 100).round(1).astype(str) + "%",
        "Objective Prob": df["objective_probability"].apply(
            lambda p: "NOT AVAILABLE" if pd.isna(p) else f"{p * 100:.1f}%"),
        "Production EV": df["production_expected_value"].round(3),
        "Objective EV": df["objective_expected_value"].apply(
            lambda v: "NOT AVAILABLE" if pd.isna(v) else f"{float(v):.3f}"),
        "Probability Difference": df["probability_difference"].apply(
            lambda v: "NOT AVAILABLE" if pd.isna(v) else f"{v * 100:+.1f}pp"),
        "Audit Status": df["audit_status"],
    })
    return out


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
