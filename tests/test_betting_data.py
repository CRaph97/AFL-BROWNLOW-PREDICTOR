"""
Tests for dashboard/betting_data.py - the read-only bridge into the separate
AFL-BROWNLOW-MARKETS repo. Does not touch, and is not touched by, any of the
Brownlow model's own tests.
"""
from __future__ import annotations

import pandas as pd

from dashboard import betting_data as bd


def test_loads_real_file_when_present():
    status = bd.data_source_status()
    if not status["file_found"]:
        return  # markets repo not checked out / not refreshed on this machine - not this repo's problem
    df = bd.load_verified_opportunities()
    assert not df.empty
    for col in ("market_type", "market_name", "selection", "odds", "model_probability",
                "implied_probability", "probability_edge_pp", "expected_value", "quality_label",
                "audit_status", "timestamp", "source_url"):
        assert col in df.columns


def test_missing_file_returns_empty_dataframe_not_an_exception(tmp_path):
    df = bd._read_verified_csv(tmp_path / "does-not-exist.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_for_display_renames_and_formats_columns():
    df = pd.DataFrame(
        [
            {
                "market_type": "WINNER", "market_name": "2026 AFL Brownlow Medal", "selection": "Player A",
                "odds": 2.0, "model_probability": 0.6, "implied_probability": 0.5,
                "probability_edge_pp": 10.0, "expected_value": 0.2, "model_disagreement": 0.1,
                "structural_break_sensitivity": 0.05, "quality_label": "HIGH-CONFIDENCE VALUE",
                "timestamp": "2026-09-17T00:00:00+00:00",
            }
        ]
    )
    out = bd.for_display(df)
    assert "Market" in out.columns and "Edge (pp)" in out.columns
    assert out["Model Probability"].iloc[0] == "60.0%"
    assert out["Implied Probability"].iloc[0] == "50.0%"


def test_data_source_status_reports_configured_path():
    status = bd.data_source_status()
    assert status["path"].endswith("sportsbet_verified_value_opportunities.csv")
