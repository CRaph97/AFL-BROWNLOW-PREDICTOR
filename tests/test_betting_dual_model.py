"""
Tests for the Production/Objective dual-model additions to
dashboard/betting_data.py -- specifically that excluded audit statuses never
leak into a headline section, and that the four agreement/disagreement
sections use the exact, documented thresholds (not an opaque score).
"""
from __future__ import annotations

import pandas as pd

from dashboard import betting_data as bd

EXCLUDED = bd.EXCLUDED_AUDIT_STATUSES


def _sample_df():
    return pd.DataFrame([
        # verified, both models see value -> should appear in "both agree"
        {"market_type": "WINNER", "market_name": "2026 AFL Brownlow Medal", "selection": "Nick Daicos",
         "player": "Nick Daicos", "team": "", "opponent_or_pair": "", "line": None, "odds": 3.0,
         "implied_probability": 1 / 3, "model_probability": 0.5, "probability_edge_pp": (0.5 - 1 / 3) * 100,
         "expected_value": 0.5 * 3 - 1, "model_disagreement": 0.1, "structural_break_sensitivity": 0.1,
         "simulation_uncertainty": 1.0, "confidence_tier": "LOW", "audit_status": "VERIFIED",
         "audit_notes": "", "quality_label": "HIGH-CONFIDENCE VALUE", "source_url": "", "timestamp": "t"},
        # excluded audit status -> must never appear in any headline section
        {"market_type": "WINNER", "market_name": "2026 AFL Brownlow Medal", "selection": "Nick Daicos",
         "player": "Nick Daicos", "team": "", "opponent_or_pair": "", "line": None, "odds": 3.0,
         "implied_probability": 1 / 3, "model_probability": 0.6, "probability_edge_pp": 30.0,
         "expected_value": 0.8, "model_disagreement": 0.1, "structural_break_sensitivity": 0.1,
         "simulation_uncertainty": 1.0, "confidence_tier": "LOW", "audit_status": "PRICE_SUSPECT",
         "audit_notes": "", "quality_label": "HIGH-CONFIDENCE VALUE", "source_url": "", "timestamp": "t"},
    ])


def test_excluded_audit_statuses_never_appear_in_agreement_sections():
    df = bd.with_objective_columns(_sample_df())
    for fn in (bd.production_value_rows, bd.objective_value_rows, bd.both_models_agree_rows, bd.models_disagree_rows):
        result = fn(df)
        assert not result["audit_status"].isin(EXCLUDED).any(), f"{fn.__name__} leaked an excluded row"


def test_both_models_agree_requires_positive_ev_and_above_implied_on_both_sides():
    df = pd.DataFrame([
        {"market_type": "WINNER", "market_name": "m", "selection": "A", "player": "A", "team": "",
         "opponent_or_pair": "", "line": None, "odds": 2.0, "audit_status": "VERIFIED",
         "model_probability": 0.6, "implied_probability": 0.5, "probability_edge_pp": 10.0,
         "expected_value": 0.2, "model_disagreement": 0.0, "structural_break_sensitivity": 0.0,
         "quality_label": "x", "timestamp": "t"},
    ])
    out = bd.with_objective_columns(df)
    out["objective_probability"] = [0.55]  # above implied (0.5), positive EV at odds 2.0
    out["objective_edge_pp"] = [5.0]
    out["objective_expected_value"] = [0.1]
    out["probability_difference"] = [out["objective_probability"][0] - out["production_probability"][0]]
    agree = bd.both_models_agree_rows(out)
    assert len(agree) == 1

    out.loc[0, "objective_expected_value"] = -0.05  # objective side no longer positive EV
    disagree_source = out.copy()
    agree2 = bd.both_models_agree_rows(disagree_source)
    assert len(agree2) == 0
    disagree = bd.models_disagree_rows(disagree_source)
    assert len(disagree) == 1


def test_not_available_used_instead_of_fabricated_number_for_unsupported_market():
    df = pd.DataFrame([
        {"market_type": "TRIFECTA", "market_name": "m", "selection": "A/B/C", "player": None, "team": "",
         "opponent_or_pair": "", "line": None, "odds": 10.0, "audit_status": "VERIFIED",
         "model_probability": 0.05, "implied_probability": 0.1, "probability_edge_pp": -5.0,
         "expected_value": -0.5, "model_disagreement": 0.0, "structural_break_sensitivity": 0.0,
         "quality_label": "x", "timestamp": "t"},
    ])
    out = bd.with_objective_columns(df)
    assert pd.isna(out.loc[0, "objective_probability"])
    disp = bd.for_display_compare(out)
    assert disp.loc[0, "Objective Prob"] == "NOT AVAILABLE"
    assert disp.loc[0, "Objective EV"] == "NOT AVAILABLE"
