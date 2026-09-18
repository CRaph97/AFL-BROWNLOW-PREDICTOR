"""
Unit tests for the confidence-classification taxonomy in
src/betting/classification.py. Every threshold is fixed by the project
brief -- these tests exercise the documented boundary conditions, not real
market rows (this task obtained zero real scraped bookmaker rows -- see
docs/BETTING_OPPORTUNITIES.md)."""
from __future__ import annotations

from src.betting import classification as c


def test_high_confidence_wheelo_confirmed():
    r = c.classify_confidence(
        production_edge_pp=5.0, objective_edge_pp=4.0, internal_gap_pp=3.0,
        wheelo_support="STRONG_WHEELO_SUPPORT", settlement_flag=None,
    )
    assert r.confidence == "HIGH_CONFIDENCE_WHEELO_CONFIRMED"


def test_high_confidence_wheelo_neutral():
    r = c.classify_confidence(
        production_edge_pp=5.0, objective_edge_pp=4.0, internal_gap_pp=3.0,
        wheelo_support="WHEELO_NEUTRAL", settlement_flag=None,
    )
    assert r.confidence == "HIGH_CONFIDENCE_WHEELO_NEUTRAL"


def test_wheelo_disagreement_downgrades_high_confidence_to_medium():
    r = c.classify_confidence(
        production_edge_pp=5.0, objective_edge_pp=4.0, internal_gap_pp=3.0,
        wheelo_support="WHEELO_DISAGREES", settlement_flag=None,
    )
    assert r.confidence == "MEDIUM_CONFIDENCE"


def test_gap_exactly_at_threshold_is_still_high_confidence():
    r = c.classify_confidence(
        production_edge_pp=5.0, objective_edge_pp=4.0, internal_gap_pp=7.5,
        wheelo_support="WHEELO_NEUTRAL", settlement_flag=None,
    )
    assert r.confidence.startswith("HIGH_CONFIDENCE")


def test_gap_just_over_threshold_is_medium():
    r = c.classify_confidence(
        production_edge_pp=5.0, objective_edge_pp=4.0, internal_gap_pp=7.51,
        wheelo_support="WHEELO_NEUTRAL", settlement_flag=None,
    )
    assert r.confidence == "MEDIUM_CONFIDENCE"
    assert r.outlier_model in ("Production", "Objective")


def test_only_one_model_positive_with_wheelo_support_is_high_risk_high_reward():
    r = c.classify_confidence(
        production_edge_pp=6.0, objective_edge_pp=-2.0, internal_gap_pp=None,
        wheelo_support="STRONG_WHEELO_SUPPORT", settlement_flag=None,
    )
    assert r.confidence == "HIGH_RISK_HIGH_REWARD"


def test_only_one_model_positive_without_wheelo_support_is_model_disagreement():
    r = c.classify_confidence(
        production_edge_pp=6.0, objective_edge_pp=-2.0, internal_gap_pp=None,
        wheelo_support="INSUFFICIENT_WHEELO_DATA", settlement_flag=None,
    )
    assert r.confidence == "MODEL_DISAGREEMENT"


def test_neither_model_positive_is_no_value():
    r = c.classify_confidence(
        production_edge_pp=-3.0, objective_edge_pp=-1.0, internal_gap_pp=2.0,
        wheelo_support="WHEELO_NEUTRAL", settlement_flag=None,
    )
    assert r.confidence == "NO_VALUE"


def test_settlement_flag_forces_no_value_regardless_of_edges():
    r = c.classify_confidence(
        production_edge_pp=10.0, objective_edge_pp=10.0, internal_gap_pp=0.0,
        wheelo_support="STRONG_WHEELO_SUPPORT", settlement_flag="PRICE_SUSPECT",
    )
    assert r.confidence == "NO_VALUE"


def test_media_agreement_alone_cannot_create_high_confidence():
    """Both edges negative (no real internal value) but external context
    fully supportive must never become High Confidence -- media/context
    agreement is explicitly excluded from ever creating confidence on its
    own, per the project brief."""
    r = c.classify_confidence(
        production_edge_pp=-1.0, objective_edge_pp=-1.0, internal_gap_pp=0.0,
        wheelo_support="STRONG_WHEELO_SUPPORT", settlement_flag=None,
    )
    assert r.confidence != "HIGH_CONFIDENCE_WHEELO_CONFIRMED"
    assert r.confidence == "NO_VALUE"


def test_wheelo_support_insufficient_data_when_wheelo_missing():
    label = c.classify_wheelo_support(True, None, 0.55, 0.50)
    assert label == "INSUFFICIENT_WHEELO_DATA"


def test_wheelo_support_strong_when_close_to_our_ev():
    label = c.classify_wheelo_support(True, wheelo_ev=25.0, production_ev=25.5, objective_ev=None)
    assert label == "STRONG_WHEELO_SUPPORT"


def test_external_support_labels():
    assert c.classify_external_support([]) == "INSUFFICIENT_DATA"
    assert c.classify_external_support(["supports", "supports"]) == "BROADER_EXTERNAL_SUPPORT"
    assert c.classify_external_support(["contradicts"]) == "BROADER_EXTERNAL_DISAGREEMENT"
    assert c.classify_external_support(["supports", "contradicts"]) == "MIXED_EXTERNAL"
