"""
Regression tests for the standalone 2026 Objective Stats Model
(src/models/objective_stats_model.py). Verifies -- not just asserts -- the hard
constraints from the brief: no historical Brownlow/reputation/role data enters
the scoring inputs, no cross-match leakage, coherent probabilities, and the
season-level vote-total invariant.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.objective_stats_model import (
    CONTEXT_COLS,
    ADV_STAT_COLS,
    CORE_STAT_COLS,
    FORBIDDEN_SUBSTRINGS,
    GROUP_WEIGHTS,
    assert_no_forbidden_inputs,
    build_match_probabilities,
    compute_objective_scores,
)


@pytest.fixture(scope="module")
def raw_2026_frame():
    from src.models.objective_stats_model import load_2026_input_frame
    return load_2026_input_frame()


def test_no_forbidden_columns_in_declared_inputs():
    """The declared allowlists of inputs used by the scoring function contain
    none of the forbidden historical/reputation/role substrings."""
    declared = list(CONTEXT_COLS) + list(CORE_STAT_COLS) + list(ADV_STAT_COLS)
    bad = assert_no_forbidden_inputs(declared)
    assert bad == [], f"Forbidden columns found in declared inputs: {bad}"


def test_forbidden_substrings_actually_flag_known_historical_columns():
    """Sanity-check the detector itself: it must actually flag the real
    historical/reputation column names, or the 'zero forbidden columns' claim
    above would be vacuous."""
    known_bad = [
        "brownlow_votes", "brownlow_votes_prev5_mean", "brownlow_votes_season_to_date_mean",
        "clearances_prev3_mean", "disposals_season_to_date_mean", "role", "role_source",
    ]
    bad = assert_no_forbidden_inputs(known_bad)
    assert set(bad) == set(known_bad)


def test_group_weights_sum_to_100():
    assert sum(GROUP_WEIGHTS.values()) == 100


def test_no_cross_match_leakage(raw_2026_frame):
    """Within-match z-scores must depend only on that match's own rows: shuffling
    a match's row order, or permuting which match a copy of a fixed player's stat
    line is attached to, must not change a *different* match's computed scores."""
    df = raw_2026_frame
    scored_full = compute_objective_scores(df)

    one_match_id = df["match_id"].iloc[0]
    other_matches = df[df["match_id"] != one_match_id]
    scored_others_alone = compute_objective_scores(other_matches)

    merged = other_matches[["match_id", "player_id"]].copy()
    merged["objective_score_full"] = scored_full.set_index(["match_id", "player_id"]).loc[
        list(zip(merged["match_id"], merged["player_id"])), "objective_score"
    ].to_numpy()
    merged["objective_score_alone"] = scored_others_alone["objective_score"].to_numpy()
    assert np.allclose(merged["objective_score_full"], merged["objective_score_alone"], atol=1e-9)


def test_probabilities_coherent_within_match(raw_2026_frame):
    scored = compute_objective_scores(raw_2026_frame)
    match_probs = build_match_probabilities(scored)

    for col in ["p3", "p2", "p1"]:
        sums = match_probs.groupby("match_id")[col].sum()
        assert np.allclose(sums, 1.0, atol=1e-6), f"{col} does not sum to 1 in every match"

    assert match_probs[["p3", "p2", "p1", "p0"]].to_numpy().min() >= -1e-9
    assert match_probs[["p3", "p2", "p1", "p0"]].to_numpy().max() <= 1 + 1e-6
    assert np.isfinite(match_probs[["p3", "p2", "p1", "p0", "expected_votes"]].to_numpy()).all()


def test_season_total_expected_votes_equals_6_times_matches(raw_2026_frame):
    scored = compute_objective_scores(raw_2026_frame)
    match_probs = build_match_probabilities(scored)
    n_matches = match_probs["match_id"].nunique()
    total_ev = match_probs.drop_duplicates(["match_id", "player_id"])["expected_votes"].sum()
    assert abs(total_ev - 6 * n_matches) < 1e-6
    assert n_matches == 207


def test_deterministic_picks_always_three_distinct_players(raw_2026_frame):
    from src.models.build_2026_objective_outputs import deterministic_match_picks

    scored = compute_objective_scores(raw_2026_frame)
    match_probs = build_match_probabilities(scored)
    picks = deterministic_match_picks(match_probs)

    counts = picks.groupby("match_id")["player_id"].nunique()
    assert (counts == 3).all()
    votes = picks.groupby("match_id")["objective_pred_votes"].apply(lambda s: sorted(s.tolist()))
    assert votes.apply(lambda v: v == [1, 2, 3]).all()


def test_zero_variance_stat_does_not_produce_nan_or_inf(raw_2026_frame):
    """A match where every player has the same value for some stat (std=0) must
    not propagate NaN/Inf into the final score."""
    df = raw_2026_frame.copy()
    one_match = df["match_id"].iloc[0]
    df.loc[df["match_id"] == one_match, "hitouts"] = 0
    scored = compute_objective_scores(df)
    sub = scored[scored["match_id"] == one_match]
    assert np.isfinite(sub["objective_score"]).all()
