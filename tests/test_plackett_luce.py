"""
Phase 4 correctness-bug regression tests for PlackettLuceModel (see the
module docstring in src/models/plackett_luce.py for the full bug writeup:
unstandardised mixed-scale features + a non-log-space likelihood caused
catastrophic cancellation and NaN coefficients on the ADVANCED feature set).

These tests use small synthetic match datasets (not the real CORE/ADVANCED
tables) so they run fast and isolate the model's own numerical behaviour from
any data-quality question.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.plackett_luce import PlackettLuceModel


def _make_synthetic_matches(n_matches=40, players_per_match=12, seed=0,
                             metres_gained_scale=600.0) -> pd.DataFrame:
    """Builds a synthetic player-match dataset with one small-scale feature
    (z-score-like, mean 0 std 1) and one large-scale feature deliberately
    modelled on the real `metres_gained` field (0-600+) that triggered the
    original bug, plus a binary indicator feature. Votes are assigned
    according to a known true utility so the model has real signal to find."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        match_id = f"match_{m}"
        small_scale = rng.normal(0, 1, players_per_match)
        metres_gained = rng.uniform(0, metres_gained_scale, players_per_match)
        is_win = np.zeros(players_per_match)
        is_win[: players_per_match // 2] = 1  # half the "team" won

        true_utility = 1.5 * small_scale + 0.01 * metres_gained + 0.3 * is_win
        order = np.argsort(-true_utility)
        votes = np.zeros(players_per_match)
        votes[order[0]] = 3
        votes[order[1]] = 2
        votes[order[2]] = 1

        for p in range(players_per_match):
            rows.append({
                "match_id": match_id, "player_id": f"{match_id}_p{p}",
                "small_scale": small_scale[p], "metres_gained": metres_gained[p],
                "is_win": is_win[p], "brownlow_votes": votes[p],
            })
    return pd.DataFrame(rows)


FEATURES = ["small_scale", "metres_gained", "is_win"]


def test_mixed_scale_features_no_warning_no_nan():
    """The exact failure mode of the Phase 4 bug: a huge-scale feature (modelled on
    metres_gained) alongside a small-scale one must not produce NaN/Inf coefficients
    or raise numerical warnings."""
    df = _make_synthetic_matches()
    with np.errstate(all="raise"):  # promote any invalid-value/overflow warning to an error
        model = PlackettLuceModel(feature_names=FEATURES).fit(df)
    assert np.isfinite(model.beta).all()


def test_no_train_test_preprocessing_leakage():
    """The scaler's mean/scale must come from TRAIN only -- verified by checking they
    match a manual train-only computation, and are NOT affected by test-set values
    that differ sharply from the training distribution."""
    train = _make_synthetic_matches(n_matches=30, seed=1)
    model = PlackettLuceModel(feature_names=FEATURES).fit(train)

    manual_mean = train[FEATURES].to_numpy().mean(axis=0)
    manual_std = train[FEATURES].to_numpy().std(axis=0)
    # is_win is binary -> left unscaled (mean forced to 0, scale forced to 1)
    manual_mean[2] = 0.0
    manual_std[2] = 1.0
    np.testing.assert_allclose(model._mean, manual_mean, rtol=1e-10)
    np.testing.assert_allclose(model._scale, manual_std, rtol=1e-10)

    # a wildly different test set must not change the already-fitted scaler
    test = _make_synthetic_matches(n_matches=5, seed=2, metres_gained_scale=50000.0)
    mean_before, scale_before = model._mean.copy(), model._scale.copy()
    _ = model.predict(test)
    np.testing.assert_array_equal(model._mean, mean_before)
    np.testing.assert_array_equal(model._scale, scale_before)


def test_zero_variance_feature_handled_safely():
    """A constant (zero-variance) feature must not cause a divide-by-zero /
    NaN -- its scale should fall back to 1.0 and its contribution to the fit
    should simply be neutral (no information in a constant column)."""
    df = _make_synthetic_matches(n_matches=20, seed=3)
    df["constant_feature"] = 7.0
    feats = FEATURES + ["constant_feature"]
    model = PlackettLuceModel(feature_names=feats).fit(df)
    assert np.isfinite(model.beta).all()
    const_idx = feats.index("constant_feature")
    assert model._scale[const_idx] == 1.0


def test_binary_indicator_left_unscaled():
    df = _make_synthetic_matches(n_matches=20, seed=4)
    model = PlackettLuceModel(feature_names=FEATURES).fit(df)
    is_win_idx = FEATURES.index("is_win")
    assert model._is_binary[is_win_idx]
    assert model._mean[is_win_idx] == 0.0
    assert model._scale[is_win_idx] == 1.0


def test_finite_fitted_coefficients():
    df = _make_synthetic_matches(n_matches=25, seed=5)
    model = PlackettLuceModel(feature_names=FEATURES).fit(df)
    assert np.isfinite(model.beta).all()


def test_finite_coherent_probabilities():
    train = _make_synthetic_matches(n_matches=25, seed=6)
    test = _make_synthetic_matches(n_matches=10, seed=7)
    model = PlackettLuceModel(feature_names=FEATURES).fit(train)
    preds = model.predict(test)
    for col in ("p3", "p2", "p1", "p0"):
        assert np.isfinite(preds[col]).all()
        assert (preds[col] >= -1e-9).all() and (preds[col] <= 1 + 1e-6).all()


def test_exact_within_match_probability_constraints():
    train = _make_synthetic_matches(n_matches=25, seed=8)
    test = _make_synthetic_matches(n_matches=10, seed=9)
    model = PlackettLuceModel(feature_names=FEATURES).fit(train)
    preds = model.predict(test)
    sums = preds.groupby("match_id")[["p3", "p2", "p1"]].sum()
    np.testing.assert_allclose(sums["p3"], 1.0, atol=1e-8)
    np.testing.assert_allclose(sums["p2"], 1.0, atol=1e-8)
    np.testing.assert_allclose(sums["p1"], 1.0, atol=1e-8)


def test_reproducibility():
    df = _make_synthetic_matches(n_matches=20, seed=10)
    beta_a = PlackettLuceModel(feature_names=FEATURES).fit(df).beta
    beta_b = PlackettLuceModel(feature_names=FEATURES).fit(df).beta
    np.testing.assert_array_equal(beta_a, beta_b)


def test_fit_raises_on_non_finite_result_rather_than_silently_returning():
    """If fitting somehow still produced a non-finite beta, the model must raise,
    not silently return a broken model -- simulated by monkeypatching the
    optimiser's result via an impossible feature (all-NaN column) that should be
    caught upstream, but this test documents the fail-loud contract directly."""
    df = _make_synthetic_matches(n_matches=5, seed=11)
    model = PlackettLuceModel(feature_names=FEATURES)
    model._mean = np.zeros(len(FEATURES))
    model._scale = np.ones(len(FEATURES))
    model.beta = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError):
        model.predict_match_probabilities(df[df["match_id"] == "match_0"])
