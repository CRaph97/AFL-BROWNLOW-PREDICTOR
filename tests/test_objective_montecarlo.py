"""
Tests for the lightweight Objective Stats Model season simulation
(src/models/run_2026_objective_montecarlo.py) and the generic order-scenario
math (src/models/order_scenarios.py) that consumes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import order_scenarios as osc
from src.models.run_2026_objective_montecarlo import UTILITY_TEMPERATURE, simulate_season


@pytest.fixture(scope="module")
def objective_sim():
    df = pd.read_csv("reports/2026_objective_match_scores.csv")
    totals, players, n_matches = simulate_season(df, n_sims=4000, seed=1)
    return totals, players, n_matches


def test_season_total_votes_always_exactly_six_per_match(objective_sim):
    totals, players, n_matches = objective_sim
    per_sim_totals = totals.sum(axis=1)
    assert (per_sim_totals == 6 * n_matches).all()


def test_simulated_mean_matches_leaderboard_ev_within_monte_carlo_noise(objective_sim):
    """Regression test for the exact bug this simulation caught during
    development: using raw objective_score (temperature=1) instead of
    objective_score/UTILITY_TEMPERATURE over-concentrated the distribution
    and inflated leading players' simulated mean by several votes."""
    totals, players, _ = objective_sim
    leaderboard = pd.read_csv("reports/2026_objective_leaderboard.csv").set_index("player_name")["objective_ev"]

    top5 = players.assign(mean_votes=totals.mean(axis=0)).sort_values("mean_votes", ascending=False).head(5)
    for _, row in top5.iterrows():
        expected = leaderboard.get(row["player_name"])
        if expected is None:
            continue
        # 4,000-sim tolerance: generous but would clearly fail under the
        # temperature=1 bug (which was off by several votes, not noise).
        assert abs(row["mean_votes"] - expected) < 1.5, (
            f"{row['player_name']}: simulated {row['mean_votes']:.2f} vs leaderboard EV {expected:.2f}"
        )


def test_utility_temperature_is_documented_constant():
    assert UTILITY_TEMPERATURE == 15.0


def test_ranks_from_totals_are_a_valid_permutation_per_simulation():
    totals = np.array([[3, 1, 2], [0, 0, 5]])
    ranks = osc.ranks_from_totals(totals)
    for row in ranks:
        assert sorted(row.tolist()) == [1, 2, 3]


def test_top_exact_orders_probabilities_are_valid():
    rng = np.random.default_rng(0)
    n_sims, n_players = 2000, 6
    totals = rng.integers(0, 10, size=(n_sims, n_players))
    players = pd.DataFrame({
        "player_id": range(n_players),
        "player_name": [f"P{i}" for i in range(n_players)],
        "team_id": ["t"] * n_players,
    })
    top = osc.top_exact_orders(totals, players, depth=3, top_k=10)
    assert (top["probability"] > 0).all()
    assert (top["probability"] <= 1.0).all()
    assert top["probability"].is_monotonic_decreasing
    assert top["cumulative_probability"].max() <= 1.0 + 1e-9
    assert len(top) <= 10


def test_contender_probabilities_monotonic_in_threshold():
    rng = np.random.default_rng(0)
    totals = rng.integers(0, 10, size=(500, 8))
    players = pd.DataFrame({
        "player_id": range(8), "player_name": [f"P{i}" for i in range(8)], "team_id": ["t"] * 8,
    })
    out = osc.contender_probabilities(totals, players)
    for _, row in out.iterrows():
        assert row["prob_winner"] <= row["prob_top3"] + 1e-9
        assert row["prob_top3"] <= row["prob_top10"] + 1e-9
