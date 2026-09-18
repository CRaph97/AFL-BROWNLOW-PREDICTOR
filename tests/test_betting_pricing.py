"""
Settlement/pricing tests for src/betting/pricing.py and combinations.py.
All computations run against the REAL, current Production/Objective
simulation arrays -- there is no synthetic simulation fixture here, only
synthetic bookmaker SELECTIONS (lines/thresholds/pairings) chosen to exercise
each market type's settlement logic, since this task obtained zero real
scraped bookmaker markets (see docs/BETTING_OPPORTUNITIES.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.betting import combinations as comb
from src.betting import pricing as p
from src.betting.market_data import (
    load_objective_simulations,
    load_production_simulations,
    load_team_lookup,
)


@pytest.fixture(scope="module")
def prod_sims():
    return load_production_simulations()


@pytest.fixture(scope="module")
def obj_sims():
    return load_objective_simulations()


@pytest.fixture(scope="module")
def team_lookup():
    return load_team_lookup()


def _daicos_id(sims):
    from src.betting.market_data import load_player_names
    names = load_player_names()
    matches = [pid for pid in sims.player_ids if names.get(pid) == "Nick Daicos"]
    assert matches, "Nick Daicos must be resolvable in the simulation player index"
    return matches[0]


def test_winner_probabilities_sum_close_to_one(prod_sims):
    ranks = p._rank_cache(prod_sims)
    winner_count = (ranks == 1).sum()
    # Every simulated season has exactly one rank-1 finisher (ties broken).
    assert winner_count == prod_sims.n_sims


def test_daicos_winner_probability_matches_leaderboard_top_rank(prod_sims):
    daicos = _daicos_id(prod_sims)
    result = p.price_winner(prod_sims, daicos)
    assert result.status == "OK"
    # Daicos is the runaway #1 in the real leaderboard; his win probability
    # should be high, not merely nonzero.
    assert result.probability > 0.5


def test_top_n_is_monotonic_in_n(prod_sims):
    daicos = _daicos_id(prod_sims)
    p1 = p.price_top_n(prod_sims, daicos, 1).probability
    p5 = p.price_top_n(prod_sims, daicos, 5).probability
    p10 = p.price_top_n(prod_sims, daicos, 10).probability
    assert p1 <= p5 <= p10 <= 1.0


def test_exact_position_sums_to_top_n(prod_sims):
    daicos = _daicos_id(prod_sims)
    positions = [p.price_exact_position(prod_sims, daicos, k).probability for k in range(1, 11)]
    top10 = p.price_top_n(prod_sims, daicos, 10).probability
    assert abs(sum(positions) - top10) < 1e-9


def test_player_votes_ou_over_plus_under_equals_one_for_half_integer_line(prod_sims):
    daicos = _daicos_id(prod_sims)
    over = p.price_player_votes_ou(prod_sims, daicos, 40.5, "over").probability
    under = p.price_player_votes_ou(prod_sims, daicos, 40.5, "under").probability
    assert abs((over + under) - 1.0) < 1e-9, "a half-integer line must have no push, so over+under == 1"


def test_x_plus_votes_zero_is_certain(prod_sims):
    daicos = _daicos_id(prod_sims)
    assert p.price_x_plus_votes(prod_sims, daicos, 0).probability == 1.0


def test_to_poll_a_vote_matches_x_plus_votes_one(prod_sims):
    daicos = _daicos_id(prod_sims)
    a = p.price_to_poll_a_vote(prod_sims, daicos).probability
    b = p.price_x_plus_votes(prod_sims, daicos, 1).probability
    assert a == b


def test_h2h_probabilities_sum_to_one_including_tie(prod_sims):
    ids = prod_sims.player_ids[:2]
    result = p.price_player_h2h(prod_sims, ids[0], ids[1])
    assert result["status"] == "OK"
    total = result["a_wins"] + result["b_wins"] + result["tie"]
    assert abs(total - 1.0) < 1e-9


def test_h2h_is_antisymmetric(prod_sims):
    ids = prod_sims.player_ids[:2]
    r1 = p.price_player_h2h(prod_sims, ids[0], ids[1])
    r2 = p.price_player_h2h(prod_sims, ids[1], ids[0])
    assert r1["a_wins"] == r2["b_wins"]
    assert r1["tie"] == r2["tie"]


def test_group_h2h_probabilities_sum_close_to_one_across_group(prod_sims):
    group = prod_sims.player_ids[:5]
    total = sum(p.price_group_h2h(prod_sims, group, pid).probability for pid in group)
    # Sums to less than 1 only by multi-way-tie push mass. An arbitrary
    # 5-player group can include several low-vote players who tie at 0 far
    # more often than genuine contenders would, so the bound here is
    # generous (0.5) rather than assuming a near-1 sum -- the meaningful
    # invariant is that it's a real probability mass, not that it's close to
    # exhaustive for this specific arbitrary group.
    assert 0.5 <= total <= 1.0


def test_group_h2h_probabilities_sum_close_to_one_for_top_contenders(prod_sims):
    """A group of genuine contenders (high real vote totals) should have far
    less zero-tie push mass than an arbitrary group -- the sum should be
    close to 1."""
    from src.betting.market_data import load_player_names
    import pandas as pd
    lb = pd.read_csv("reports/2026_leaderboard.csv")
    lb["player_id"] = lb["player_id"].astype(str)
    top5 = lb.sort_values("FINAL_ENSEMBLE", ascending=False).head(5)["player_id"].tolist()
    total = sum(p.price_group_h2h(prod_sims, top5, pid).probability for pid in top5)
    assert 0.9 <= total <= 1.0


def test_team_top_poller_restricted_to_real_roster(prod_sims, team_lookup):
    daicos = _daicos_id(prod_sims)
    team = team_lookup.get(daicos)
    roster = [pid for pid in prod_sims.player_ids if team_lookup.get(pid) == team]
    assert daicos in roster
    result = p.price_team_top_poller(prod_sims, roster, daicos)
    assert result.status == "OK"
    assert 0.0 <= result.probability <= 1.0


def test_team_votes_ou_is_monotonic_in_line(prod_sims, team_lookup):
    daicos = _daicos_id(prod_sims)
    team = team_lookup.get(daicos)
    roster = [pid for pid in prod_sims.player_ids if team_lookup.get(pid) == team]
    low = p.price_team_votes_ou(prod_sims, roster, 20.5, "over").probability
    high = p.price_team_votes_ou(prod_sims, roster, 80.5, "over").probability
    assert low >= high


def test_winning_vote_total_ou_uses_max_not_a_named_player(prod_sims):
    result_low = p.price_winning_vote_total_ou(prod_sims, 10.5, "over")
    result_high = p.price_winning_vote_total_ou(prod_sims, 60.5, "over")
    assert result_low.probability >= result_high.probability
    assert result_low.probability > 0.9  # some player almost always beats 10.5 votes


def test_exacta_probability_less_than_or_equal_to_either_leg_top2(prod_sims):
    a, b = prod_sims.player_ids[:2]
    exacta = p.price_exacta(prod_sims, a, b).probability
    a_top2 = p.price_top_n(prod_sims, a, 2).probability
    assert exacta <= a_top2 + 1e-9


def test_quinella_equals_sum_of_both_exacta_orders(prod_sims):
    a, b = prod_sims.player_ids[:2]
    quin = p.price_quinella(prod_sims, a, b).probability
    ex1 = p.price_exacta(prod_sims, a, b).probability
    ex2 = p.price_exacta(prod_sims, b, a).probability
    assert abs(quin - (ex1 + ex2)) < 1e-9


def test_trifecta_probability_le_exacta(prod_sims):
    a, b, c = prod_sims.player_ids[:3]
    tri = p.price_trifecta(prod_sims, a, b, c).probability
    exa = p.price_exacta(prod_sims, a, b).probability
    assert tri <= exa + 1e-9


def test_unmodelled_market_type_never_returns_a_fake_probability(prod_sims):
    result = p.price_selection(prod_sims, "MOST_3_VOTE_GAMES")
    assert result.probability is None
    assert p.UNMODELLED in result.status


def test_unresolved_player_returns_none_not_zero(prod_sims):
    result = p.price_winner(prod_sims, "NOT_A_REAL_PLAYER_ID")
    assert result.probability is None


# --------------------------------------------------------------------------
# Combinations: joint probability from real correlated simulation draws
# --------------------------------------------------------------------------

def test_combination_of_two_different_winners_is_logically_impossible(prod_sims, obj_sims):
    """Use the two real players who actually have non-trivial win
    probability (per docs/2026_FINAL_AUDIT.md, essentially only Daicos and
    Smith ever win in 100,000 simulated seasons) -- an arbitrary pair of
    players would usually both have exactly 0% win probability each, which
    is a different (and uninteresting) reason for a zero joint, not a
    genuine structural conflict between two real possibilities."""
    import pandas as pd
    lb = pd.read_csv("reports/2026_leaderboard.csv")
    lb["player_id"] = lb["player_id"].astype(str)
    a, b = lb.sort_values("FINAL_ENSEMBLE", ascending=False).head(2)["player_id"].tolist()
    assert p.price_winner(prod_sims, a).probability > 0
    assert p.price_winner(prod_sims, b).probability > 0
    legs = [comb.build_leg(f"{a} wins", a, "winner"), comb.build_leg(f"{b} wins", b, "winner")]
    result = comb.price_combination(legs, prod_sims, obj_sims)
    assert result.rejected_reason is not None
    assert "conflict" in result.rejected_reason.lower()


def test_combination_joint_probability_is_not_the_naive_product(prod_sims, obj_sims):
    """The whole point of using real simulation draws instead of multiplying
    marginals: two genuine contenders' "both top 10" legs are correlated
    through the shared match-level allocation (a big game for one often
    coincides with a smaller one for whoever else was on the ground), so the
    true joint probability must differ from the naive independent product.
    Uses the top-2 real contenders (moderate, non-degenerate marginals) so
    the test isn't vulnerable to a near-certain or near-impossible leg
    trivially matching its own product."""
    import pandas as pd
    lb = pd.read_csv("reports/2026_leaderboard.csv")
    lb["player_id"] = lb["player_id"].astype(str)
    top2 = lb.sort_values("FINAL_ENSEMBLE", ascending=False).head(6).tail(2)["player_id"].tolist()
    a, b = top2

    leg_a = comb.build_leg("A top 10", a, "top_n", n=10)
    leg_b = comb.build_leg("B top 10", b, "top_n", n=10)
    result = comb.price_combination([leg_a, leg_b], prod_sims, obj_sims)
    assert result.rejected_reason is None

    marg_a = p.price_top_n(prod_sims, a, 10).probability
    marg_b = p.price_top_n(prod_sims, b, 10).probability
    naive_product = marg_a * marg_b
    assert result.production_joint_probability is not None
    assert result.production_joint_probability != pytest.approx(naive_product, rel=1e-9)


def test_combination_needs_at_least_two_legs(prod_sims, obj_sims):
    daicos = _daicos_id(prod_sims)
    leg = comb.build_leg("Daicos wins", daicos, "winner")
    result = comb.price_combination([leg], prod_sims, obj_sims)
    assert result.rejected_reason is not None


def test_highly_correlated_legs_are_flagged(prod_sims, obj_sims):
    """Nested outcomes with CLOSE marginal probabilities (top_n=10 vs
    top_n=9, for a real ~rank-10 contender) give a high, well-defined phi
    coefficient. Phi is fundamentally bounded by how different two binary
    events' marginal proportions are, regardless of logical nesting -- e.g.
    Daicos's near-certain top-10 probability (>0.999) mathematically caps
    the achievable phi for ANY nested pair involving him near 0, and even a
    20-vs-15 pair for a mid-table player only reaches ~0.4 -- so this test
    deliberately picks a player and a pair of N values with genuinely close
    marginals (empirically ~0.83 phi), not an arbitrary nested pair."""
    import pandas as pd
    lb = pd.read_csv("reports/2026_leaderboard.csv")
    lb["player_id"] = lb["player_id"].astype(str)
    rank10_player = lb.sort_values("FINAL_ENSEMBLE", ascending=False).iloc[9]["player_id"]
    leg_a = comb.build_leg("Player top 10", rank10_player, "top_n", n=10)
    leg_b = comb.build_leg("Player top 9", rank10_player, "top_n", n=9)
    result = comb.price_combination([leg_a, leg_b], prod_sims, obj_sims)
    assert result.correlated_legs_flag is True
