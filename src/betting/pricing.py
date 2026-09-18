"""
Market-pricing engine: given a bookmaker selection (market type + the
player(s)/team/line it refers to), compute Production's and Objective's own
probability of that selection winning, from their real, already-persisted
Monte Carlo simulation draws (src/betting/market_data.py) -- never from EV
alone, never approximated, never fitted/retrained.

Every function here is a pure probability computation over an
(n_sims, n_players) integer-vote-total array. Ties in a simulated season are
broken with the exact same argsort-of-argsort convention as
src/models/order_scenarios.py (imported, not re-implemented) so results are
consistent with every other part of this project that ranks simulated
seasons.

Market coverage (see docs/BETTING_OPPORTUNITIES.md for the full rationale):
implemented -- WINNER, TOP_N, EXACT_POSITION, PLAYER_VOTES_OU, X_PLUS_VOTES,
TO_POLL_A_VOTE, PLAYER_H2H, GROUP_H2H, TEAM_TOP_POLLER, TEAM_VOTES_OU,
WINNING_VOTE_TOTAL_OU, EXACTA, QUINELLA, TRIFECTA. Deliberately UNMODELLED --
MOST_3_VOTE_GAMES (would require a persisted PER-MATCH simulation array;
mc_totals_2026.npy only carries season totals, not match-by-match vote
allocations per simulated season, so this cannot be honestly priced from
what is currently persisted without a new, separate simulation run outside
this task's scope).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.betting.market_data import SimulationSet
from src.models.order_scenarios import order_matrix, ranks_from_totals

UNMODELLED = "UNMODELLED"

# Market types this engine can honestly price. Anything not in this set must
# be reported as UNMODELLED by callers rather than guessed.
SUPPORTED_MARKET_TYPES = {
    "WINNER", "TOP_N", "EXACT_POSITION", "PLAYER_VOTES_OU", "X_PLUS_VOTES",
    "TO_POLL_A_VOTE", "PLAYER_H2H", "GROUP_H2H", "TEAM_TOP_POLLER",
    "TEAM_VOTES_OU", "WINNING_VOTE_TOTAL_OU", "EXACTA", "QUINELLA", "TRIFECTA",
}
KNOWN_UNMODELLED_MARKET_TYPES = {"MOST_3_VOTE_GAMES"}


@dataclass(frozen=True)
class PricingResult:
    probability: float | None  # None (never 0/NaN-as-zero) when unmodellable
    status: str  # "OK" or a reason string
    n_sims: int


_RANK_CACHE: dict[int, np.ndarray] = {}


def _rank_cache(sims: SimulationSet) -> np.ndarray:
    """Genuinely caches the (n_sims, n_players) rank matrix, keyed by the
    identity of sims.totals. Despite the name, this previously recomputed a
    full double-argsort over every player on EVERY call with no memoisation
    at all -- harmless for a handful of calls (tests, a few manual prices),
    but a real, discovered performance bug once a betting-refresh run prices
    hundreds of WINNER/TOP_N/EXACT_POSITION selections in one pass (each one
    a separate price_selection() call): it stalled a real pipeline run for
    8+ minutes before being caught. SimulationSet's own docstring guarantees
    its arrays are never mutated after being loaded, so caching by the
    totals array's object id is safe for the lifetime of a single process."""
    key = id(sims.totals)
    cached = _RANK_CACHE.get(key)
    if cached is None:
        cached = ranks_from_totals(sims.totals)
        _RANK_CACHE[key] = cached
    return cached


def price_winner(sims: SimulationSet, player_id: str) -> PricingResult:
    col = sims.col(player_id)
    if col is None:
        return PricingResult(None, "player not in simulation set", sims.n_sims)
    ranks = _rank_cache(sims)
    p = float((ranks[:, col] == 1).mean())
    return PricingResult(p, "OK", sims.n_sims)


def price_top_n(sims: SimulationSet, player_id: str, n: int) -> PricingResult:
    col = sims.col(player_id)
    if col is None:
        return PricingResult(None, "player not in simulation set", sims.n_sims)
    ranks = _rank_cache(sims)
    p = float((ranks[:, col] <= n).mean())
    return PricingResult(p, "OK", sims.n_sims)


def price_exact_position(sims: SimulationSet, player_id: str, position: int) -> PricingResult:
    col = sims.col(player_id)
    if col is None:
        return PricingResult(None, "player not in simulation set", sims.n_sims)
    ranks = _rank_cache(sims)
    p = float((ranks[:, col] == position).mean())
    return PricingResult(p, "OK", sims.n_sims)


def price_player_votes_ou(sims: SimulationSet, player_id: str, line: float, side: str) -> PricingResult:
    """side: 'over' or 'under'. Line is expected to be a bookmaker-style
    half-integer (e.g. 24.5) so no push/void case arises; if a whole-number
    line is passed, an exact-equal simulated total is excluded from both
    sides (a push), matching standard totals-market settlement."""
    col = sims.col(player_id)
    if col is None:
        return PricingResult(None, "player not in simulation set", sims.n_sims)
    vals = sims.totals[:, col].astype(float)
    if side == "over":
        p = float((vals > line).mean())
    elif side == "under":
        p = float((vals < line).mean())
    else:
        return PricingResult(None, f"unknown side '{side}'", sims.n_sims)
    return PricingResult(p, "OK", sims.n_sims)


def price_x_plus_votes(sims: SimulationSet, player_id: str, threshold: int) -> PricingResult:
    col = sims.col(player_id)
    if col is None:
        return PricingResult(None, "player not in simulation set", sims.n_sims)
    p = float((sims.totals[:, col] >= threshold).mean())
    return PricingResult(p, "OK", sims.n_sims)


def price_to_poll_a_vote(sims: SimulationSet, player_id: str) -> PricingResult:
    return price_x_plus_votes(sims, player_id, 1)


def price_player_h2h(sims: SimulationSet, player_a: str, player_b: str) -> dict:
    """Returns {'a_wins': p, 'b_wins': p, 'tie': p} -- an exact tie on
    simulated integer vote totals is a real, non-negligible outcome for this
    market (unlike a half-integer totals line) and bookmakers settle H2H
    markets on a tie via dead-heat rules or void, not by assigning it to
    either side, so it is surfaced explicitly rather than dropped."""
    col_a, col_b = sims.col(player_a), sims.col(player_b)
    if col_a is None or col_b is None:
        return {"a_wins": None, "b_wins": None, "tie": None, "status": "player not in simulation set"}
    a, b = sims.totals[:, col_a], sims.totals[:, col_b]
    n = sims.n_sims
    return {
        "a_wins": float((a > b).sum() / n),
        "b_wins": float((b > a).sum() / n),
        "tie": float((a == b).sum() / n),
        "status": "OK",
    }


def price_group_h2h(sims: SimulationSet, player_ids: list[str], target_player_id: str) -> PricingResult:
    """Probability that `target_player_id` finishes with the strictly
    highest simulated total among `player_ids` (a "group H2H" / "betting
    without the favourite"-style market). A multi-way tie for the group's
    top total is excluded from every player's probability (a push for that
    simulated season), matching PLAYER_H2H's tie handling."""
    cols = [sims.col(p) for p in player_ids]
    if target_player_id not in player_ids or any(c is None for c in cols):
        return PricingResult(None, "one or more players not in simulation set / target not in group", sims.n_sims)
    target_col = sims.col(target_player_id)
    group_vals = sims.totals[:, cols]
    group_max = group_vals.max(axis=1)
    is_unique_max = (group_vals == group_max[:, None]).sum(axis=1) == 1
    target_wins = (sims.totals[:, target_col] == group_max) & is_unique_max
    p = float(target_wins.mean())
    return PricingResult(p, "OK", sims.n_sims)


def price_team_top_poller(sims: SimulationSet, team_players: list[str], target_player_id: str) -> PricingResult:
    """Same computation as GROUP_H2H, restricted to one team's real roster
    -- kept as a separate named function per the brief's market taxonomy
    rather than aliased silently, since a caller building a TEAM_TOP_POLLER
    selection shouldn't need to know it shares GROUP_H2H's implementation."""
    return price_group_h2h(sims, team_players, target_player_id)


def price_team_votes_ou(sims: SimulationSet, team_players: list[str], line: float, side: str) -> PricingResult:
    cols = [sims.col(p) for p in team_players]
    if any(c is None for c in cols):
        return PricingResult(None, "one or more team players not in simulation set", sims.n_sims)
    if not cols:
        return PricingResult(None, "no players supplied for team", sims.n_sims)
    team_totals = sims.totals[:, cols].sum(axis=1).astype(float)
    if side == "over":
        p = float((team_totals > line).mean())
    elif side == "under":
        p = float((team_totals < line).mean())
    else:
        return PricingResult(None, f"unknown side '{side}'", sims.n_sims)
    return PricingResult(p, "OK", sims.n_sims)


def price_winning_vote_total_ou(sims: SimulationSet, line: float, side: str) -> PricingResult:
    """The eventual Brownlow WINNER's own vote total, over/under a line --
    independent of who the winner turns out to be."""
    winner_totals = sims.totals.max(axis=1).astype(float)
    if side == "over":
        p = float((winner_totals > line).mean())
    elif side == "under":
        p = float((winner_totals < line).mean())
    else:
        return PricingResult(None, f"unknown side '{side}'", sims.n_sims)
    return PricingResult(p, "OK", sims.n_sims)


def _exact_order_probability(sims: SimulationSet, player_ids: list[str]) -> PricingResult:
    """Shared machinery for EXACTA (len 2) / TRIFECTA (len 3): probability
    that the given players finish in EXACTLY this order, in the top
    len(player_ids) positions."""
    cols = [sims.col(p) for p in player_ids]
    if any(c is None for c in cols):
        return PricingResult(None, "one or more players not in simulation set", sims.n_sims)
    depth = len(player_ids)
    top = order_matrix(sims.totals, depth)  # (n_sims, depth) column-index order
    target = np.array(cols)
    matches = (top == target[None, :]).all(axis=1)
    p = float(matches.mean())
    return PricingResult(p, "OK", sims.n_sims)


def price_exacta(sims: SimulationSet, first: str, second: str) -> PricingResult:
    return _exact_order_probability(sims, [first, second])


def price_trifecta(sims: SimulationSet, first: str, second: str, third: str) -> PricingResult:
    return _exact_order_probability(sims, [first, second, third])


def price_quinella(sims: SimulationSet, player_a: str, player_b: str) -> PricingResult:
    """Probability that {player_a, player_b} occupy the top 2 finishing
    positions in EITHER order."""
    col_a, col_b = sims.col(player_a), sims.col(player_b)
    if col_a is None or col_b is None:
        return PricingResult(None, "one or more players not in simulation set", sims.n_sims)
    top2 = order_matrix(sims.totals, 2)
    pair = {col_a, col_b}
    matches = np.array([set(row) == pair for row in top2])
    p = float(matches.mean())
    return PricingResult(p, "OK", sims.n_sims)


def price_selection(sims: SimulationSet, market_type: str, **kwargs) -> PricingResult | dict:
    """Single dispatch entry point used by the refresh pipeline -- routes a
    normalised bookmaker selection to the right pricing function by market
    type, or returns an UNMODELLED result for anything not in
    SUPPORTED_MARKET_TYPES rather than guessing."""
    if market_type not in SUPPORTED_MARKET_TYPES:
        reason = "known unmodellable market type (no persisted match-level simulation data)" \
            if market_type in KNOWN_UNMODELLED_MARKET_TYPES else "unrecognised market type"
        return PricingResult(None, f"{UNMODELLED}: {reason}", sims.n_sims)

    dispatch = {
        "WINNER": lambda: price_winner(sims, kwargs["player_id"]),
        "TOP_N": lambda: price_top_n(sims, kwargs["player_id"], kwargs["n"]),
        "EXACT_POSITION": lambda: price_exact_position(sims, kwargs["player_id"], kwargs["position"]),
        "PLAYER_VOTES_OU": lambda: price_player_votes_ou(sims, kwargs["player_id"], kwargs["line"], kwargs["side"]),
        "X_PLUS_VOTES": lambda: price_x_plus_votes(sims, kwargs["player_id"], kwargs["threshold"]),
        "TO_POLL_A_VOTE": lambda: price_to_poll_a_vote(sims, kwargs["player_id"]),
        "PLAYER_H2H": lambda: price_player_h2h(sims, kwargs["player_a"], kwargs["player_b"]),
        "GROUP_H2H": lambda: price_group_h2h(sims, kwargs["player_ids"], kwargs["target_player_id"]),
        "TEAM_TOP_POLLER": lambda: price_team_top_poller(sims, kwargs["team_players"], kwargs["target_player_id"]),
        "TEAM_VOTES_OU": lambda: price_team_votes_ou(sims, kwargs["team_players"], kwargs["line"], kwargs["side"]),
        "WINNING_VOTE_TOTAL_OU": lambda: price_winning_vote_total_ou(sims, kwargs["line"], kwargs["side"]),
        "EXACTA": lambda: price_exacta(sims, kwargs["first"], kwargs["second"]),
        "QUINELLA": lambda: price_quinella(sims, kwargs["player_a"], kwargs["player_b"]),
        "TRIFECTA": lambda: price_trifecta(sims, kwargs["first"], kwargs["second"], kwargs["third"]),
    }
    return dispatch[market_type]()
