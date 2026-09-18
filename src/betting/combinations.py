"""
Multi-leg combination candidates, priced from ACTUAL joint simulation draws
-- never by multiplying each leg's marginal probability, which would silently
assume independence and misprice any combination whose legs are correlated
(e.g. two players on the same team compete for the same match votes, so
"both finish top 10" is NOT independent of either leg alone; two mutually
exclusive legs, e.g. "Player A wins" and "Player B wins", have a TRUE joint
probability of exactly 0, which a naive product would never reveal).

Each leg here is a boolean per-simulated-season outcome vector (n_sims,) --
the joint probability of a combination is simply the fraction of simulated
seasons where every leg's vector is True, computed independently for
Production's and Objective's own simulation sets (their joint probabilities
are NOT combined with each other; each model prices its own joint outcome).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.betting.market_data import SimulationSet


@dataclass(frozen=True)
class Leg:
    label: str  # human-readable, e.g. "Nick Daicos to win"
    player_ids: tuple[str, ...]  # every player_id this leg's outcome depends on
    outcome_fn: callable  # SimulationSet -> np.ndarray[bool] of shape (n_sims,)
    confidence: str = ""
    wheelo_support: str = ""


@dataclass(frozen=True)
class CombinationResult:
    legs: list[Leg]
    production_joint_probability: float | None
    objective_joint_probability: float | None
    joint_model_gap: float | None
    rejected_reason: str | None = None
    correlated_legs_flag: bool = False


def leg_mask(sims: SimulationSet, leg: Leg) -> np.ndarray | None:
    for pid in leg.player_ids:
        if sims.col(pid) is None:
            return None
    return leg.outcome_fn(sims)


def _mutually_exclusive_conflict(legs: list[Leg], sims: SimulationSet) -> bool:
    """A combination is logically impossible if the joint mask is
    identically zero across every simulated season, AND every individual leg
    has positive probability on its own -- the second condition matters: if
    one leg is simply near-impossible by itself (e.g. a bench player's
    "top 10" leg with ~0% marginal probability), a zero joint is just that
    leg's own rarity, not a structural conflict between the two legs, and
    must not be reported as "logically conflicting" (that label is reserved
    for genuinely exclusive outcomes, e.g. two different players both
    claimed as the outright winner)."""
    masks = [leg_mask(sims, leg) for leg in legs]
    if any(m is None for m in masks):
        return False
    if any(not m.any() for m in masks):
        return False  # a leg with zero marginal probability, not a conflict
    joint = np.logical_and.reduce(masks)
    return not joint.any()


def _correlation_flag(legs: list[Leg], sims: SimulationSet, threshold: float = 0.6) -> bool:
    """Flags strongly correlated/redundant legs: if any two legs' outcome
    masks have a phi-coefficient (binary correlation) above `threshold`,
    they are carrying largely the same information (e.g. "Player X top 10"
    and "Player X top 5" are not independent legs worth combining)."""
    masks = [leg_mask(sims, leg) for leg in legs]
    masks = [m for m in masks if m is not None]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = masks[i].astype(float), masks[j].astype(float)
            if a.std() == 0 or b.std() == 0:
                continue
            corr = np.corrcoef(a, b)[0, 1]
            if abs(corr) >= threshold:
                return True
    return False


def price_combination(legs: list[Leg], production_sims: SimulationSet,
                       objective_sims: SimulationSet) -> CombinationResult:
    if len(legs) < 2:
        return CombinationResult(legs, None, None, None, rejected_reason="a combination needs at least 2 legs")

    for sims, label in ((production_sims, "Production"), (objective_sims, "Objective")):
        if _mutually_exclusive_conflict(legs, sims):
            return CombinationResult(
                legs, None, None, None,
                rejected_reason=f"logically conflicting legs -- {label}'s simulations show these outcomes never co-occur"
            )

    masks_prod = [leg_mask(production_sims, leg) for leg in legs]
    prod_joint = float(np.logical_and.reduce(masks_prod).mean()) if all(m is not None for m in masks_prod) else None

    masks_obj = [leg_mask(objective_sims, leg) for leg in legs]
    obj_joint = float(np.logical_and.reduce(masks_obj).mean()) if all(m is not None for m in masks_obj) else None

    gap = None
    if prod_joint is not None and obj_joint is not None:
        gap = abs(prod_joint - obj_joint)

    correlated = _correlation_flag(legs, production_sims) or _correlation_flag(legs, objective_sims)

    return CombinationResult(
        legs=legs,
        production_joint_probability=prod_joint,
        objective_joint_probability=obj_joint,
        joint_model_gap=gap,
        rejected_reason=None,
        correlated_legs_flag=correlated,
    )


def outcome_top_n(player_id: str, n: int) -> callable:
    def _fn(sims: SimulationSet) -> np.ndarray:
        # Uses src.betting.pricing's genuinely-memoised rank cache rather
        # than calling order_scenarios.ranks_from_totals() directly, which
        # recomputes a full (n_sims, n_players) double-argsort with NO
        # caching on every call. A combination search calls this per leg,
        # per candidate combination (potentially thousands of times for a
        # handful of legs) -- the uncached version stalled a real refresh
        # run for several minutes before being caught here.
        from src.betting.pricing import _rank_cache
        ranks = _rank_cache(sims)
        return ranks[:, sims.col(player_id)] <= n
    return _fn


def outcome_winner(player_id: str) -> callable:
    return outcome_top_n(player_id, 1)


def outcome_x_plus_votes(player_id: str, threshold: int) -> callable:
    def _fn(sims: SimulationSet) -> np.ndarray:
        return sims.totals[:, sims.col(player_id)] >= threshold
    return _fn


def outcome_team_votes_ou(team_players: tuple[str, ...], line: float, side: str) -> callable:
    def _fn(sims: SimulationSet) -> np.ndarray:
        cols = [sims.col(p) for p in team_players]
        team_totals = sims.totals[:, cols].sum(axis=1).astype(float)
        return team_totals > line if side == "over" else team_totals < line
    return _fn


def build_leg(label: str, player_id: str, kind: str, **kwargs) -> Leg:
    """Convenience constructor for the common leg kinds used by the
    Suggested Combinations UI -- `kind` in {"winner", "top_n",
    "x_plus_votes", "team_votes_ou"}. For "team_votes_ou", `player_id` is
    ignored and `kwargs["team_players"]` (a tuple of every player_id on the
    team) is used instead -- the leg's correlation/conflict checks then
    correctly span every one of those columns, not a single player's."""
    if kind == "winner":
        fn = outcome_winner(player_id)
        leg_player_ids = (player_id,)
    elif kind == "top_n":
        fn = outcome_top_n(player_id, kwargs["n"])
        leg_player_ids = (player_id,)
    elif kind == "x_plus_votes":
        fn = outcome_x_plus_votes(player_id, kwargs["threshold"])
        leg_player_ids = (player_id,)
    elif kind == "team_votes_ou":
        team_players = tuple(kwargs["team_players"])
        fn = outcome_team_votes_ou(team_players, kwargs["line"], kwargs["side"])
        leg_player_ids = team_players
    else:
        raise ValueError(f"unknown leg kind '{kind}'")
    return Leg(label=label, player_ids=leg_player_ids, outcome_fn=fn,
               confidence=kwargs.get("confidence", ""), wheelo_support=kwargs.get("wheelo_support", ""))
