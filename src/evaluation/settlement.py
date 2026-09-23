"""
Pure settlement + flat-stake P/L logic for the 2026 post-Brownlow betting
evaluation. No I/O, no probabilities computed here -- only "given the actual
outcome, did this priced selection win, lose, push or dead-heat, and what is
the 1-unit flat-stake result". Every rule is explicit so it can be unit
tested (tests/test_2026_evaluation.py) and audited.

Conventions
- `odds` are decimal (bookmaker-specific, exactly as captured pre-count).
- Flat 1-unit stake, retrospective analytical metric only: win -> odds - 1,
  loss -> -1, push/void -> 0, dead heat -> odds * fraction - 1 where
  fraction = places remaining / runners tied (standard dead-heat reduction).
- Ranks use the "min" method over ALL vote-getters INCLUDING ineligible
  players (bookmakers' own "Includes Ineligible" wording; the AFL leaderboard
  lists ineligible players with their votes). Callers may pass an
  eligible-only rank to flag rows whose settlement would change.
- Nothing is inferred: a selection whose player identity or actual value is
  unavailable is "unsettleable", never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WIN, LOSS, PUSH, DEAD_HEAT, UNSETTLEABLE = "win", "loss", "push", "dead_heat", "unsettleable"


@dataclass(frozen=True)
class Settlement:
    result: str            # win | loss | push | dead_heat | unsettleable
    fraction: float        # share of stake paid at full odds (1.0 win, 0 loss/push, (0,1) dead heat)
    detail: str


def _ok(v) -> bool:
    """True for a real value; False for None / NaN / pandas NA."""
    if v is None:
        return False
    try:
        return not bool(pd.isna(v))
    except (TypeError, ValueError):
        return True


def settle_top_n(rank: int | None, tied: int | None, n: int) -> Settlement:
    """Finish in the top n (any order). rank = min-method rank, tied = number
    of players sharing that rank (1 = outright)."""
    if not _ok(rank) or not _ok(tied) or not _ok(n):
        return Settlement(UNSETTLEABLE, 0.0, "missing rank")
    rank, tied, n = int(rank), int(tied), int(n)
    if rank > n:
        return Settlement(LOSS, 0.0, f"rank {rank} > {n}")
    if rank + tied - 1 <= n:
        return Settlement(WIN, 1.0, f"rank {rank} within top {n}")
    places = n - rank + 1
    return Settlement(DEAD_HEAT, places / tied, f"{tied} tied at rank {rank} for {places} place(s)")


def settle_winner(rank, tied) -> Settlement:
    return settle_top_n(rank, tied, 1)


def settle_exact_position(rank, tied, position: int) -> Settlement:
    if not _ok(rank) or not _ok(tied) or not _ok(position):
        return Settlement(UNSETTLEABLE, 0.0, "missing rank")
    rank, tied, position = int(rank), int(tied), int(position)
    if rank <= position <= rank + tied - 1:
        if tied == 1:
            return Settlement(WIN, 1.0, f"finished {position} outright")
        return Settlement(DEAD_HEAT, 1.0 / tied, f"{tied} tied covering position {position}")
    return Settlement(LOSS, 0.0, f"rank {rank}, not {position}")


def settle_over_under(actual: float | None, line: float | None, side: str | None) -> Settlement:
    if not _ok(actual) or not _ok(line) or side not in ("over", "under"):
        return Settlement(UNSETTLEABLE, 0.0, "missing actual/line/side")
    if actual == line:
        return Settlement(PUSH, 0.0, f"actual {actual} == line {line}")
    won = actual > line if side == "over" else actual < line
    return Settlement(WIN if won else LOSS, 1.0 if won else 0.0, f"actual {actual} vs line {line} ({side})")


def settle_threshold(actual: float | None, threshold: float | None) -> Settlement:
    """X+ votes / to poll a vote (threshold 1): actual >= threshold wins."""
    if not _ok(actual) or not _ok(threshold):
        return Settlement(UNSETTLEABLE, 0.0, "missing actual/threshold")
    won = actual >= threshold
    return Settlement(WIN if won else LOSS, 1.0 if won else 0.0, f"actual {actual} vs {threshold}+")


def settle_h2h(own: float | None, opponent: float | None) -> Settlement:
    if not _ok(own) or not _ok(opponent):
        return Settlement(UNSETTLEABLE, 0.0, "missing a side's actual votes")
    if own == opponent:
        return Settlement(PUSH, 0.0, f"tie {own}-{opponent}")
    won = own > opponent
    return Settlement(WIN if won else LOSS, 1.0 if won else 0.0, f"{own} vs {opponent}")


def flat_unit_pnl(settlement: Settlement, odds: float | None) -> float | None:
    """1-unit flat stake profit/loss. None when the row cannot be settled or
    has no captured price (never inferred)."""
    if settlement.result == UNSETTLEABLE or not _ok(odds):
        return None
    if settlement.result == LOSS:
        return -1.0
    if settlement.result == PUSH:
        return 0.0
    return float(odds) * settlement.fraction - 1.0


def implied_probability(odds: float | None) -> float | None:
    return 1.0 / float(odds) if _ok(odds) and float(odds) > 0 else None


def summarise_bets(pnl: list[float | None], results: list[str]) -> dict:
    """bets / wins / pushes / hit rate / P&L / ROI for a group. Hit rate
    excludes pushes from the denominator; ROI denominator is every settled
    bet (1 unit staked each, pushes included since the stake was risked)."""
    settled = [(p, r) for p, r in zip(pnl, results) if p is not None and r != UNSETTLEABLE]
    n = len(settled)
    wins = sum(1 for _, r in settled if r in (WIN, DEAD_HEAT))
    pushes = sum(1 for _, r in settled if r == PUSH)
    total = sum(p for p, _ in settled)
    return {
        "bets": n, "wins": wins, "pushes": pushes, "losses": n - wins - pushes,
        "hit_rate": (wins / (n - pushes)) if n - pushes > 0 else None,
        "flat_unit_pnl": round(total, 4) if n else None,
        "roi": round(total / n, 4) if n else None,
    }
