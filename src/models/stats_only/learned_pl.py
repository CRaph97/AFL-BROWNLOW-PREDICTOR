"""
2027 candidate C -- LEARNED STATS-ONLY model (successor to the 2026 Objective
model's philosophy, with learned instead of hand-set weights).

Strict definition, enforced programmatically: only features whose registry
timing class is `same_match` (this match's own box score, this match's
player pool for relative transforms, this match's result). No role (a
cross-match proxy), no lagged form, no reputation, no team strength history,
no player identity. Linear Plackett-Luce on those features so every learned
weight is inspectable; a nonlinear variant (PerformanceRanker on the same
features) is evaluated separately as "stats_only_ml".
"""
from __future__ import annotations

from src.models.structural.pl_model import StructuralPL

FORBIDDEN_SUBSTRINGS = ("brownlow", "prev3", "prev5", "prev10", "prevN", "season_to_date", "role", "def_x", "ruck_x", "fwd_x",
                        "mid_x", "win_pct", "strength", "career", "has_prior", "last_season", "prior_seasons", "season_idx", "vs_prev", "vs_season")


def assert_stats_only(features: list[str]) -> None:
    bad = [f for f in features if any(s in f for s in FORBIDDEN_SUBSTRINGS)]
    assert not bad, f"stats-only model received forbidden features: {bad}"


class StatsOnlyPL(StructuralPL):
    family = "stats_only_pl"

    def __init__(self, features: list[str], l2: float = 1.0):
        assert_stats_only(features)
        super().__init__(features, l2=l2)
