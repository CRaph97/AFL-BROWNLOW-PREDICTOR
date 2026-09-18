"""
Confidence classification for a priced betting selection: evidence
convergence, never certainty. Thresholds below are fixed by the project
brief BEFORE looking at any real market data (this task obtained zero real
scraped bookmaker rows -- see docs/BETTING_OPPORTUNITIES.md -- so these
thresholds could not have been, and were not, tuned against results).

Inputs are already-computed probabilities/edges (from src/betting/pricing.py
+ implied probability from real bookmaker odds) and Wheelo/external evidence
(from src/external/aggregate.py) -- this module does no probability
computation of its own, only classification logic.
"""
from __future__ import annotations

from dataclasses import dataclass

INTERNAL_GAP_HIGH_CONFIDENCE_PP = 7.5  # brief section "CONFIDENCE CLASSIFICATION"

# Agreement/disagreement thresholds for Wheelo/external support labels,
# expressed in the same units as internal_gap (percentage points of
# probability, or EV votes for EV-type comparisons) -- reuses
# src.external.aggregate's already-fixed AGREEMENT thresholds for
# EV-type Wheelo comparisons so this module doesn't invent a second,
# inconsistent definition of "agree".
from src.external.aggregate import AGREEMENT_ABS_THRESHOLD, AGREEMENT_REL_THRESHOLD


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: str
    wheelo_support: str
    external_support: str
    outlier_model: str | None
    rationale: str


def _agree_ev(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None:
        return None
    if abs(a - b) <= AGREEMENT_ABS_THRESHOLD:
        return True
    denom = (abs(a) + abs(b)) / 2 or 1.0
    return abs(a - b) / denom <= AGREEMENT_REL_THRESHOLD


def classify_wheelo_support(production_direction_positive: bool, wheelo_ev: float | None,
                             production_ev: float | None, objective_ev: float | None) -> str:
    """direction_positive: whether OUR models see this selection as good
    value (used to determine whether Wheelo "agreeing" means a high or low
    Wheelo number, direction-appropriately)."""
    if wheelo_ev is None:
        return "INSUFFICIENT_WHEELO_DATA"
    our_relevant = production_ev if production_ev is not None else objective_ev
    if our_relevant is None:
        return "INSUFFICIENT_WHEELO_DATA"
    agrees = _agree_ev(wheelo_ev, our_relevant)
    if agrees:
        return "STRONG_WHEELO_SUPPORT"
    # Materially higher/lower than our number, same direction as our claim,
    # but outside the strict "agree" band -> partial; opposite direction
    # entirely -> disagrees.
    same_side = (wheelo_ev >= our_relevant) == production_direction_positive or wheelo_ev == our_relevant
    if same_side:
        return "PARTIAL_WHEELO_SUPPORT"
    gap_material = abs(wheelo_ev - our_relevant) > max(AGREEMENT_ABS_THRESHOLD, 0.3 * our_relevant if our_relevant else 0)
    return "WHEELO_DISAGREES" if gap_material else "WHEELO_NEUTRAL"


def classify_external_support(labels: list[str]) -> str:
    """labels: the individual context-source (ESPN/AFL/Betfair) directional
    verdicts already computed by the caller ('supports' / 'neutral' /
    'contradicts'), never re-derived here from raw values -- context sources
    are explicitly NOT quantitatively comparable per the project brief, so
    this only aggregates qualitative verdicts, never numbers."""
    if not labels:
        return "INSUFFICIENT_DATA"
    supports = labels.count("supports")
    contradicts = labels.count("contradicts")
    if supports and not contradicts:
        return "BROADER_EXTERNAL_SUPPORT"
    if contradicts and not supports:
        return "BROADER_EXTERNAL_DISAGREEMENT"
    if supports and contradicts:
        return "MIXED_EXTERNAL"
    return "INSUFFICIENT_DATA"


def classify_confidence(
    production_edge_pp: float | None,
    objective_edge_pp: float | None,
    internal_gap_pp: float | None,
    wheelo_support: str,
    settlement_flag: str | None,
) -> ConfidenceResult:
    """Implements the exact taxonomy in the project brief. `settlement_flag`
    being anything other than None (e.g. PRICE_SUSPECT, IDENTITY_AMBIGUOUS,
    SETTLEMENT_REVIEW_REQUIRED) forces NO_VALUE regardless of edges, per the
    brief's "flagged rows must never appear in High/Medium headline
    opportunities" hard rule."""
    both_positive = (production_edge_pp is not None and production_edge_pp > 0
                      and objective_edge_pp is not None and objective_edge_pp > 0)
    only_one_positive = (
        (production_edge_pp is not None and production_edge_pp > 0)
        != (objective_edge_pp is not None and objective_edge_pp > 0)
    ) if production_edge_pp is not None and objective_edge_pp is not None else False

    if settlement_flag is not None:
        return ConfidenceResult("NO_VALUE", wheelo_support, "N/A", None,
                                 f"suppressed: {settlement_flag}")

    if both_positive and internal_gap_pp is not None and internal_gap_pp <= INTERNAL_GAP_HIGH_CONFIDENCE_PP:
        if wheelo_support in ("STRONG_WHEELO_SUPPORT", "PARTIAL_WHEELO_SUPPORT"):
            return ConfidenceResult("HIGH_CONFIDENCE_WHEELO_CONFIRMED", wheelo_support, "N/A", None,
                                     "both models positive EV, both above implied probability, "
                                     f"internal gap {internal_gap_pp:.1f}pp <= {INTERNAL_GAP_HIGH_CONFIDENCE_PP}pp, Wheelo supports")
        if wheelo_support == "WHEELO_DISAGREES":
            return ConfidenceResult(
                "MEDIUM_CONFIDENCE", wheelo_support, "N/A", None,
                "internal models strongly agree, but Wheelo materially contradicts -- downgraded from High Confidence"
            )
        return ConfidenceResult("HIGH_CONFIDENCE_WHEELO_NEUTRAL", wheelo_support, "N/A", None,
                                 "both models positive EV and agree; Wheelo neither confirms nor materially contradicts")

    if both_positive and internal_gap_pp is not None and internal_gap_pp > INTERNAL_GAP_HIGH_CONFIDENCE_PP:
        outlier = "Objective" if (production_edge_pp or 0) > (objective_edge_pp or 0) else "Production"
        return ConfidenceResult("MEDIUM_CONFIDENCE", wheelo_support, "N/A", outlier,
                                 f"both models see value but internal gap {internal_gap_pp:.1f}pp > "
                                 f"{INTERNAL_GAP_HIGH_CONFIDENCE_PP}pp -- {outlier} is the higher-edge outlier")

    if only_one_positive:
        if wheelo_support in ("STRONG_WHEELO_SUPPORT", "PARTIAL_WHEELO_SUPPORT"):
            outlier = "Objective" if production_edge_pp and production_edge_pp > 0 else "Production"
            return ConfidenceResult("HIGH_RISK_HIGH_REWARD", wheelo_support, "N/A", outlier,
                                     f"only {'Production' if production_edge_pp and production_edge_pp>0 else 'Objective'} "
                                     "sees value; Wheelo materially supports it; uncertainty remains high")
        return ConfidenceResult("MODEL_DISAGREEMENT", wheelo_support, "N/A", None,
                                 "only one internal model sees value and Wheelo does not corroborate -- "
                                 "evidence too divided for a confident conclusion")

    return ConfidenceResult("NO_VALUE", wheelo_support, "N/A", None, "no meaningful positive edge from either model")
