"""
2026 Objective Stats Model -- Experimental. COMPLETELY SEPARATE from the production
model (Plackett-Luce Scenarios A/B/C/D + ensemble in train_2026_scenarios.py /
build_2026_ensemble.py). Does not import from, modify, retrain, or blend into any of
that code.

Hypothesis under test: because 2026 umpires now receive approved player statistics
after each match, voting may be driven more directly by objective match performance
than by historical Brownlow voting tendencies. This module builds an independent
Brownlow-style vote estimate using ONLY:

  - the player's own statistics from that single 2026 match
  - that match's own player pool (for within-match relative transforms)
  - that match's team result/margin

It uses ZERO historical Brownlow vote data, reputation, prior-season performance,
role (role is itself inferred from cross-match/historical form -- see
build_role_lagged.py -- so it is excluded here even though it is not a "vote"
column), player identity effects, or any historical model coefficient. Weights
below are hand-specified football-impact judgements, NOT fitted against historical
Brownlow votes -- see docs/2026_OBJECTIVE_MODEL.md for the full rationale and a
weight-sensitivity analysis.

The only piece of shared code from the production stack is
`plackett_luce._attach_pl_probabilities`, the exact within-match marginalisation
math (a pure function of a utility vector -> coherent P(3)/P(2)/P(1)/P(0)). This is
shared MATH INFRASTRUCTURE, not the trained model or its coefficients, so reusing it
does not blend this model into the production one.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.plackett_luce import _attach_pl_probabilities

PROCESSED = Path("data/processed")

# Columns/substrings that must NEVER appear in the objective model's feature matrix.
# Enforced programmatically (see assert_no_forbidden_inputs) rather than by convention.
FORBIDDEN_SUBSTRINGS = (
    "brownlow_votes",
    "prev3_mean", "prev5_mean", "prev10_mean", "prevN_count",
    "season_to_date_mean", "season_to_date_n_games",
    "role", "role_source",  # role is a lagged, cross-match/historical proxy -- excluded
)

# Identity/context columns that ARE allowed: they describe *this* match only.
CONTEXT_COLS = [
    "match_id", "player_id", "player_name", "team_id", "opponent_id", "season",
    "date", "venue", "home_away",
    "team_score", "opponent_score", "margin", "absolute_margin",
    "is_win", "is_loss", "is_draw", "is_blowout", "is_close_game",
]

# Raw current-match stat columns used as scoring inputs (CORE + ADVANCED, 2026 only).
CORE_STAT_COLS = [
    "disposals", "disposals_match_z",
    "contested_possessions", "contested_possessions_match_z",
    "contested_marks", "contested_marks_match_z",
    "clearances", "clearances_match_z",
    "goals", "behinds",
    "inside_50s", "inside_50s_match_z",
    "marks", "marks_match_z",
    "tackles", "tackles_match_z",
    "one_percenters", "rebound_50s",
    "hitouts",
    "frees_for", "frees_against", "clangers",
]
ADV_STAT_COLS = [
    # goal_assists is already present in CORE (afltables-sourced) -- not pulled again
    # from ADVANCED to avoid a duplicate-column merge collision.
    "effective_disposals", "disposal_efficiency_pct",
    "score_involvements",
    "metres_gained", "centre_clearances",
    "intercepts", "turnovers", "tackles_inside_50",
]

# Candidate inputs from the brief that are NOT available in this project's data
# (documented, not silently dropped -- see docs/2026_STRUCTURAL_BREAK.md sec.2 for
# the same finding in the production model's Scenario C):
#   kick-ins, intercept marks, spoils, time_on_ground-weighted adjustments
# `time_on_ground_pct` IS present in CORE but is deliberately not used as a scoring
# input: it would reward players simply for game time rather than impact per minute,
# which is not one of the brief's requested signal categories.
UNAVAILABLE_INPUTS = ["kick_ins", "intercept_marks", "spoils"]


def load_2026_input_frame() -> pd.DataFrame:
    """CORE 2026 rows left-joined with the ADVANCED 2026 extra stat columns.
    2026-only, no other season. ~1.9% of rows have no ADVANCED match (footywire
    join gap, same documented gap as the production Scenario C) -- handled with
    an explicit fallback in the scoring function, not dropped.
    """
    core = pd.read_parquet(PROCESSED / "model_core_2026.parquet")
    core = core[core["season"] == 2026].copy()
    adv = pd.read_parquet(PROCESSED / "model_advanced_2026.parquet")
    adv = adv[adv["season"] == 2026].copy()
    adv_cols = ["match_id", "player_id"] + ADV_STAT_COLS
    merged = core.merge(adv[adv_cols], on=["match_id", "player_id"], how="left")
    return merged


def assert_no_forbidden_inputs(cols) -> list[str]:
    """Returns the list of column names in `cols` that violate FORBIDDEN_SUBSTRINGS.
    Empty list == clean. Used both by the pipeline (fail loudly) and by tests
    (verify, not just assert, that no historical/reputation/role data ever enters
    the feature matrix actually used for scoring).
    """
    return [c for c in cols if any(s in c for s in FORBIDDEN_SUBSTRINGS)]


_Z_CLIP = 3.5  # winsorise within-match z-scores: several inputs (e.g. hitouts, centre
# clearances) are near-zero for most players in a match, so their within-match std can
# be tiny and produce an extreme z for the one player who touched the stat at all.
# Clipping at +/-3.5 (a standard winsorisation bound) keeps every group composite on a
# comparable, bounded scale -- both for interpretability of the hand-specified weights,
# and because the shared Plackett-Luce marginalisation is only numerically exact for a
# bounded utility spread within a match (see _attach_pl_probabilities's own docstring).


def _within_match_z(df: pd.DataFrame, col: str, match_col: str = "match_id") -> pd.Series:
    """Within-match z-score: uses ONLY that match's own player rows (mean/std computed
    per match_id group). No other match's data enters this computation. Clipped to
    +/-_Z_CLIP -- see _Z_CLIP comment above."""
    g = df.groupby(match_col)[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    z = (df[col].astype(float) - mean) / std
    return z.fillna(0.0).clip(-_Z_CLIP, _Z_CLIP)


def _goal_hinge(goals: pd.Series) -> pd.Series:
    """Convex (superlinear) credit for goalkicking, mirroring the project's own
    established hinge pattern (feature_sets.py's goals_over3/goals_over5), but with
    fresh, un-fitted thresholds -- these coefficients are hand-specified football
    judgement, not fitted to historical votes. Marginal reward increases at each
    threshold: 1st-2nd goal worth 1.0 each, 3rd-4th worth 1.6 each, 5th+ worth 2.2
    each -- so a 5-goal haul (1+1+1.6+1.6+2.2=7.4) is much more than 2.5x a 2-goal
    game (2.0), reflecting "5 goals should carry substantially more weight than 2."
    """
    g = goals.astype(float)
    tier1 = np.minimum(g, 2) * 1.0
    tier2 = np.clip(g - 2, 0, 2) * 1.6
    tier3 = np.clip(g - 4, 0, None) * 2.2
    return tier1 + tier2 + tier3


def _disposal_hinge(disposals: pd.Series) -> pd.Series:
    """Convex credit for disposal volume beyond typical levels, same rationale as
    _goal_hinge. Below 20 disposals: no bonus (a hinge, not a linear reward -- an
    average game shouldn't be inflated). 20-28: modest marginal credit. 28+: higher
    marginal credit, reflecting genuinely elite disposal counts."""
    d = disposals.astype(float)
    tier1 = np.clip(d - 20, 0, 8) * 0.15   # 20-28
    tier2 = np.clip(d - 28, 0, None) * 0.28  # 28+
    return tier1 + tier2


# Group weights: a 100-point hand-specified budget, NOT fitted against historical
# Brownlow votes. Every weight has a one-line rationale below; the full writeup and
# a +/-25% perturbation sensitivity analysis live in docs/2026_OBJECTIVE_MODEL.md.
GROUP_WEIGHTS = {
    "POSSESSION_QUALITY": 16,  # foundational two-way involvement signal; capped so raw
                               # disposal counts alone can't dominate the score
    "CONTEST": 14,             # contested-ball winning signals two-way influence beyond volume
    "CLEARANCE": 12,           # clearances directly create scoring chances from stoppages
    "SCORING": 16,             # goals are the most visible, game-deciding stat; nonlinear
    "SCORE_CREATION": 10,      # rewards players who create scores, not just kick them
    "TERRITORY": 8,            # advancing the ball forward has real but more team-dependent value
    "DEFENCE": 8,              # rewards defensive/negating work -- partially offsets the known
                               # production-model tendency to under-credit defenders
    "PRESSURE": 8,             # tackling/forcing-turnover effort, two-way but non-possession
    "RUCK": 4,                 # hitouts are near-zero for ~95% of players; small weight avoids
                               # inflating ruck scores while still rewarding dominant ruck games
    "TEAM_RESULT": 4,          # winning helps but must not dominate -- see _team_result_adjustment
}
assert sum(GROUP_WEIGHTS.values()) == 100


def _team_result_adjustment(margin: pd.Series) -> pd.Series:
    """Bounded, saturating function of the player's own team's signed margin
    (positive = win). tanh saturates so a 40-point win and an 80-point win are
    NOT twice as different -- "large win = modest additional support, not
    automatic votes" -- and a close loss gets a near-zero penalty while a big
    loss gets a real (but still bounded) penalty. Max magnitude = GROUP_WEIGHTS
    ["TEAM_RESULT"], i.e. +/-4 points out of the 100-point scale.
    """
    cap = GROUP_WEIGHTS["TEAM_RESULT"]
    return cap * np.tanh(margin.astype(float) / 40.0)


def _clip_precomputed_z(s: pd.Series) -> pd.Series:
    """Applies the same _Z_CLIP winsorisation to the CORE table's own precomputed
    Phase 3/4 *_match_z columns (which this module also consumes directly), so every
    z-score feeding the objective score is bounded consistently regardless of whether
    it was computed here or upstream."""
    return s.fillna(0.0).clip(-_Z_CLIP, _Z_CLIP)


def compute_objective_scores(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Attaches `objective_score` (and its group components, for driver explanations)
    to df. `weights` defaults to GROUP_WEIGHTS; overriding it is how the sensitivity
    analysis perturbs one group at a time without touching this function.
    """
    w = weights or GROUP_WEIGHTS
    out = df.copy()

    has_adv = out["effective_disposals"].notna()

    # POSSESSION_QUALITY: effective disposals is the primary quality signal (available
    # for ~98% of rows); when missing, fall back to disposals_match_z alone rather than
    # silently zero-filling. Raw disposal volume still contributes, but at reduced
    # weight, so total disposals and effective disposals are not both counted in full --
    # avoids double-counting two highly correlated stats.
    eff_z = _within_match_z(out, "effective_disposals")
    eff_pct_z = _within_match_z(out, "disposal_efficiency_pct")
    disp_z = _clip_precomputed_z(out["disposals_match_z"])
    disp_hinge_z = _within_match_z(out.assign(_dh=_disposal_hinge(out["disposals"])), "_dh")
    possession_full = 0.40 * eff_z + 0.25 * eff_pct_z + 0.20 * disp_z + 0.15 * disp_hinge_z
    possession_fallback = 0.55 * disp_z + 0.45 * disp_hinge_z
    out["grp_possession_quality"] = np.where(has_adv, possession_full, possession_fallback)

    # CONTEST: contested possessions + contested marks, both real distinct signals,
    # equal sub-weight (no correlation discount needed -- they capture different skills).
    out["grp_contest"] = 0.5 * _clip_precomputed_z(out["contested_possessions_match_z"]) + \
        0.5 * _clip_precomputed_z(out["contested_marks_match_z"])

    # CLEARANCE: clearances is the primary signal; centre clearances (ADVANCED, ~85%
    # coverage) adds a bonus for stoppage-specific dominance when available.
    cc_z = _within_match_z(out, "centre_clearances")
    has_cc = out["centre_clearances"].notna()
    out["grp_clearance"] = np.where(
        has_cc,
        0.7 * _clip_precomputed_z(out["clearances_match_z"]) + 0.3 * cc_z,
        _clip_precomputed_z(out["clearances_match_z"]),
    )

    # SCORING: convex goal hinge (z-scored within match to keep the group on the same
    # scale as the others) plus a small linear behinds credit.
    goal_hinge_z = _within_match_z(out.assign(_gh=_goal_hinge(out["goals"])), "_gh")
    behinds_z = _within_match_z(out, "behinds")
    out["grp_scoring"] = 0.85 * goal_hinge_z + 0.15 * behinds_z

    # SCORE_CREATION: score involvements (ADVANCED) is the best single "created scoring
    # value" signal; goal assists is a purer, sparser signal for direct assists.
    si_z = _within_match_z(out, "score_involvements")
    ga_z = _within_match_z(out, "goal_assists")
    has_si = out["score_involvements"].notna()
    out["grp_score_creation"] = np.where(has_si, 0.65 * si_z + 0.35 * ga_z, ga_z)

    # TERRITORY: inside 50s (CORE, always present) + metres gained (ADVANCED, ~98%).
    mg_z = _within_match_z(out, "metres_gained")
    has_mg = out["metres_gained"].notna()
    out["grp_territory"] = np.where(
        has_mg,
        0.5 * _clip_precomputed_z(out["inside_50s_match_z"]) + 0.5 * mg_z,
        _clip_precomputed_z(out["inside_50s_match_z"]),
    )

    # DEFENCE: one-percenters + rebound 50s (both CORE, always present) + intercepts
    # (ADVANCED proxy, partial coverage, documented as an imperfect proxy in
    # docs/2026_STRUCTURAL_BREAK.md sec.2). This group exists specifically because the
    # production model is documented to under-credit elite defensive performances.
    op_z = _within_match_z(out, "one_percenters")
    rb_z = _within_match_z(out, "rebound_50s")
    ic_z = _within_match_z(out, "intercepts")
    has_ic = out["intercepts"].notna()
    out["grp_defence"] = np.where(has_ic, (op_z + rb_z + ic_z) / 3.0, (op_z + rb_z) / 2.0)

    # PRESSURE: tackles (positive) plus a small penalty for clangers/turnovers and
    # frees against -- pressure/discipline are two sides of the same "effort without
    # the ball" coin. Turnovers (ADVANCED) preferred over clangers (CORE) when present
    # since it's a cleaner definition; never both, to avoid double-counting.
    tk_z = _clip_precomputed_z(out["tackles_match_z"])
    fa_z = _within_match_z(out, "frees_against")
    to_z = _within_match_z(out, "turnovers")
    cl_z = _within_match_z(out, "clangers")
    has_to = out["turnovers"].notna()
    turnover_penalty = np.where(has_to, to_z, cl_z)
    out["grp_pressure"] = 0.7 * tk_z - 0.2 * fa_z - 0.1 * turnover_penalty

    # RUCK: hitouts, z-scored within match. Near-zero for the ~90% of players who don't
    # ruck, so this group only meaningfully rewards genuine ruck contests.
    out["grp_ruck"] = _within_match_z(out, "hitouts")

    # TEAM_RESULT: bounded, saturating function of the player's own team's margin.
    out["grp_team_result"] = _team_result_adjustment(out["margin"]) / w["TEAM_RESULT"]  # normalise to [-1,1]-ish

    group_cols = {
        "POSSESSION_QUALITY": "grp_possession_quality",
        "CONTEST": "grp_contest",
        "CLEARANCE": "grp_clearance",
        "SCORING": "grp_scoring",
        "SCORE_CREATION": "grp_score_creation",
        "TERRITORY": "grp_territory",
        "DEFENCE": "grp_defence",
        "PRESSURE": "grp_pressure",
        "RUCK": "grp_ruck",
        "TEAM_RESULT": "grp_team_result",
    }
    score = np.zeros(len(out))
    for group, col in group_cols.items():
        score = score + w[group] * out[col].to_numpy()
    out["objective_score"] = score

    # UTILITY_TEMPERATURE: the 0-100-point GROUP_WEIGHTS budget is deliberately sized
    # for human interpretability of `objective_score` (a "how good was this game"
    # points scale), not for numerical use as a Plackett-Luce log-utility -- raw point
    # gaps of 100+ between the best and worst player in a match would make some
    # players' exp(utility) exceed float64 precision relative to the match total,
    # corrupting the marginalisation (the exact numerical failure mode documented for
    # the production model in docs/PHASE4_DECISIONS.md). Dividing by a fixed constant
    # before the Plackett-Luce step compresses this onto a realistic utility scale
    # (comparable to a fitted logistic-style model's coefficient range) without
    # changing the RELATIVE ordering or relative shape of any player's score --
    # `objective_score` (used for display/ranking of raw quality) is left unscaled;
    # only `objective_utility` (fed to the probability step) is divided.
    UTILITY_TEMPERATURE = 15.0
    out["objective_utility"] = score / UTILITY_TEMPERATURE
    return out


def build_match_probabilities(df_scored: pd.DataFrame) -> pd.DataFrame:
    """Per match, feed `objective_score` as the utility into the shared exact
    Plackett-Luce marginalisation (production MATH infrastructure, not the trained
    production model) to get coherent p3/p2/p1/p0/expected_votes per player."""
    out_frames = []
    for match_id, g in df_scored.groupby("match_id", sort=False):
        g = g.reset_index(drop=True)
        u = g["objective_utility"].to_numpy()
        scored = _attach_pl_probabilities(g, u)
        out_frames.append(scored)
    return pd.concat(out_frames, ignore_index=True)


def top_drivers(row: pd.Series, group_cols: dict[str, str], n: int = 3) -> str:
    """Plain-text top-n contributing groups for a scored player row, for the
    match-view 'principal statistical reasons' display."""
    contribs = {g: GROUP_WEIGHTS[g] * row[col] for g, col in group_cols.items()}
    ranked = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return "; ".join(f"{g.replace('_', ' ').title()} ({v:+.1f})" for g, v in ranked if v > 0.05)
