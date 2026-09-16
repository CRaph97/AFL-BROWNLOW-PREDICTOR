# 2026 Structural Break — Umpire-Visible Statistics Rule Change

Status: **Complete (methodology); outcome genuinely unknown until real 2026 votes are revealed.**
Last updated: 2026-09-17

## 1. What actually changed

From the 2026 AFL and AFLW seasons, the four field umpires are given 17 approved statistics via a
secure Champion Data link, after the match and before voting (confirmed public information, recapped
in `docs/2026_STATS_MIRROR.md`). This is a genuine, confirmed process change with NO precedent in any
prior season this project's data covers — every single historical Brownlow vote used to train every
model in this project was cast under the old, non-statistics-assisted process.

**This means no historical data can directly measure the effect of the 2026 change.** Any model
claiming to "learn" the 2026 effect from pre-2026 data is overstating what is estimable. This document
is about managing that uncertainty honestly, not resolving it.

## 2. What Scenario C (stats-assisted) can and cannot cover

| # | Umpire-visible stat | Coverage in Scenario C |
|---|---|---|
| 1-9, 11-13 | Kicks, handballs, disposals, marks, contested marks, tackles, goals, behinds, goal assists, clearances, contested possessions | Direct (CORE, in every scenario) |
| 10 | Score involvements | Direct (`score_involvements`, ADVANCED/footywire, 2015+) |
| 13 (hitouts) | Hitouts | Direct (CORE) |
| 14 | Kick-ins | **NOT COVERED** — not on afltables/footywire; only on the AFL's semi-public API, not scraped in this run |
| 15 | Intercept marks | **NOT COVERED** — same reason |
| 16 | Intercept possessions | Partial proxy only — footywire's `ITC`/`intercepts`, NOT confirmed identical to Champion Data's definition |
| 17 | Spoils | **NOT COVERED** — same reason as kick-ins |

**Honest summary: 14 of 17 confirmed umpire-visible stats are covered directly or via a documented
proxy; 3 (kick-ins, intercept marks, spoils) are not covered in this production run.** Scenario C is
therefore a real, useful approximation of "an umpiring process that weighs displayed stats more
heavily" — not a literal reconstruction of the umpire's screen. This is disclosed rather than
smoothed over.

## 3. Why a blended sensitivity analysis, not a single "corrected" model

Because there are zero 2026 Brownlow votes to fit against, there is no statistically defensible way
to estimate exactly how much more weight umpires now place on objective, visible stats versus
historical latent/reputation-style factors. Any single point estimate of that shift would be a
disguised guess. Instead, three round, clearly-labelled sensitivity levels blend Scenario A's and
Scenario C's PROBABILITIES directly (a linear opinion pool), per player, per match:

    p_D(k) = (1 - alpha) * p_A_historical(k) + alpha * p_C_stats_assisted(k)     for k in {3, 2, 1}

(An earlier draft of this blend combined unit-variance-standardised UTILITIES instead of
probabilities. That was found to be wrong during Phase 5 QA — see `src/models/build_2026_ensemble.py`'s
module docstring for the full diagnosis — because a Plackett-Luce utility's scale is not arbitrary the
way its additive per-match shift is; forcibly equalising every scenario's variance before averaging
discarded each model's own learned confidence and mechanically flattened the blend. Blending already-
coherent probabilities avoids this failure mode entirely and is the standard, well-justified technique
for combining probabilistic forecasts from multiple models.)

| Level | alpha | Interpretation |
|---|---|---|
| LOW | 0.15 | Umpires' behaviour barely changes; the new screen is background information at most |
| MEDIUM | 0.35 | A moderate, plausible shift toward objective/visible stats |
| HIGH | 0.60 | Umpires lean heavily on the newly displayed numbers |

**No claim is made that any one of these is correct.** They exist so a reader can see, for any given
player, whether their projection is stable across the whole plausible range (high confidence) or
swings a lot (report as `structural_break_sensitivity` in the leaderboard, computed as
`max(A_historical, C_stats_assisted) EV - min(...)` per player — see `reports/2026_scenario_comparison.csv`).

## 4. Final ensemble weighting is a judgement call, not a fitted result

The production ensemble (`docs/2026_MODELLING_METHODOLOGY.md` §6, weights documented in
`src/models/build_2026_ensemble.py`) uses **0.45 / 0.20 / 0.35** for Scenario A / B / C respectively,
blended the same way (a probability-weighted linear opinion pool, each scenario renormalised to the
common player set first).
This is stated plainly as a documented judgement call informed by (a) Phase 4's out-of-sample
validation strength of the underlying CORE architecture, and (b) how directly each scenario addresses
the actual 2026 rule change — not a number derived from any statistical estimation procedure, because
no such procedure exists without 2026 votes. A different, equally defensible analyst could choose
different weights; the leaderboard's `structural_break_sensitivity` and `model_disagreement_range`
columns exist specifically so a reader is not forced to trust this one weighting blindly.

## 5. What would resolve this uncertainty

Once actual 2026 Brownlow votes are revealed (after the home-and-away season, before the Grand
Final), a genuine test becomes possible: check whether the historical model or the stats-assisted
model (or some alpha in between) better predicts the ACTUAL 2026 count. This is the natural first
task for a future phase, once the target exists.
