# Experimental Game-State Model — Phase 4, Model 5

Status: **Complete.** Rerun and confirmed after the Phase 4 Plackett-Luce correctness fix (see
`docs/PHASE4_DECISIONS.md`) — the original run used the pre-fix, unstandardised model and its result
was flagged provisional; this is the confirmed rerun.
Last updated: 2026-09-17

## What this tests

The brief's central open research question: **does intra-match timing/game-state information
materially improve Brownlow vote prediction beyond normal match-level statistics?** This is a
controlled comparison — same base architecture (Plackett-Luce), same base feature set, with the ONLY
difference being the presence of 5 event-derived leverage features.

## Data and features

- Source: the `torp`/`torpdata` chains feed (2021-2025), validated in `docs/EVENT_DATA_2021_AUDIT.md`
  (Phase 2/3) and further hardened in Phase 4 (root-caused the 2 anomalous 2024 matches as genuine
  source-data completeness gaps, now excluded via a >=4-periods rule).
- Features built in `src/features/build_experimental_features.py`: for every player-match, every
  disposal-type event (Kick/Handball/Ground Kick) and scoring-type event (Goal/Behind) attributed to
  that player is weighted by a `simple_leverage(absolute_margin, period, period_seconds)` function
  (closeness x time-elapsed, unfit to any Brownlow outcome — a transparent, arbitrary-but-documented
  placeholder, not a tuned weighting). Aggregated to: `leverage_weighted_disposals`,
  `mean_leverage_disposals`, `leverage_weighted_scoring`, `n_disposal_events`, `n_scoring_events`.
- Player identity resolved from Champion-Data provider IDs back to the canonical afltables player_id
  via normalised surname + team + season (96% resolution rate after fixing a team-naming gap — see
  `docs/PHASE4_DECISIONS.md` §A5).

## Known limitation of the EXPERIMENTAL sample

Players with literally zero qualifying disposal or scoring events in a match (rare, but possible for a
brief substitute appearance) are absent from the event aggregation entirely and are therefore dropped
from this comparison's match rosters via the inner join — a small, documented completeness gap specific
to the EXPERIMENTAL dataset, not present in CORE/ADVANCED.

## Method

Test seasons: 2023, 2024, 2025 (train on all earlier available EXPERIMENTAL seasons, expanding window
— the longest feasible sequence given the dataset only spans 2021-2025).

## Results

| Season | Variant | n_train | n_test | correct_3 | exact_321 | log_loss |
|---|---|---|---|---|---|---|
| 2023 | without_gamestate | 17,351 | 9,230 | 0.470 | 0.049 | 0.194 |
| 2023 | with_gamestate | 17,351 | 9,230 | 0.475 | 0.055 | 0.195 |
| 2024 | without_gamestate | 26,581 | 9,040 | 0.500 | 0.067 | 0.179 |
| 2024 | with_gamestate | 26,581 | 9,040 | 0.500 | 0.073 | 0.179 |
| 2025 | without_gamestate | 35,621 | 9,182 | 0.549 | 0.058 | 0.179 |
| 2025 | with_gamestate | 35,621 | 9,182 | 0.549 | 0.069 | 0.179 |

**Mean across the 3 test seasons:**

| Variant | mean_correct_3 | mean_exact_321 | mean_log_loss | mean_rank_corr |
|---|---|---|---|---|
| with_gamestate | 0.5082 | 0.0657 | 0.18428 | 0.3938 |
| without_gamestate | 0.5064 | 0.0581 | 0.18411 | 0.3938 |

## Answer to the brief's central question

**No, not meaningfully.** Adding the event-derived leverage features (`leverage_weighted_disposals`,
`mean_leverage_disposals`, `leverage_weighted_scoring`, event counts) to an otherwise identical
Plackett-Luce model produces differences that are essentially noise: correct-3% improves by 0.18
percentage points on average (0.5082 vs 0.5064), log loss is statistically indistinguishable
(0.18428 vs 0.18411, i.e. *very slightly worse* with game-state features), and rank correlation is
identical to 4 decimal places. The one metric showing a consistent small improvement across all 3
seasons is exact-3-2-1% (0.0657 vs 0.0581), a genuinely interesting but modest signal — getting the
full 3-2-1 order exactly right is a much harder target than picking the 3-vote winner alone, and this
is the one place a small, real edge shows up consistently.

**This is an honest, confirmed null-to-marginal result, not a data or methodology failure**: the
feature-building pipeline was validated end-to-end in Phase 3 (96% chain-level score reconciliation
accuracy), the identity/team-mapping and match-id joins were debugged and fixed in Phase 4, and the
result was rerun and reconfirmed after fixing an unrelated but serious numerical bug in the model
itself (see `docs/PHASE4_DECISIONS.md`). The most defensible interpretation is that the specific
leverage features built here — a simple, unfit closeness × time-elapsed weighting applied to raw
disposal and scoring event counts — do not capture information beyond what match-total box-score
stats, margin, and win/loss already provide, **except possibly for refining exact-order predictions
specifically**. This does not rule out a genuinely better-designed leverage feature (e.g. one that
distinguishes clearance types, or accounts for score-differential trajectory rather than just the
instantaneous margin) succeeding where this first attempt did not — that remains open for a future
phase, not disproven.

## Recommendation

Do **not** promote Model 5 (game-state-augmented) to the primary/recommended model. The experimental
leverage features may be worth retaining as a minor, low-priority addition specifically for improving
exact-3-2-1-order predictions in the 2021+ era where they are available, but do not materially change
the ranking or calibration performance of the underlying architecture.
