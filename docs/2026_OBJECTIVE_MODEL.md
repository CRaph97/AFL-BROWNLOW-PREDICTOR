# 2026 Objective Stats Model — Experimental

Status: **Complete.**
Last updated: 2026-09-17

## 1. Purpose and hypothesis

From 2026, umpires receive approved player-performance statistics after each match, before
voting (see `docs/2026_STRUCTURAL_BREAK.md`). This raises a hypothesis: voting in 2026 could be
driven more directly by objective match performance than by historical Brownlow voting
tendencies. This model tests that hypothesis by building a completely independent Brownlow-style
vote estimate that uses **zero historical Brownlow voting data of any kind** — only a player's
own statistics from a single 2026 match, that match's own player pool, and that match's result.

**This is a standalone comparison tool, not a replacement for the production model.** It does
not import from, retrain, modify, or blend into `train_2026_scenarios.py`,
`build_2026_ensemble.py`, or any other production 2026 file. The only shared code is
`plackett_luce._attach_pl_probabilities` — the exact within-match probability marginalisation
math (a pure function mapping a utility vector to coherent P(3)/P(2)/P(1)/P(0)). This is shared
mathematical infrastructure, not the trained model or its coefficients, so reusing it does not
blend this model into the production one; see `src/models/objective_stats_model.py`'s module
docstring for the same reasoning in code.

## 2. Hard exclusions (verified, not just asserted)

`objective_stats_model.assert_no_forbidden_inputs()` greps the actual columns fed into
`compute_objective_scores()` for `brownlow_votes`, every lagged/`prevN`/`season_to_date` reputation
or form column, and `role`/`role_source` (role is excluded even though it isn't a "vote" column,
because it is itself inferred from a player's cross-match/historical form — see
`build_role_lagged.py` — and the brief requires *no cross-match history at all*, not even the
player's own prior 2026 rounds). `tests/test_objective_stats_model.py` re-verifies this
programmatically rather than trusting the pipeline's own assertion. `reports/2026_objective_quality_checks.json`'s
`forbidden_columns_in_scoring_inputs` field is empty, confirmed on every run.

## 3. Inputs used, and what's genuinely unavailable

All inputs come from the already-validated `data/processed/model_core_2026.parquet` (afltables,
2026 rows only) and `model_advanced_2026.parquet` (footywire extended stats, 2026 rows only,
joined on `match_id`+`player_id`; ~1.9% of rows have no ADVANCED match — the same documented
footywire join gap as the production Scenario C — handled with an explicit fallback per group,
never silently dropped).

Used: disposals, effective disposals, disposal efficiency, contested possessions, contested
marks, marks, clearances, centre clearances, goals, behinds, goal assists, score involvements,
inside 50s, metres gained, tackles, one-percenters, rebound 50s, intercepts (proxy), hitouts,
frees for/against, clangers/turnovers, time-on-ground (available but deliberately unused — see
below), team score/opponent score/margin/win-loss-draw.

**Not available, per the earlier structural-break audit** (`docs/2026_STRUCTURAL_BREAK.md` §2):
kick-ins, intercept marks, spoils. Not scraped for this task per the brief's efficiency
constraint (no new data collection unless genuinely necessary for a required input).
`time_on_ground_pct` is available but not used — it would reward game time rather than
performance quality, which isn't one of the requested signal categories.

## 4. Relative and nonlinear transforms

Within-match z-scores (mean/std computed from *that match's own player rows only* — verified in
`tests/test_objective_stats_model.py::test_no_cross_match_leakage`) are the primary relative
measure, reused directly from Phase 3/4's own precomputed `*_match_z` columns where they already
exist (disposals, contested possessions, contested marks, clearances, marks, tackles, goals,
inside 50s), and computed fresh via `_within_match_z()` for stats that don't have one yet
(effective disposals, disposal efficiency, score involvements, metres gained, centre clearances,
one-percenters, rebound 50s, intercepts, hitouts, frees against, turnovers/clangers).

All z-scores (precomputed and fresh) are winsorised to ±3.5 (`_Z_CLIP`). This matters for two
reasons: (1) several inputs (hitouts, centre clearances) are near-zero for most players in a
match, so their within-match standard deviation can be tiny, producing an extreme raw z for the
one player who touched the stat at all; (2) the shared Plackett-Luce marginalisation code is only
numerically exact for a bounded utility spread within a match — an unbounded z could reproduce the
exact catastrophic-cancellation failure mode documented for the production model in
`docs/PHASE4_DECISIONS.md`.

Nonlinear ("elite performance") credit is applied via two hand-specified convex hinge functions,
following the same *pattern* as the project's own established `disposals_over25/30`,
`goals_over3/5` hinge features (`src/models/feature_sets.py`) — but with fresh thresholds, since
the original features' coefficients were fitted against historical Brownlow votes and cannot be
reused here:

- **Goals** (`_goal_hinge`): 1st–2nd goal worth 1.0 each, 3rd–4th worth 1.6 each, 5th+ worth 2.2
  each. A 5-goal haul scores 7.4 — more than 2.5× a 2-goal game's 2.0 — satisfying "5 goals should
  carry substantially more weight than 2."
- **Disposals** (`_disposal_hinge`): no bonus below 20 (an average game isn't inflated); 0.15/disposal
  from 20–28; 0.28/disposal above 28 — increasing marginal credit for genuinely elite volume.

Both hinge outputs are then z-scored within-match (same winsorisation) so they combine with the
other group members on a comparable scale.

## 5. Team result handling

`_team_result_adjustment()`: `TEAM_RESULT_weight * tanh(margin / 40)`, where `margin` is the
player's own team's signed scoring margin (positive = win) in *that* match. This is a bounded,
saturating function: a 40-point win and an 80-point win are not twice as different (tanh
saturates), satisfying "large win = modest additional support, not automatic votes," while a
close loss gets a near-zero penalty and a genuinely large loss gets a real, but still bounded,
penalty. Maximum magnitude is exactly the `TEAM_RESULT` group weight (±4 of the 100-point budget).

## 6. Weighting scheme (100-point hand-specified budget)

**Not fitted against historical Brownlow votes anywhere in this pipeline** — every weight below is
a documented football-impact judgement call, per the brief's explicit instruction. Source of truth:
`GROUP_WEIGHTS` in `src/models/objective_stats_model.py`.

| Group | Weight | Rationale |
|---|---|---|
| POSSESSION_QUALITY | 16 | Foundational two-way involvement signal; capped so raw disposal counts alone can't dominate the score |
| CONTEST | 14 | Contested-ball winning signals two-way influence beyond mere volume |
| CLEARANCE | 12 | Clearances directly create scoring chances from stoppages |
| SCORING | 16 | Goals are the most visible, game-deciding stat; deliberately nonlinear |
| SCORE_CREATION | 10 | Rewards players who create scores, not just kick them |
| TERRITORY | 8 | Advancing the ball forward has real but more team-dependent value |
| DEFENCE | 8 | Rewards defensive/negating work — partially offsets the production model's documented tendency to under-credit defenders (`docs/ERROR_ANALYSIS.md`) |
| PRESSURE | 8 | Tackling/forcing-turnover effort — two-way value that isn't about having the ball |
| RUCK | 4 | Hitouts are near-zero for ~95% of players; kept small so it can't inflate ruck scores, but still rewards genuinely dominant ruck games |
| TEAM_RESULT | 4 | Winning should help but must not dominate (§5) |

**Avoiding double-counting of correlated stats** (per brief): total disposals and effective
disposals are not both given full independent weight — effective disposals is the primary
possession-quality signal (40% of the group when available), with raw disposal volume and the
disposal hinge contributing the remainder at reduced weight, and a full fallback to
disposals-only when the ADVANCED join is missing. Clearances/centre clearances and inside
50s/metres gained follow the same primary-signal-plus-bonus pattern. Tackles are assigned only to
PRESSURE (not also to CONTEST), and turnovers (ADVANCED) supersede clangers (CORE) when both are
available rather than penalising the same tendency twice.

## 7. From score to probabilities

`objective_score` (the 0–100-point-budget sum above) is a human-interpretable "how good was this
game" number, reported as-is in `reports/2026_objective_match_scores.csv` for driver explanations.
For the actual Plackett-Luce probability step, it is divided by a fixed `UTILITY_TEMPERATURE = 15.0`
before being passed to `_attach_pl_probabilities` as the utility. This is necessary because the
100-point weighting budget was sized for interpretability, not as a Plackett-Luce log-utility —
without this, some players' relative utility gap exceeded what float64 can represent inside the
match-level marginalisation, reproducing the exact numerical failure mode documented in
`docs/PHASE4_DECISIONS.md` for the production model (verified: the pipeline failed loudly with a
`p2 outside [0,1]` assertion before this fix, exactly as that shared code is designed to do rather
than silently emit bad probabilities). The temperature only rescales the utility fed to the
probability step; it does not change any player's relative *ranking* or the shape of
`objective_score` itself.

## 8. Sensitivity analysis

Each of the 10 group weights was perturbed ±25% (renormalising the other 9 groups so the 100-point
budget stays fixed), and the resulting Top-20 leaderboard was compared against the base
leaderboard's Top 20 for all 20 perturbation scenarios. Full detail:
`reports/2026_objective_sensitivity.json`.

**Result: Top-20 overlap ranged from 19/20 to 20/20 across all 20 scenarios (mean 19.5/20).** No
single ±25% weight change moved more than one player in or out of the top 20. **The leaderboard is
not an artifact of any single arbitrary coefficient.**

## 9. Outputs

- `reports/2026_objective_match_scores.csv` — every 2026 match × player row: group-component
  scores, `objective_score`, P(3)/P(2)/P(1)/P(0), expected votes, plain-text primary drivers.
- `reports/2026_objective_votes.csv` — same grain, plus the deterministic (argmax) 3/2/1 pick.
- `reports/2026_objective_leaderboard.csv` — season aggregation, ranked by objective expected votes.
- `reports/2026_objective_vs_production.csv` — side-by-side comparison with the production
  `reports/2026_leaderboard.csv` (Production EV/Rank, Objective EV/Rank, Difference, Rank
  Difference).
- `reports/2026_objective_quality_checks.json` — see §10.
- `reports/2026_objective_sensitivity.json` — see §8.

## 10. Quality control (verified, not asserted)

From `reports/2026_objective_quality_checks.json`, this run:

- All 207 2026 home-and-away matches present.
- Season total expected votes = 1242.0 (207 × 6), to float precision.
- Zero forbidden (historical-vote/reputation/role) columns in the actual scoring inputs.
- Zero duplicate player-match rows.
- Every match's P(3)/P(2)/P(1) sums to 1 within ~1.6e-13 (far inside tolerance).
- Deterministic 3/2/1 picks are always three distinct players, every match.
- Existing production report files are untouched (`git status` shows this pipeline only adds new,
  untracked files — no tracked file was modified).
- Full existing test suite (40 tests) plus new tests in `tests/test_objective_stats_model.py`
  all pass — see the test run in this task's final report.

## 11. Interpretation and limitations

- **Main value is comparison, not a competing production forecast.** Large disagreements between
  this model and the production model plausibly identify players whose 2026 votes could be
  particularly sensitive to how much umpires actually rely on the new post-match statistics —
  exactly the structural-break question the project cannot resolve until real 2026 votes exist.
- **Three players (Jason Horne-Francis, Luke Davies-Uniacke, Nasiah Wanganeen-Milera) show
  Production EV of exactly 0.0** in `reports/2026_leaderboard.csv` — this is a **pre-existing,
  already-documented production-pipeline gap** (their Scenario D/ensemble/simulation columns are
  blank; their `A_historical`/`B_recent_era` scenario values are real and nonzero), previously
  flagged in the Phase 5 executive summary as a footywire join failure, not a new issue introduced
  by this work. Their large "objective rates them higher" gap in the comparison file is a data
  artifact on the production side, not a genuine signal about the 2026 rule change, and is
  reported as such rather than silently included as a real finding.
- **This model has no historical validation.** There is no meaningful backtest for it (Brownlow
  votes before 2026 were cast without post-match objective statistics being shown to umpires, so
  testing this scoring system against pre-2026 votes would test a different hypothesis entirely).
  Its credibility rests on the transparency of its inputs and weights, not on out-of-sample
  accuracy — this is disclosed plainly rather than implied away.
- **Weights remain a judgement call.** The sensitivity analysis (§8) shows the top-20 is robust to
  ±25% perturbation of any single group, but a wholesale re-weighting philosophy (e.g. much higher
  weight on TEAM_RESULT) was not explored and could shift results more than the tested
  perturbations.
