# 2026 Brownlow Medal Prediction — Final Report

Status: **Complete.**
Last updated: 2026-09-17

This is the master reference for Phase 5. It ties together `docs/PHASE4_FINAL.md`,
`docs/2026_DATA_VALIDATION.md`, `docs/2026_MODELLING_METHODOLOGY.md`,
`docs/2026_STRUCTURAL_BREAK.md` and `docs/2026_CONTENDER_ANALYSIS.md` into one place, and reports the
pre-delivery quality checks required by the brief.

## 1. What this project is predicting, and what it genuinely cannot know yet

The complete, real 2026 AFL home-and-away season (207 matches, 25 rounds, 18 teams, 2026-03-05 to
2026-08-23) has been played. **Brownlow votes for it have not been revealed** (the count happens
after the home-and-away season, before the Grand Final — finals were underway at the time this report
was produced). This report is a genuine predictive forecast of an outcome that is real and fixed but
not yet public — not a retrospective evaluation, and not a forecast of a hypothetical/future season.

## 2. Quality checks (section 15 of the brief) — actually run, not just claimed

From `reports/2026_quality_checks.json`, produced by `src/models/build_2026_outputs.py`:

| Check | Result | Pass? |
|---|---|---|
| Every deterministic match allocates exactly 3+2+1=6 votes | max deviation 0 across all 201 scored matches | **PASS** |
| P(3)/P(2)/P(1) sum to 1 within every match | max error 2.2e-16 (float precision) | **PASS** |
| Expected votes sum to 6 per match | max error 8.9e-16 (float precision) | **PASS** |
| Every simulated season (100,000 sims) sums to exactly 6 x 201 = 1,206 votes | min=max=1,206 across all 100,000 sims | **PASS** |
| No duplicate player-match rows | 0 found | **PASS** |
| All home-and-away matches represented in the raw data | 207/207 | **PASS** |
| All 207 matches SCORED by the model | **201/207 (97.1%)** | **PARTIAL — disclosed, not silent** |
| No missing core box-score stats (disposals/goals/clearances) | 0 of 9,522 2026 rows | **PASS** |

**The one partial result is disclosed and explained, not hidden**: 6 of 207 matches (5 actual Round-1
fixtures + 1 Round-2 fixture for the 2 teams that had a Round-1 bye) cannot be scored because
`disposals_season_to_date_mean` and its 3 lagged-form siblings are structurally undefined for a team's
first game of a season — true in every historical season, not a 2026-specific defect (see
`docs/2026_MODELLING_METHODOLOGY.md` §6b). Effect: season-total expected votes for any player are a
small, systematic, disclosed undercount if that player had a big game in their team's actual first
fixture of 2026. This was a genuine, non-blocking data/feature limitation, resolved per the brief's
instruction to document and continue rather than fabricate a value for it.

## 3. Headline result

**Nick Daicos (Collingwood) is the clear projected 2026 Brownlow Medal leader**, with a FINAL_ENSEMBLE
expected vote tally of **44.9** (Monte Carlo 95% range: 39-50), separated from #2 Bailey Smith (36.2)
by a wider margin than separates #2 from #6. This is also the single most STABLE projection in the
field (structural-break sensitivity 0.04, model disagreement 0.91 — both the smallest of any top-10
contender), meaning this result does not hinge on an assumption about how the 2026 rule change plays
out. Full detail: `docs/2026_CONTENDER_ANALYSIS.md`.

## 4. How the 2026 rule change was handled

No historical vote has ever been cast under the new umpire-statistics-assisted process, so no model
can "learn" its true effect. Rather than picking one arbitrary adjustment, four scenarios were built
(historical / recent-era / stats-assisted / structural-break sensitivity bands) and combined into a
documented, non-uniform ensemble (0.45 historical / 0.20 recent-era / 0.35 stats-assisted) — full
rationale in `docs/2026_STRUCTURAL_BREAK.md`. The leaderboard's `structural_break_sensitivity` column
lets a reader see exactly which players' rank depends on this unresolved assumption (Zak Butters,
4.49, is by far the most sensitive top-10 case) versus which are robust to it (Nick Daicos, 0.04;
Lachie Neale, 0.02).

## 5. Uncertainty — four distinct kinds, not one blended confidence number (section 13)

| Type | What it captures | Where it shows up |
|---|---|---|
| **Aleatoric** (genuine voting randomness) | Even a perfect model cannot predict a close, subjective 3-vs-2 umpire call with certainty | The width of each match's own P(3)/P(2)/P(1) — e.g. Daicos round 8 is a genuine 46/50/4 split, not model uncertainty |
| **Model uncertainty** | Different validated architectures/windows disagree | `model_disagreement_range` per player (Zak Butters 6.53 vs Nick Daicos 0.91) |
| **Structural-break uncertainty** | The 2026 rule change's true size is unknown and unknowable from historical data | `structural_break_sensitivity` per player, and the explicit LOW/MEDIUM/HIGH bands in `docs/2026_STRUCTURAL_BREAK.md` |
| **Data uncertainty** | Missing/incomplete statistics for specific players or matches | The 6 unscored matches (§2 above); the 3 players with a failed footywire join (`docs/2026_CONTENDER_ANALYSIS.md` closing note); the 21.1% of 2026 player-rows with `role_source = unknown_insufficient_history` |

Collapsing these into one number would hide the fact that a player can be LOW on one axis and HIGH on
another — e.g. Daicos is low on all four; Zak Butters is low on data/aleatoric uncertainty but high
specifically on structural-break and model uncertainty.

## 6. Model weaknesses carried into this forecast

Phase 4's error analysis (`docs/PHASE4_FINAL.md` §7) found the model's biggest, most consistent blind
spot is **key and medium defenders** (12.5% / 33.9% accuracy identifying them as the actual 3-vote
winner, vs 62.0% for midfielders) — this weakness is inherited unchanged into the 2026 forecast, since
no defensive-role-specific correction was introduced in Phase 5. A reader should treat any defender's
position in the 2026 leaderboard with additional caution beyond what the reported intervals capture.
Phase 4 also found a small, consistent positive bias in season-total predictions (+0.014 to +0.051
votes per player per season, every one of 8 backtested seasons) — the 2026 totals reported here likely
carry a similar small upward bias on average, on top of the disclosed 6-match undercount from §2.

## 7. What would most improve this forecast

1. Real 2026 Brownlow votes, once revealed, would finally allow a genuine (not sensitivity-band) test
   of the structural-break hypothesis.
2. Scraping the AFL's semi-public API for kick-ins/intercept marks/spoils would let Scenario C cover
   all 17 confirmed umpire stats instead of 14.
3. A defender-specific model correction, informed directly by Phase 4's error analysis, was out of
   scope for this run but is the single most promising accuracy lever identified across the whole
   project.

## 8. File index

| File | Contents |
|---|---|
| `docs/PHASE4_FINAL.md` | Consolidated Phase 4 conclusions carried into this forecast |
| `docs/2026_DATA_VALIDATION.md` | 2026 data completeness/integrity checks |
| `docs/2026_MODELLING_METHODOLOGY.md` | Full scenario/ensemble methodology, window comparison |
| `docs/2026_STRUCTURAL_BREAK.md` | The rule-change sensitivity analysis in full |
| `docs/2026_CONTENDER_ANALYSIS.md` | Top-20 leaderboard detail and contender notes |
| `reports/2026_leaderboard.csv` | Full 569-player leaderboard |
| `reports/2026_match_probabilities.csv` | Every player-match P(3)/P(2)/P(1)/P(0)/EV (FINAL_ENSEMBLE) |
| `reports/2026_predicted_votes.csv` | Deterministic 3-2-1 pick per match |
| `reports/2026_round_by_round_contenders.csv` | Every game, top 20 contenders |
| `reports/2026_match_explanations.csv` | Plain-language drivers for 202 material games |
| `reports/2026_scenario_comparison.csv` | Per-player EV under every scenario |
| `reports/2026_model_disagreement.csv` | Top 30 players by cross-scenario disagreement |
| `reports/2026_simulation_summary.csv` | Monte Carlo distributional stats per player |
| `reports/2026_quality_checks.json` | The checks in §2, machine-readable |
| `reports/window_comparison_2026*.csv` | Recent-history window comparison, raw results |
