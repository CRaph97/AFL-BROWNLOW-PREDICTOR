# Phase 4 — Final Consolidated Conclusions (carried into Phase 5)

Status: **Complete.** This document consolidates Phase 4's already-written, already-validated
findings (full detail remains in the individual Phase 4 docs listed below) into the single reference
Phase 5 builds on. No new analysis is introduced here.

## 1. Winning model architecture

**Model 1 (Plackett-Luce, recent-8-season training window) wins on every one of 8 tracked metrics**
against Model 0 (multinomial logistic benchmark), Model 2a (GBM-utility) and Model 2b (GBM-multiclass),
across 22 walk-forward folds, 11 test seasons (2015-2025):

| Model | Window | correct_3 | exact_321 | rank_corr | log_loss | brier |
|---|---|---|---|---|---|---|
| **Model1_PlackettLuce** | **recent8** | **0.575** | **0.091** | **0.422** | **0.183** | **0.088** |
| Model0_Benchmark | expanding | 0.572 | 0.087 | 0.421 | 0.195 | 0.091 |
| Model2a_GBM_utility | expanding | 0.568 | 0.088 | 0.417 | 0.275 | 0.119 |
| Model2b_GBM_multiclass | expanding | 0.546 | 0.072 | 0.421 | 0.197 | 0.091 |

Full table: `docs/MODEL_BACKTEST.md`.

## 2. A correctness bug found and fixed mid-phase

The first ADVANCED backtest produced impossible results (correct_3 ~10-20%, exact_321 exactly 0 every
season) caused by unstandardised mixed-scale features breaking a non-log-space likelihood computation.
Fixed via train-only standardisation + a fully log-space likelihood, with 9 new regression tests. Full
writeup: `docs/PHASE4_DECISIONS.md`, `docs/MODEL_BACKTEST.md` §2.

## 3. Feature ablation and hypothesis tests

- **Win/margin context is the single biggest lever** (+5.7pp correct_3).
- **Role adds large, stable value** — `role_KEY_DEFENDER` is the single largest, most stable
  coefficient in the entire model (8-fold stability analysis).
- Match-relative features add less than Phase 3's univariate analysis suggested.
- Teammate competition: confirmed, stable, negative coefficient.
- The win×margin interaction term is completely redundant.
Full detail: `docs/FEATURE_ABLATION.md`.

## 4. Calibration

Plackett-Luce and Benchmark are both excellently calibrated out of the box (ECE 0.004 and 0.002
respectively). GBM-utility is dramatically miscalibrated (ECE 0.032, ~8-14x worse) despite reasonable
ranking accuracy — fixable via isotonic regression but unusable raw. Full detail: `docs/CALIBRATION.md`.

## 5. Game-state / event-data experiment

**Confirmed null result** (twice — before and after the Phase 4 correctness fix): game-state/event
augmentation (torpdata chain features, 2021-2025) did not materially improve historical vote
prediction. Not pursued further in Phase 5, per the standing instruction not to keep tuning a
confirmed-null experiment. Full detail: `docs/EXPERIMENTAL_GAMESTATE_MODEL.md`.

## 6. Reputation experiment

A lagged prior-vote-rate ("reputation") feature pair gives a **consistent, modest improvement**: better
or tied `correct_3` in 6 of 7 test seasons (2019-2025), better `exact_321` in 4 of 7. Not included in
the primary/default model per the brief's pre-registered interpretation rule wanting a stronger,
more decisive signal before treating it as a default feature — but retained as a documented, optional
comparison arm. Explicit caveat: an improvement here could reflect umpire recognition of a known
high-polling player rather than a genuine unmeasured performance signal. Full detail:
`docs/REPUTATION_EXPERIMENT.md`.

## 7. Error analysis — the model's biggest weakness

**Defenders are the model's single biggest, most consistent blind spot.** When a key defender
actually wins the 3 votes, the model correctly identifies them only 12.5% of the time (vs. 62.0% for
midfielders) — despite `role_KEY_DEFENDER` being the single largest coefficient in the model. The
model has learned defenders deserve a boost, but not one large enough to close the gap in their rare
best-on-ground games. Blowout matches are the model's easiest cases (79.4% accuracy vs. 53.8-59.1% in
closer games). Full detail: `docs/ERROR_ANALYSIS.md`.

## 8. Season-level pseudo-live backtest

Across 8 seasons (2018-2025): MAE 0.77-1.08 votes/player, Spearman 0.73-0.79, and **a small, consistent
positive bias in every single season** (+0.014 to +0.051 votes) — the model slightly over-predicts
season totals on average, a real and disclosed pattern. Full detail: `docs/MODEL_BACKTEST.md` §7.

## 9. What Phase 5 changed relative to a naive continuation

Phase 5 does NOT simply apply this Phase 4 model unchanged to 2026. The 2026 umpire-statistics rule
change is a genuine structural break with zero historical precedent in this project's data — Phase 5
builds explicit alternative scenarios (historical / recent-era / stats-assisted) and a sensitivity
band around the unknowable size of that break, rather than presenting one number as if the break's
effect were known. See `docs/2026_MODELLING_METHODOLOGY.md` and `docs/2026_STRUCTURAL_BREAK.md`.
