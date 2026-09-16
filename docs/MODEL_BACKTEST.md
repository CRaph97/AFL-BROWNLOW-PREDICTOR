# Model Backtest — Phase 4

Status: **Complete.**
Last updated: 2026-09-17

## 1. Datasets used

- **CORE**: `data/processed/model_core.parquet`, 2003-2025, 193,346 player-match rows.
- **ADVANCED**: `data/processed/model_advanced.parquet`, 2015-2025, CORE features plus footywire extended
  stats.
- **EXPERIMENTAL**: CORE joined (inner) with `experimental_gamestate_features.parquet`, 2021-2025,
  complete-coverage matches only.

Kept strictly separate — no model is trained on a merged CORE+ADVANCED+EXPERIMENTAL table.

## 2. A correctness bug found and fixed mid-phase

The first backtest run produced impossible numbers on the ADVANCED dataset (correct-3% ~10-20%,
exact-3-2-1% exactly 0.000 every season, log loss ~2.5). Root cause: `PlackettLuceModel` fit raw,
unstandardised features — CORE's modest-scale stats happened not to trigger a failure, but ADVANCED's
`metres_gained` (0-600+) caused catastrophic cancellation in a non-log-space likelihood computation,
corrupting the optimiser. Fixed via (1) train-only feature standardisation (binary indicators left
unscaled) and (2) a fully log-space likelihood (`log1p`-based, per-match log-sum-exp) that cannot suffer
this failure mode. 9 regression tests added (`tests/test_plackett_luce.py`), all passing, including the
exact mixed-scale scenario that caused the original failure. Full writeup: `docs/PHASE4_DECISIONS.md`.
A second, independent bug (found during error analysis) is also documented there: filtering by
player-level attributes before computing match-level metrics silently breaks the within-match
comparison set — fixed by computing correctness once on the full match, then segmenting by the actual
vote-winner's attributes.

## 3. Validation method

Walk-forward (rolling-origin) validation exclusively — training uses only seasons strictly before the
test season. Two window strategies compared: **expanding** (all available prior seasons) and
**recent-8** (the 8 most recent prior seasons only). Test seasons: 2015-2025 (11 seasons) for the
headline comparison.

## 4. Models compared

- **Model 0 (Benchmark)**: multinomial logistic regression, independent per player-match, post-hoc
  within-match renormalisation.
- **Model 1 (Plackett-Luce)**: linear-utility rank-ordered choice model, exact marginalisation,
  coherent probabilities by construction.
- **Model 2a (GBM utility)**: `HistGradientBoostingRegressor` utility fed into the same Plackett-Luce
  marginalisation.
- **Model 2b (GBM multiclass)**: `HistGradientBoostingClassifier`, independent per-row, post-hoc
  renormalised.

XGBoost could not be used (macOS libomm ABI mismatch); `HistGradientBoosting*` substituted — a
comparable histogram-based gradient boosting implementation, documented in `src/models/gbm_model.py`.

## 5. Headline results (mean across 11 test seasons, 2015-2025)

| Model | Window | correct_3 | exact_321 | all_3_id | top3_prec | rank_corr | log_loss | brier | vote_mae |
|---|---|---|---|---|---|---|---|---|---|
| **Model1_PlackettLuce** | **recent8** | **0.5750** | **0.0913** | **0.2383** | **0.6766** | **0.4217** | **0.1834** | **0.0882** | **0.1275** |
| Model0_Benchmark | expanding | 0.5719 | 0.0872 | 0.2334 | 0.6695 | 0.4208 | 0.1951 | 0.0906 | 0.1389 |
| Model1_PlackettLuce | expanding | 0.5693 | 0.0813 | 0.2184 | 0.6678 | 0.4209 | 0.1854 | 0.0892 | 0.1366 |
| Model0_Benchmark | recent8 | 0.5692 | 0.0878 | 0.2262 | 0.6724 | 0.4213 | 0.1983 | 0.0898 | 0.1327 |
| Model2a_GBM_utility | expanding | 0.5684 | 0.0876 | 0.2342 | 0.6692 | 0.4173 | 0.2747 | 0.1187 | 0.2357 |
| Model2a_GBM_utility | recent8 | 0.5576 | 0.0852 | 0.2241 | 0.6663 | 0.4166 | 0.2717 | 0.1173 | 0.2324 |
| Model2b_GBM_multiclass | expanding | 0.5458 | 0.0718 | 0.1995 | 0.6705 | 0.4211 | 0.1971 | 0.0913 | 0.1407 |
| Model2b_GBM_multiclass | recent8 | 0.5361 | 0.0670 | 0.1927 | 0.6699 | 0.4210 | 0.2014 | 0.0916 | 0.1354 |

Full per-season detail: `reports/model_metrics_by_season.csv`. Aggregated: `reports/model_comparison.csv`.
Per-match detail for the winning configuration: `reports/match_prediction_metrics.csv`.

## 6. Interpretation

**Model 1 (Plackett-Luce) with the recent-8-season window wins outright — first place on every single
metric tracked**, including the probability-quality metrics (log loss, Brier), not just ranking accuracy.
This is a clean result, not a marginal one: it beats every other model/window combination on 8 of 8
tracked metrics simultaneously.

**GBM-utility is badly miscalibrated** (log loss 0.27+ vs ~0.18-0.20 for everyone else) despite
respectable ranking accuracy — confirmed and quantified separately in `docs/CALIBRATION.md` (ECE 8x
worse than Plackett-Luce). Its ranking is fine; its probabilities should not be trusted as-is.

**GBM-multiclass (the "naive" formulation, ignoring within-match structure at training time) is the
worst model on ranking accuracy** (correct_3 0.536-0.546, exact_321 0.067-0.072 — clearly behind
everyone else). This directly answers Phase 4's Q2 ("does a ranking/choice model outperform standard
classification?"): **yes** — the structurally-correct Plackett-Luce beats the flexible-but-unstructured
GBM-multiclass formulation, and the gap is not small (correct_3 differs by 3-4 percentage points).

**Recency helps, but only for the structurally-correct model.** recent8 beats expanding for
Plackett-Luce (+0.6pp correct_3) but the pattern reverses or is mixed for every other model (GBM-utility
does *worse* with recent8; Benchmark and GBM-multiclass both do slightly worse with recent8 on
correct_3). This is a genuine, non-obvious interaction between model choice and training-window choice —
recency weighting is not a universal improvement, it specifically helps the model that is already
getting the structure right.

## 7. Season-level backtest / full historical pseudo-live simulation (sections I/J)

For every test season 2018-2025, Model 1 (Plackett-Luce, expanding window) was trained on ONLY the
seasons strictly before it, used to predict every match that season, aggregated to season-total expected
votes per player, and compared against the real totals (summed independently from the CORE table, never
derived from the model) — literally re-enacting "pretend it is immediately before this season's count."

| Test season | Train seasons | Players | MAE | RMSE | Bias | Spearman | Kendall | Top-5 incl. | Top-10 incl. |
|---|---|---|---|---|---|---|---|---|---|
| 2018 | 15 | 552 | 0.972 | 1.916 | +0.020 | 0.791 | 0.655 | 0.2 | 0.5 |
| 2019 | 16 | 552 | 0.878 | 1.587 | +0.014 | 0.774 | 0.644 | 0.6 | 0.8 |
| 2020 | 17 | 522 | 0.774 | 1.347 | +0.046 | 0.763 | 0.632 | 0.8 | 0.8 |
| 2021 | 18 | 569 | 0.812 | 1.572 | +0.018 | 0.770 | 0.643 | 0.6 | 0.8 |
| 2022 | 19 | 573 | 0.861 | 1.622 | +0.023 | 0.791 | 0.657 | 0.8 | 0.7 |
| 2023 | 20 | 573 | 1.040 | 2.012 | +0.051 | 0.731 | 0.600 | 0.8 | 0.7 |
| 2024 | 21 | 564 | 1.076 | 2.116 | +0.039 | 0.733 | 0.604 | 0.4 | 0.7 |
| 2025 | 22 | 559 | 1.000 | 2.041 | +0.034 | 0.744 | 0.618 | 0.6 | 0.6 |

Full detail: `reports/season_prediction_metrics.csv`; every predicted-vs-actual player row:
`reports/pseudo_live_backtest_detail.csv`.

**A small, consistent positive bias appears in every single one of the 8 test seasons** (+0.014 to
+0.051 votes) — the model slightly over-predicts season totals on average, a real, systematic,
worth-disclosing pattern rather than random noise (8 of 8 seasons the same sign). Season rank
correlation is consistently strong (0.73-0.79 Spearman) with no clear degrading trend over time.
Top-10 inclusion (0.5-0.8) and especially top-5 inclusion (0.2-0.8, notably weak in 2018) show more
season-to-season volatility — the model is more reliable at getting the general ranking right than at
nailing the exact top-5/top-10 cutoff in any given season.

**The single largest miss found**: a player with 45 actual votes in 2024 was predicted at only 27.7
(error −17.3) — likely a historically exceptional individual season (45 votes is near-record territory)
that no model trained purely on match-level statistics could have flagged as that much of an outlier in
advance. Reported as a genuine limitation, not adjusted for — per the brief's explicit caution against
tuning based on isolated anecdotes. The largest overprediction (+14.1, a player predicted for 39.1 votes
who actually received 25) suggests the reverse case: a statistically dominant player whose performances
were, for whatever reason, less rewarded by actual voters than the model expected.
