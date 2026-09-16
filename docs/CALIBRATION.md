# Probability Calibration — Phase 4, Section K

Status: **Complete.**
Last updated: 2026-09-17

## Method

Pooled, genuinely out-of-sample predictions from the main backtest (`data/processed/oos_predictions_core.parquet`
— expanding window, test seasons 2015-2025). For each model, P(3) is binned into 10 quantile-width
reliability bins and compared against observed frequency of actually receiving 3 votes (Expected
Calibration Error, ECE). Isotonic recalibration fit ONLY on 2015-2021 predictions, evaluated on
2022-2025 — never fit on the seasons it's evaluated against.

## Results

| Model | ECE (all seasons) | ECE (2022-25, raw) | ECE (2022-25, isotonic) | Isotonic improves? |
|---|---|---|---|---|
| Model0_Benchmark | 0.00223 | 0.00077 | 0.00210 | No |
| **Model1_PlackettLuce** | 0.00402 | 0.00241 | 0.00246 | No (already good) |
| Model2a_GBM_utility | **0.03187** | **0.02974** | 0.00198 | **Yes, dramatically** |
| Model2b_GBM_multiclass | 0.00321 | 0.00274 | 0.00168 | Yes, mildly |

## Interpretation

**Answer to Q11 ("which model is best calibrated?"): Benchmark and Plackett-Luce are both naturally
very well calibrated already** (ECE < 0.005, i.e. predicted P(3) is within half a percentage point of
observed frequency on average) — Benchmark is marginally better on this specific metric, but both are
in the same tier and isotonic recalibration doesn't meaningfully improve either (for Plackett-Luce, ECE
is essentially unchanged 0.00241→0.00246; for Benchmark, isotonic actually makes it *worse*,
0.00077→0.00210 — the raw probabilities were already better than what a coarse post-hoc correction
could achieve).

**GBM-utility is dramatically miscalibrated** — ECE roughly 8-14x worse than every other model — which
directly explains its poor log loss/Brier scores in the main comparison (`docs/MODEL_BACKTEST.md`).
Isotonic regression fixes most of this (0.0297→0.00198, better than any other model's *raw* ECE),
confirming the problem is genuinely a calibration issue (the ranking/ordering information is fine) not
a fundamental modelling failure — but this means GBM-utility's raw probabilities must never be reported
as-is; they would need mandatory post-hoc recalibration before any real use.

**Practical implication for Phase 5**: if Model 1 (Plackett-Luce) is carried forward as recommended
(`docs/MODEL_COMPARISON.md`), no calibration correction is needed — its probabilities can be reported
directly. If a GBM-based model is ever used instead, isotonic recalibration (fit on a held-out window,
never the evaluation season) is a hard requirement, not an optional refinement.
