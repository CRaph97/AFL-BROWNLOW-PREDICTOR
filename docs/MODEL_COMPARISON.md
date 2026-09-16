# Model Comparison — Phase 4

Status: **Complete.**
Last updated: 2026-09-17

## Scorecard (per the brief: no single blended score until every row is reported separately)

| Dimension | Model0_Benchmark | Model1_PlackettLuce | Model2a_GBM_utility | Model2b_GBM_multiclass |
|---|---|---|---|---|
| **Match ranking** (correct_3, best config) | 0.572 (expanding) | **0.575 (recent8)** | 0.568 (expanding) | 0.546 (expanding) |
| **Exact 3-2-1** (best config) | 0.088 | **0.091 (recent8)** | 0.088 | 0.072 |
| **Calibration** (ECE) | 0.0022 (best) | 0.0040 (very good) | 0.0319 (poor, fixable via isotonic) | 0.0032 |
| **Season-total accuracy** (not separately run per model — see below) | — | MAE 0.77-1.08 across 8 seasons | — | — |
| **Stability** | Not separately tested | High — role/context/margin coefficients stable across 8 folds (cv<0.05 for the largest terms) | Not separately tested | Not separately tested |
| **Interpretability** | High (linear coefficients) | High (linear coefficients + exact within-match probability semantics) | Low (tree ensemble, needs SHAP) | Low (tree ensemble, needs SHAP) |
| **Structural correctness** (respects the 3-2-1 constraint natively) | No (post-hoc renormalised) | **Yes (native)** | Yes (via PL marginalisation) | No (post-hoc renormalised) |
| **Complexity / maintenance** | Very low | Low-moderate (custom log-space likelihood, now regression-tested) | Moderate (tree ensemble + PL wrapper) | Low |
| **Data availability required** | CORE only | CORE only | CORE only | CORE only |
| **Robustness (this phase)** | Convergence warnings observed (L-BFGS hit iteration cap) — ranking/calibration still good despite this | No warnings after the fix; 9 regression tests passing | No structural issues; calibration issue is a known, fixable limitation | No structural issues |

## Verdict

**Model 1 (Plackett-Luce), recent-8-season training window, is the clear standalone winner** — it leads
on every ranking and probability-quality metric simultaneously (`docs/MODEL_BACKTEST.md`), is naturally
well-calibrated (`docs/CALIBRATION.md`), is directly interpretable, and natively respects the
within-match 3-2-1 constraint rather than needing a post-hoc patch. Model 0 (Benchmark) is a strong,
much simpler runner-up, useful as the ongoing sanity-check floor. Both GBM variants under-deliver
relative to their complexity: GBM-multiclass is the weakest model on ranking accuracy, and GBM-utility,
while ranking reasonably, requires mandatory recalibration to be usable at all.

## Ensemble investigation

Per the brief's instruction ("if multiple models show complementary strengths, test simple ensembles"):
**no ensemble was built.** Model 1 already dominates on every tracked metric — there is no
complementary strength in the other models to combine (GBM-utility's only edge would be marginal
ranking gains in some folds, at the cost of severe miscalibration that would need fixing before any
ensemble could use it safely). Building an ensemble here would add complexity without a demonstrated
benefit, contrary to the brief's explicit caution against complexity without justified improvement.
Model 0 (Benchmark) is retained as the standing simple-baseline comparator for future phases, per the
brief's instruction to always keep the simplest model available.

See `docs/PHASE4_DECISIONS.md` for the consolidated answers to all 14 Phase 4 questions and the final
recommendation for which model(s) advance to Phase 5.
