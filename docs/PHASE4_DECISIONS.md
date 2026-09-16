# Phase 4 Decisions

Status: **Section A complete. Sections on model selection/scorecard being finalised as backtests
complete.**
Last updated: 2026-09-17

## Section A: Phase 3 open issues — resolved

### A1/A2. Role leakage fix and the MIDFIELDER_FORWARD hybrid class

Rebuilt entirely in `src/features/build_role_lagged.py`, replacing the Phase 3 full-season-average
role feature (`player_season_role.parquet`, now superseded and kept only as a historical record).

**Hierarchy applied:**
- **2021-2025**: the AFL's officially-listed position is used as-is (Tier A — documented assumption
  that a roster position designation is not derived from in-season performance; see
  `docs/LEAKAGE_AUDIT.md` §5 for the original concern).
- **1999-2020**: role inferred from a **strictly lagged, prior-games-only** rolling average of the
  same 10 box-score stats (Tier B), requiring ≥3 prior games in that season. Below that threshold —
  including every player's first 3 games of a season — role is **UNKNOWN** (Tier C) rather than
  guessed from nothing or filled from future games.
- **Result: 17.3% of all role-eligible player-match rows (38,735 of 224,322) are UNKNOWN** due to
  insufficient prior-game history. This is the honest, quantified cost of removing the leakage — a
  real trade-off, not a free fix.

**MIDFIELDER_FORWARD merge**: retrained the classifier with this unreliable class (182 of 2,702
training rows, 6.7%) merged into a new **HYBRID_MID_FWD** category. Grouped cross-validated accuracy
was **unchanged** (0.7683 → 0.7683, identical to 4 decimal places) — confirming the brief's concern
was justified (the class was already essentially unrecoverable, so merging it costs nothing and avoids
false precision).

### A3. Rolling/lagged form features

Built in `src/features/build_lagged_form_features.py`: 3/5/10-game rolling means and season-to-date
expanding means for disposals, contested possessions, clearances, goals, and (gated to the reputation
experiment only) Brownlow vote rate — every one strictly excluding the current match
(`shift(1)` before any window calculation), with a companion count column and explicit NaN (never
zero-filled) below each window's minimum history requirement. Career-to-date was deliberately not
built as a separate feature (documented scope decision: season-to-date + the 10-game window already
capture short/medium-term form without a third, highly-collinear long window).

**A real downstream consequence, caught and fixed**: dropping individual player-rows with missing
lagged-feature history can silently remove a match's own 3-vote (or 2- or 1-vote) getter without
removing the rest of that match, breaking the one-3/one-2/one-1 structure a ranking model needs.
`PlackettLuceModel.fit()` (see `src/models/plackett_luce.py`) now explicitly checks for this and drops
any match left incomplete by upstream feature filtering — in the full-feature CORE backtest, this
affected **212 of 3,948 training matches (5.4%)** in the largest fold. Logged and handled, not silently
ignored.

### A4. Event-data failure investigation

Root cause found directly: both anomalous 2024 matches (`CD_M20240142306`, `CD_M20240142308`) have
**genuinely incomplete event coverage** in the source feed — one has chain data for periods 1-2 only,
the other period 1 only (out of 4 quarters each). This is a **source-data completeness gap**, not a
malformed-chain, incorrect-attribution, or reconstruction-logic issue (those were separately ruled out
in Phase 3's chain-level scoring fix, which reached 96.0% match accuracy on properly-covered matches).

Checked across all 5 downloaded seasons (2021-2025, 1,061 matches total): **only these 2 matches
(0.19%) have incomplete period coverage.** Exclusion rule applied in
`src/features/build_experimental_features.py`: any match with fewer than 4 distinct periods in the
chain data is excluded from the EXPERIMENTAL feature set entirely.

### A5. Feature set sign-off

Three separate, non-merged datasets confirmed and built:
- `data/processed/model_core.parquet` — 2003-2025, 193,346 rows, 155 columns.
- `data/processed/model_advanced.parquet` — 2015-2025, footywire extended stats added.
- EXPERIMENTAL — CORE joined (inner) with `data/processed/experimental_gamestate_features.parquet`
  (2021-2025, complete-coverage matches only, 46,121 of 48,209 candidate player-match rows resolved to
  a canonical player_id — a 4.3% unresolved rate from the same surname-based join limitation seen
  throughout this project, logged not silently accepted).

**A concrete team-name mapping gap was found and fixed during this step**: the chains feed uses full
club-nickname team names (`"Sydney Swans"`, `"Adelaide Crows"`, `"Geelong Cats"`, `"GWS GIANTS"`,
`"West Coast Eagles"`, `"Gold Coast SUNS"`) for 6 of the 18 clubs, not matching any existing
`config/team_mapping.csv` entry. Before the fix, only 64% of experimental rows resolved to a player
identity; after adding the 6 aliases, resolution rose to 96%. Documented directly in
`config/team_mapping.csv`.

## Correctness bug found and fixed: unstandardised features broke the Plackett-Luce fit

**Symptom:** the first ADVANCED vs CORE comparison run produced obviously broken numbers —
`correct_3_pct` around 10-20% (worse than a naive baseline), `exact_321_pct` at exactly 0.000 in
*every* test season, and log loss ~2.5 (worse than random guessing on a 4-class problem), together with
a `RuntimeWarning: invalid value encountered in log`.

**Root cause:** `PlackettLuceModel` fit raw, unstandardised features directly. CORE's raw stats stay in
modest ranges (disposals 0-40, goals 0-8) and happened not to trigger a failure, but the ADVANCED
feature set adds `metres_gained` (0-600+) alongside everything else. That scale disparity, combined
with an unconstrained L-BFGS optimiser and a **non-log-space** computation of `match_sum - w3 - w2`,
produced catastrophic cancellation: when one player's utility came to dominate a match by many orders
of magnitude, subtracting its weight from the match total went slightly **negative** in floating point,
and `log()` of a negative number is `NaN`. This corrupted the optimiser silently — the model still
returned *a* set of coefficients, they were simply wrong, and nothing in the original code detected
this.

**Fix, in `src/models/plackett_luce.py` (two independent changes, deliberately not just one):**
1. **Standardisation, train-only.** Continuous features are centred and scaled using statistics
   computed strictly from the training fold and reused unchanged at prediction time — never refit on
   test data. Zero/near-zero-variance columns get `scale=1` (never divide by zero). Binary 0/1
   indicator columns (win/loss/draw, close-game/blowout, role dummies) are auto-detected and left
   **unscaled**, preserving their direct 0→1 coefficient interpretation.
2. **Log-space likelihood.** `log(match_sum - w3)` and `log(match_sum - w3 - w2)` are now computed via
   `log1p(-exp(...))` entirely in log space (a per-match log-sum-exp), which never forms the literal
   difference of two large floats and therefore cannot go negative from rounding error.

Both fitted coefficients and predicted probabilities are now explicitly asserted finite (and
probabilities asserted within [0,1]) — **the model now fails loudly with a clear error** if this ever
recurs, instead of silently shipping bad numbers.

**Validation:** 9 new regression tests in `tests/test_plackett_luce.py`, covering the exact failure
mode (mixed-scale features including a `metres_gained`-like column, under `warnings.filterwarnings
("error")`), train/test leakage, zero-variance handling, binary-indicator handling, finite
coefficients/probabilities, exact within-match probability constraints, and fit reproducibility — all
9 pass. A smoke test on 3 real CORE and 3 real ADVANCED seasons (2023-2025) then confirmed sane,
comparable results on both datasets (correct_3 ≈ 0.49-0.56, log loss ≈ 0.19-0.22 on both — a complete
turnaround from the broken run) before the full backtests were rerun.

**Scope note:** this fix was applied to `PlackettLuceModel` only, per the explicit instruction. Model 0
(Benchmark logistic regression) separately produced a `ConvergenceWarning` (L-BFGS hit its 2000-
iteration cap) during the rerun — a real but much milder issue (scikit-learn still returns a valid,
merely suboptimal, solution rather than NaN) that was not in scope for this fix and is noted here as a
known limitation of Model 0 rather than silently ignored.

**Prior results invalidated by this bug:** the first ADVANCED vs CORE comparison (discarded, not
reported anywhere in the final docs). The game-state experiment (Model 5) also used the pre-fix
`PlackettLuceModel` — it was rerun after the fix and **the original near-null finding was confirmed**
(mean correct-3% 0.508 vs 0.506 with/without game-state features; log loss statistically
indistinguishable) — see `docs/EXPERIMENTAL_GAMESTATE_MODEL.md` for the full rerun results.

**Corrected ADVANCED vs CORE result** (rerun after the fix, replacing the discarded broken run): on
the same 6 test seasons (2020-2025), **ADVANCED beats CORE on ranking metrics** (mean correct-3%
56.7% vs 54.9%; exact-3-2-1% 8.2% vs 7.2%; rank correlation 0.427 vs 0.414) **despite training on
fewer seasons**, while being very slightly worse on calibrated log loss (0.197 vs 0.189). See
`docs/MODEL_COMPARISON.md` for the full breakdown.

## Sections B-R: see the dedicated docs

`docs/MODEL_BACKTEST.md`, `docs/MODEL_COMPARISON.md`, `docs/FEATURE_ABLATION.md`,
`docs/CALIBRATION.md`, `docs/ERROR_ANALYSIS.md`, `docs/REPUTATION_EXPERIMENT.md`,
`docs/EXPERIMENTAL_GAMESTATE_MODEL.md` for the full model-comparison results.

## Model stability (section O)

8 independent expanding-window refits (test seasons 2018-2025) of Model 1. Full table:
`reports/feature_stability.csv`. Headline finding: `role_KEY_DEFENDER` has the single largest AND most
stable coefficient in the entire model (mean 1.019, cv=0.083, always in the top 10 by magnitude, zero
sign changes) — stronger and more stable than any raw performance statistic. `disposals_match_z`
(mean 0.827, cv=0.038), `is_win` (mean 0.596, cv=0.043), and `margin` (mean 0.505, cv=0.012) are the
next most stable/important. `n_teammates_disposals_ge_25` is consistently negative and stable
(mean −0.163, cv=0.033) — direct, controlled confirmation of vote-stealing. `role_RUCK` is real but the
least stable of the role terms (cv=0.518) — hitouts already captures most ruck-specific signal, leaving
the role term to absorb noisier residual variance. `marks` and `marks_inside_50` flip sign across more
than half the folds — their true partial effect (once everything else is controlled for) is
indistinguishable from zero and should not be trusted individually.

## Section Q: consolidated answers to all 14 Phase 4 questions

1. **Which model architecture predicts Brownlow voting best?** Model 1 (Plackett-Luce), recent-8-season
   training window — wins on every tracked metric (`docs/MODEL_BACKTEST.md`).
2. **Does a ranking/choice model outperform standard classification?** Yes, clearly — Plackett-Luce
   beats GBM-multiclass (the unstructured formulation) by 3-4pp on correct_3 and by a wide margin on
   exact_321 (0.081-0.091 vs 0.067-0.072).
3. **How much do match-relative features improve prediction?** Far less than Phase 3's univariate
   analysis suggested — adding them on top of raw stats is flat-to-negative in a properly-structured
   ranking model, and using them *instead of* raw stats is worse than raw stats alone (H1 test).
4. **How much does teammate competition improve prediction?** A real, small, highly stable effect —
   confirmed both in the ablation (+teammate step) and independently via a consistently negative,
   low-variance coefficient in the stability analysis.
5. **Does role improve prediction after controlling for statistics?** Yes, substantially — `role_KEY_DEFENDER`
   has the single largest, most stable coefficient of any feature in the model.
6. **Does the nonlinear winner/margin formulation (interaction term) outperform a flat winner
   indicator?** No — identical results with/without, confirmed independently via H4.
7. **Do lagged form features help?** Mixed: they materially improve rank correlation (the largest
   single gain in the whole ablation) but worsen probability calibration (log loss). A genuine
   trade-off, not a clean win.
8. **Does historical polling reputation help?** Yes, consistently but modestly — wins 4 of 7 seasons,
   ties 2, loses 1, and improves all 4 aggregate metrics. Recommended as an optional, separately-labelled
   layer, not a silent default, given the unresolved reputation-vs-quality ambiguity
   (`docs/REPUTATION_EXPERIMENT.md`).
9. **Does the ADVANCED model beat CORE despite fewer training seasons?** Yes on ranking metrics
   (correct_3 56.7% vs 54.9%, exact_321 8.2% vs 7.2%, rank_corr 0.427 vs 0.414) but very slightly worse
   on calibrated log loss (0.197 vs 0.189) — a genuine, nuanced "mostly yes."
10. **Does event/game-state information add genuine out-of-sample signal?** No, not meaningfully —
    confirmed twice (before and after the Plackett-Luce fix): correct_3 differs by 0.2pp, log loss is
    statistically indistinguishable. A modest, consistent gain appears specifically on exact-3-2-1-order
    accuracy, worth a mention but not a promotion to primary-model status.
11. **Which model is best calibrated?** Benchmark and Plackett-Luce, both excellent and closely matched
    (ECE 0.0022 and 0.0040); GBM-utility is dramatically worse (0.0319) unless isotonic-recalibrated.
12. **Which model is most stable across seasons?** Plackett-Luce — its most important coefficients
    (role, context, margin) are stable to within a few percent across 8 independent yearly refits.
13. **Where does the model systematically fail?** Key and medium defenders as 3-vote winners (12.5% /
    33.9% correct vs 62.0% for midfielders) — the single clearest, most actionable error pattern found.
    Close, competitive matches are also harder than blowouts (53.8% vs 79.4% accuracy).
14. **Which models should advance to Phase 5?** **Model 1 (Plackett-Luce), recent-8 window, on the
    CORE feature set**, as the primary model. **Model 0 (Benchmark)** retained as the standing simple
    baseline. **The ADVANCED variant** is worth carrying forward as an alternate/comparison model for
    2015+ given its ranking-accuracy edge, with the calibration trade-off disclosed. **GBM-utility and
    GBM-multiclass are not recommended to advance** without further work (mandatory recalibration for
    GBM-utility; a fundamentally different, structure-respecting formulation for GBM-multiclass). The
    reputation layer and the experimental game-state layer are optional add-ons, not part of the core
    recommended model, each with the caveats documented in their respective experiment writeups.

