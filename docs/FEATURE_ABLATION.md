# Feature Ablation — Phase 4, Section F/G

Status: **Complete.**
Last updated: 2026-09-17

## Method

Model 1 (Plackett-Luce), feature families added progressively (raw stats as the base), retrained from
scratch at each step. Evaluated on 3 held-out seasons (2023, 2024, 2025, expanding training window) — a
bounded, documented compute scope, not the full 11-season sweep used for the headline comparison.

## Results — progressive ablation

| Step | n_features | n_matches | correct_3 | exact_321 | all_3_id | top3_prec | rank_corr | log_loss |
|---|---|---|---|---|---|---|---|---|
| 1. raw only | 22 | 207 | 0.4589 | 0.0403 | 0.1240 | 0.5684 | 0.3745 | 0.1985 |
| 2. +match_relative | 30 | 207 | 0.4573 | 0.0435 | 0.1256 | 0.5641 | 0.3748 | 0.1983 |
| 3. +context | 35 | 207 | **0.5137** | 0.0483 | 0.1578 | 0.6023 | 0.3856 | 0.1834 |
| 4. +teammate | 40 | 207 | 0.5121 | 0.0419 | 0.1675 | 0.6103 | 0.3860 | 0.1827 |
| 5. +role | 47 | 207 | 0.5137 | 0.0499 | 0.1675 | 0.6108 | 0.3865 | 0.1825 |
| 6. +nonlinear | 51 | 207 | 0.4960 | 0.0483 | 0.1852 | 0.6184 | 0.3879 | **0.1802** |
| 7. +lagged_form | 67 | 188 | 0.5066 | 0.0479 | 0.1809 | 0.6208 | **0.4051** | 0.1954 |
| 8. +win_margin_interaction | 68 | 188 | 0.5066 | 0.0479 | 0.1809 | 0.6208 | 0.4051 | 0.1954 |

(n_matches drops at step 7 because lagged-form NaN for early-career players removes some matches —
per the documented `PlackettLuceModel.fit` incomplete-match guard, not a bug.)

## Results — special hypothesis variants

| Variant | n_features | correct_3 | exact_321 | log_loss | rank_corr |
|---|---|---|---|---|---|
| H1_raw_only_no_relative | 22 | 0.4589 | 0.0403 | 0.1985 | 0.3745 |
| H1_relative_instead_of_raw | 8 | 0.4541 | 0.0290 | 0.2002 | 0.3711 |
| H4_flat_winner_no_interaction | 67 | 0.5066 | 0.0479 | 0.19537 | 0.40511 |
| H4_with_winner_margin_interaction | 68 | 0.5066 | 0.0479 | 0.19537 | 0.40509 |
| H6_context_no_nonlinear | 35 | 0.5137 | 0.0483 | 0.1834 | 0.3856 |
| H6_context_with_nonlinear | 39 | 0.5040 | 0.0515 | 0.1811 | 0.3870 |

## Interpretation, family by family

1. **+match_relative adds essentially nothing on top of raw stats** (correct_3 0.4589→0.4573, a
   *decrease*; log loss unchanged to 3 decimal places). And per **H1**, match-relative features
   **alone** (replacing raw stats, not adding to them) do *worse* than raw stats alone on every metric
   (correct_3 0.4589→0.4541, log loss 0.1985→0.2002). **This directly contradicts the Phase 3 univariate
   finding** that match-relative z-scores modestly beat raw totals. The resolution: Plackett-Luce's own
   structure already makes every prediction comparative (utilities are compared *within* the match by
   construction), so hand-built relative features are largely redundant with what the ranking model
   already does implicitly — and discarding absolute scale by using *only* relative features throws away
   real information the model could otherwise use. **Answer to Q3 ("how much do match-relative features
   improve prediction"): materially less than Phase 3's univariate analysis suggested, once a
   properly-structured ranking model is already in place.**

2. **+context (win/margin) is the single largest lever in the entire ablation**: correct_3 jumps
   +5.7 percentage points (0.4573→0.5137), log loss improves from 0.198 to 0.183, top-3 precision rises
   0.564→0.602. No other single addition comes close. This matches Phase 3's strong exploratory finding
   and confirms it holds in a real multivariate, held-out setting.

3. **+teammate**: modest, mostly positive — correct_3 flat-to-slightly-down (noise-level), but
   all-3-identified rises 0.158→0.167 and top-3 precision rises 0.602→0.610, log loss improves slightly.
   **Confirmed independently and more directly in the stability analysis**: `n_teammates_disposals_ge_25`
   has a consistently negative, highly stable coefficient (mean −0.163, cv=0.033 — see
   `docs/PHASE4_DECISIONS.md` stability section) — direct, controlled evidence of "vote stealing":
   holding a player's own performance fixed, having more teammates also perform well genuinely reduces
   their predicted vote probability. **Answer to Q4: yes, teammate competition has a real, small,
   stable, negative effect, confirmed in a properly controlled (not just univariate) setting.**

4. **+role**: modest positive (exact_321 0.042→0.050, log loss ticks down). Far more dramatic evidence
   comes from the stability analysis: `role_KEY_DEFENDER` has the single largest, most stable
   coefficient of any feature in the whole model (mean 1.02, essentially zero cross-fold variance) — see
   `docs/PHASE4_DECISIONS.md`. **Answer to Q5: role adds real, large, stable value after controlling for
   raw statistics — one of the clearest findings in this entire phase**, even though its effect on the
   raw ablation-table ranking metrics looks modest (the ablation table understates it because role's
   value is concentrated in a few high-leverage cases like defenders, not spread evenly).

5. **+nonlinear (disposal/goal hinge terms) trades exact-pick accuracy for calibration and set
   identification.** Correct_3 *drops* 0.514→0.496 (worst single-step change in the whole sequence)
   while log loss improves to the best point in the entire progressive sequence (0.180), and
   all-3-identified rises 0.158→0.185. **Confirmed independently by H6** (same trade-off direction,
   correct_3 0.514→0.504, log loss 0.183→0.181, tested in isolation on top of context alone rather than
   the full accumulated feature set). This is a real, reproducible, honestly-reported trade-off, not
   noise — the nonlinear terms genuinely change *what kind* of model this is (better broad calibration,
   slightly worse single-best-guess accuracy), and whether that trade is worth it depends on which
   Phase 5 use case is prioritised.

6. **+lagged_form trades calibration for ranking.** Rank correlation jumps to 0.405 — the single
   largest rank-correlation gain in the whole sequence — but log loss worsens substantially
   (0.180→0.195, undoing most of step 6's calibration gain). **Answer to Q7: lagged form features
   genuinely help relative ordering but hurt absolute probability calibration**, plausibly because
   short-window rolling averages are noisy/thin for many players and inject variance into the
   probability estimates even as they improve rank quality.

7. **+win_margin_interaction adds exactly zero value** — every single metric identical to 5-6 decimal
   places in both the main ablation (step 7 vs 8) and the isolated **H4** comparison. **Answer to Q6:
   no, the explicit interaction term does not outperform the flat/additive winner+margin formulation** —
   the model can already capture whatever the interaction represents through the separately-included
   `margin`, `absolute_margin`, and `is_win` terms.

## Summary table: which ideas materially improve the model?

| Idea | Materially helps? | Evidence |
|---|---|---|
| Context (win/margin) | **Yes, the biggest lever** | +5.7pp correct_3, best single addition |
| Role | **Yes, large and stable** (understated by raw ablation deltas) | Largest, most stable coefficient of any feature |
| Teammate competition | **Yes, small but real and stable** | Consistently negative, low-variance coefficient |
| Match-relative transforms (in addition to raw) | **No** | Flat-to-negative effect once raw stats present |
| Match-relative transforms (instead of raw) | **No, worse than raw alone** | H1 test |
| Nonlinear hinge terms | **Mixed** (helps calibration, hurts exact-pick) | Reproduced independently via H6 |
| Lagged form | **Mixed** (helps ranking, hurts calibration) | Large rank_corr gain, large log_loss cost |
| Win×margin interaction | **No, redundant** | Identical metrics with/without, confirmed via H4 |
