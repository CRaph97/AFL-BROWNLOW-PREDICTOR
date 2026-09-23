# 2027 Brownlow model — freeze specification

What is frozen before 2027 Round 1, and how. No 2027 model is trained until the 2027 input rows exist
(the season has not started); this document fixes every choice so training is a mechanical step.
Audit evidence: `data/experiments/analysis/audit_*.csv`, `docs/2027_MODEL_R&D_RESULTS.md`.

## 1. Feature schema

- Store: `data/features/player_match_features.parquet`, builder `src/features/point_in_time.py`,
  `BUILD_VERSION = 1.0.0`, registry `data/features/feature_registry.json` (17 families, 225 columns,
  timing classes). The 2027 build must produce byte-identical columns for 2003–2026 rows (hash of
  `build_manifest.json` input hashes recorded) and append 2027 rows.
- Candidate feature sets are the registry families, never hand lists:
  - A (Structural): raw, match_relative, match_relative_ext, team_relative, context, teammate,
    dominance, nonlinear, role, role_interactions, lagged_form, baseline_relative, team_strength,
    reputation_pit (210 columns). Role families are retained for now despite a neutral-to-negative
    ablation (see results §6) because they are inputs to the ensemble weights already validated; a
    role-free A is an *investigate* item, not a freeze change.
  - B (Performance ML): same 210 columns.
  - C (Stats-only): the same_match families only (154 columns), guarded by `assert_stats_only`.
  - Excluded everywhere: `reputation_legacy` (same-season unrevealed votes), `era`, `advanced_stats`
    (2015+ only; no log-loss gain).

## 2. Model definitions

| | Class | Fixed hyperparameters | Window |
|---|---|---|---|
| A | `src/models/structural/pl_model.StructuralPL` (Plackett-Luce, analytic gradient `fast_pl.py`) | l2 = 1.0, L-BFGS maxiter 500, train-median imputation | expanding, 2003–2026 |
| B | `src/models/performance_ml/ranker.PerformanceRanker` (XGBoost 2.x `rank:ndcg`, match = query) | max_depth 6, n_estimators 400, learning_rate 0.05, min_child_weight 20, subsample 0.8, colsample 0.8, reg_lambda 5, lambdarank topk / 8 pairs; temperature fit on the last training season (2026) then refit on all | expanding |
| C | `src/models/stats_only/learned_pl.StatsOnlyPL` | as A | expanding |
| Ensemble | `src/models/ensemble/stacking` log-linear pool of P3 with weights fit by minimising OOF P3 log loss on seasons ≤ 2026 (walk-forward OOF files `data/experiments/oof/`) | weights refit once, recorded in `ensemble_weights.csv` (2026 row: A 0.387, B 0.609, C 0.004) | — |
| Calibration | Walk-forward Platt (logistic on logit P3, renormalised within match), fit on OOF seasons ≤ 2026 | `LogisticRegression(C=1e6)` | — |

Headline 2027 probability model: **Platt-calibrated learned ensemble**. Primary standalone: **B**.
Structural challenger: **A** (headline 3-vote pick model, reported alongside). Stats-only challenger:
**C** (benchmark only). Wheelo: external corroboration and evaluation benchmark only, never a target
or ensemble input.

## 3. Seeds and determinism

`SEED = 20270101` for XGBoost (`random_state`), metrics bootstraps and the simulator
(`src/simulation/season_sim.SEED`). Thread limits `OMP/OPENBLAS/MKL/VECLIB = 1` in every fit
process. Plackett-Luce fits are deterministic (tested). XGBoost `hist` with fixed seed and
`n_jobs = 2` is deterministic for a fixed thread count; the frozen B must record `n_jobs`.

## 4. Simulation interface

`src/simulation/season_sim.simulate_matches(preds, n_sims, seed)` takes any match-level frame with
coherent P3 (columns `match_id, round, player_id, player_name, team_id, p3`) and returns
`(totals, players, votes_by_round)`; `summarise`, `h2h`, `exact_order`, `team_leader`,
`clinch_round` sit on top. Production 2027 run: n_sims = 100,000, seed 20270101. Validated against
the frozen 2026 simulation (`validate_against_frozen`).

## 5. Metadata to record at freeze

`data/deployment/2027_model_manifest.json` with: git commit, feature `BUILD_VERSION` and input hashes,
experiment ids of A/B/C/ensemble (registry: `A_structural_pl__*`, `B_performance_xgb_rank_tuned__*`,
`C_stats_only_pl__*`), hyperparameters, training seasons (2003–2026), ensemble weights, Platt
coefficients, seeds, xgboost / scikit-learn / numpy versions, and the SHA-256 of every output.

## 6. Data cutoff rules

- Training labels: seasons 2003–2026 only; 2026 labels from `data/actual/` (label_source recorded).
- 2027 rows are built from the same feature store code with no 2027 votes present; every
  prior_matches / prior_seasons feature uses only data dated before the match.
- Role for 2027 rows: lagged proxy only (no torpdata labels), identical to 2026 handling.
- No model is refit during the season; per-round forecasts re-run inference only.
- Freeze date: before 2027 Round 1; the frozen output directory is immutable thereafter and the
  post-season evaluation reuses `src/evaluation/build_2026_evaluation.py`'s method.
