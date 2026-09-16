# Changelog

## Unreleased

- Phase 1: repository scaffolding created, no data or code yet.
- Phase 1: `docs/DATA_SOURCE_AUDIT.md`, `docs/MODELLING_PLAN.md`, `docs/FEATURE_CANDIDATES.md`,
  `PROJECT_STATE.md` written.
- Phase 2: canonical CORE (1984-2025) and ADVANCED (2010-2025) player-match datasets built and
  validated against live AFL Tables pages; target-variable integrity confirmed with zero exceptions
  across 7,413 matches; identity-resolution and round-numbering bugs found and fixed; experimental
  2021-2026 event-level dataset (`torp`/`torpdata`) discovered, validated, and scoped as pilot-only;
  2026 umpire-stats mirror audit completed, correcting several Phase 1 "unavailable" findings.
  New docs: `docs/TARGET_VALIDATION.md`, `docs/DATA_COVERAGE.md`, `docs/2026_STATS_MIRROR.md`,
  `docs/EVENT_DATA_2021_AUDIT.md`. Updated: `DATA_DICTIONARY.md`, `PROJECT_STATE.md`,
  `docs/DATA_SOURCE_AUDIT.md` (Phase 2 addendum).
- Phase 3: versioned analytical feature table (148 columns) built on top of the untouched Phase 2
  canonical datasets -- match/team-relative features, teammate-competition features, a validated
  role classifier (real labels 2021-2025, 76.8% cross-validated statistical proxy 1999-2020), and
  6 transparent composite indices. Full descriptive/univariate exploratory analysis: role effects,
  temporal drift, nonlinearity, winner x margin interaction, and a formal leakage audit. Found and
  fixed a significant bug in event-chain score reconstruction (naive method: 11.6% match rate;
  corrected: 96.0%) via full-season-scale validation. New docs: `docs/FEATURE_REGISTRY.md`,
  `docs/EXPLORATORY_ANALYSIS.md`, `docs/ROLE_ANALYSIS.md`, `docs/TEMPORAL_DRIFT.md`,
  `docs/LEAKAGE_AUDIT.md`, `docs/PHASE3_DECISIONS.md`. No predictive model has been fit.
- Phase 4: resolved Phase 3 open issues (strictly-lagged role inference, lagged form features,
  event-data root cause, three separate CORE/ADVANCED/EXPERIMENTAL datasets). Built and rolling-origin
  backtested 4 model architectures (Benchmark logistic, Plackett-Luce ranking, GBM-utility,
  GBM-multiclass) across 22 folds/11 test seasons; Model 1 (Plackett-Luce, recent-8 window) wins on
  every tracked metric. Found and fixed a serious correctness bug (unstandardised mixed-scale features
  + non-log-space likelihood corrupting the ADVANCED-dataset fit) with 9 new regression tests
  (`tests/test_plackett_luce.py`); found and fixed a second bug in error-analysis segmentation. Ran and
  documented: feature ablation + 3 hypothesis-test pairs, reputation experiment (consistent modest
  improvement), game-state experiment (confirmed null result), calibration analysis (GBM badly
  miscalibrated, others excellent), error analysis by segment (defenders are the model's biggest blind
  spot), 8-fold stability analysis (`role_KEY_DEFENDER` is the single largest, most stable coefficient
  in the model), and an 8-season season-level pseudo-live backtest. New docs: `docs/MODEL_BACKTEST.md`,
  `docs/MODEL_COMPARISON.md`, `docs/FEATURE_ABLATION.md`, `docs/CALIBRATION.md`,
  `docs/ERROR_ANALYSIS.md`, `docs/REPUTATION_EXPERIMENT.md`, `docs/EXPERIMENTAL_GAMESTATE_MODEL.md`,
  `docs/PHASE4_DECISIONS.md`. No 2026 forecasts, Monte Carlo simulation, or leaderboard produced.
