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
- Phase 5: produced the full 2026 Brownlow production forecast. Extended CORE/ADVANCED with the
  complete, real 2026 home-and-away season (207 matches; additive files only, Phase 1-4 outputs
  untouched). Built 4 scenarios on Phase 4's validated Plackett-Luce architecture (historical /
  recent-era / stats-assisted / structural-break sensitivity bands) to address the 2026
  umpire-statistics rule change -- a genuine structural break with no historical precedent -- combined
  into a documented, non-uniform ensemble (0.45/0.20/0.35). Ran a recent-history window comparison
  (recent5 confirmed best, second independent confirmation of Phase 4's recency finding), a
  reputation on/off comparison, and a 100,000-simulation Monte Carlo season simulation (generative
  mixture sampling, exact 6-vote-per-match conservation verified in all 100,000 sims). Found and fixed
  two real methodological issues: (1) utility-space ensembling was mechanically flattening
  probabilities -- switched to a probability-space linear opinion pool; (2) reputation/season-to-date
  features cannot exist for an unrevealed season -- fixed via a documented freeze-at-end-of-last-season
  proxy for reputation, and disclosed (not patched) as a genuine 6-of-207-match gap for the CORE
  lagged-form season-to-date features. Headline result: Nick Daicos projected as the 2026 leader
  (~44.9 EV), the most scenario-stable top-10 projection; Zak Butters is the most structural-break-
  sensitive top-10 player. All 19 Phase 1-4 tests still pass. New docs: `docs/2026_DATA_VALIDATION.md`,
  `docs/2026_MODELLING_METHODOLOGY.md`, `docs/2026_STRUCTURAL_BREAK.md`, `docs/2026_FINAL_REPORT.md`,
  `docs/2026_CONTENDER_ANALYSIS.md`, `docs/PHASE4_FINAL.md`. Not committed to git per instruction.
- Phase 5 audit: targeted production audit of the 2026 forecast. Fixed the 6 unscored matches
  (a season-to-date lagged-form feature is undefined-by-construction for a player's first game
  of a season; carried forward each player's last-completed-season value instead, using no
  future 2026 data) -- all 207/207 matches now score, season total exactly 1,242.0 (207 x 6),
  re-verified with 100,000 Monte Carlo runs. Audited Nick Daicos's revised 44.9 -> 47.2 EV
  (the increase is mechanical -- his Round 1 was one of the fixed matches -- and the projection
  survived the audit: broad accumulation across 19/22 matches, no probability
  over-concentration, consistent with his real 2022-2025 vote trajectory). Audited top-10
  probability concentration (nothing unusual), quantified defender-bias impact on 2026 (58/90
  elite defensive performances league-wide get EV<0.5; 15 specific games listed, not
  corrected), and re-ran the structural-break sanity check (Zak Butters remains the most
  sensitive player). Full integrity check passed (zero duplicates, exact probability-sum and
  season-total conservation, no NaN/Inf). All 19 project tests still pass. New docs:
  `docs/2026_FINAL_AUDIT.md`. New reports: `reports/2026_daicos_round_by_round.csv`,
  `reports/2026_top10_probability_audit.csv`; existing 2026 leaderboard/simulation outputs
  refreshed in place. Not committed to git per instruction.
- 2026 review dashboard: read-only Streamlit dashboard (`app.py`, `dashboard/data.py`,
  `pages/*.py`) over the frozen, audited 2026 production outputs -- no retraining, no weight
  changes, no altered predictions. Landing page (summary cards + Top 20), Player Detail
  (round-by-round, cumulative EV chart, auto-classified significant games), Scenario
  Comparison, Model Disagreement (season-level from the production leaderboard, plus a
  round-level scenario-spread view derived directly from
  `data/processed/scenario_predictions_2026.parquet`), a local session-only Brownlow Night
  Tracker, Round View, Match Detail (with a defender-bias warning banner), Defender Bias
  Watchlist, a generic Projection Concentration view (not hardcoded to any one player), and an
  Uncertainty page separating simulation/model-disagreement/structural-break uncertainty. Added
  8 lightweight integrity tests (`tests/test_dashboard_data.py`) verifying the dashboard never
  changes a probability or vote total; all 27 project tests (19 modelling + 8 dashboard) pass.
  New doc: `docs/2026_BROWNLOW_REVIEW_REPORT.md`. `streamlit`/`plotly` added to
  `requirements.txt`. Not committed to git per instruction.
- 2026 round-label integrity fix: user-reported evidence (Geelong vs Collingwood 2026-05-09
  and Brisbane vs Geelong 2026-05-14 shown one round later than official) traced to a real gap
  in the Phase 1/2 round-numbering finding -- afltables' raw `Round` column labels 2026's split
  Opening Round as Round "1" and numbers every subsequent round +1 relative to the AFL's
  official round number. Classified as **display/label-only** (bug class A): grep-verified no
  join/sort/feature anywhere uses `round`; the fully-rebuilt post-fix leaderboard is numerically
  identical to the pre-fix leaderboard to 14 decimal places, proving the label change altered no
  prediction. Fixed at the root (`src/data/round_normalization_2026.py`,
  `build_2026_extension.py`) and propagated by rebuilding the entire 2026 pipeline (features,
  scenario training, ensemble, Monte Carlo, outputs) from the corrected source -- not by
  hand-patching individual files (an early hand-patch attempt had a real double-application bug,
  caught before touching any deliverable, and was deleted in favour of the clean rebuild). All
  207 matches reconciled against the raw fixture in `reports/2026_match_identity_audit.csv`
  (207/207 on identity, round, score, and prediction checks). 10 new regression tests
  (`tests/test_2026_match_integrity.py`); all 37 project tests pass. New doc:
  `docs/2026_ROUND_INTEGRITY_AUDIT.md`; addendum added to `docs/DATA_SOURCE_AUDIT.md`. 2023-2025
  likely carry the same mislabeling in validated Phase 2/4 data (round is never a feature, so no
  Phase 4 result is affected) -- flagged as an unactioned follow-up requiring separate approval.
  Not committed to git per instruction.
  **Process note:** discovered mid-fix that a prior fork committed the initial 2026 production
  model and first audit pass to git (`b8a6c52`) despite explicit "do not commit" instructions in
  both briefs -- flagged to the user; not reversed here since undoing a commit is a decision
  outside this fix's scope.
