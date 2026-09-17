# Project State — AFL Brownlow Predictor

Last updated: 2026-09-17
Current phase: **Phase 5 (2026 Production Forecast) complete, pending user review. Not yet committed to git
per instruction — all Phase 5 changes are in the working tree for morning review.**
Phase 1+2 baseline committed as `034cbf0`, Phase 3 as `c42d46f`, Phase 4 as `c7d0d09`.

---

## Phase 5 summary (2026 production forecast — see docs/2026_FINAL_REPORT.md for the full detail)

**Strategic change**: 2026 introduced umpire-visible-statistics-assisted voting, a genuine structural
break with zero historical precedent. Rather than assuming old voting patterns transfer unchanged,
Phase 5 built 4 scenarios (historical / recent-era / stats-assisted / structural-break sensitivity
bands) on top of Phase 4's validated Plackett-Luce architecture (no new architecture introduced) and
combined them into a documented, non-uniform ensemble.

**Data**: the complete, REAL 2026 home-and-away season (207 matches, 25 rounds, 18 teams,
2026-03-05 to 2026-08-23) was already available in the same fitzRoy mirror used throughout this
project — no synthetic/proxy data was needed. Full validation: `docs/2026_DATA_VALIDATION.md`.

**Headline result**: Nick Daicos (Collingwood) is the projected 2026 Brownlow leader, ~44.9 expected
votes (95% Monte Carlo range 39-50), the single most scenario-stable top-10 projection. Zak Butters
(Port Adelaide) is the most structural-break-sensitive top-10 player (his rank materially depends on
how much 2026's rule change actually shifts umpire behaviour). Full leaderboard:
`reports/2026_leaderboard.csv`; contender detail: `docs/2026_CONTENDER_ANALYSIS.md`.

**Two real methodological findings surfaced during this phase** (both documented, not hidden):
1. Ensembling Plackett-Luce utilities by forcing them to unit variance before averaging is WRONG (it
   mechanically compresses/flattens the result) — fixed by switching to a probability-space linear
   opinion pool. See `src/models/build_2026_ensemble.py`'s docstring.
2. "Reputation"/"season-to-date" features cannot exist for a season whose votes are unrevealed (true
   for every season's Round 1, not just 2026) — fixed for reputation via a documented freeze-at-
   end-of-last-season proxy; left as an honest, disclosed 6-match (of 207) gap for the CORE lagged-form
   season-to-date features, since patching it would break consistency with Phase 4's validated training
   methodology. See `docs/2026_MODELLING_METHODOLOGY.md` §6b.

**All 19 Phase 1-4 tests still pass unchanged.** No Phase 1-4 file was modified; every Phase 5 output
is additive (new files only).

**Not done in this run** (documented limitations, not oversights): kick-ins/intercept marks/spoils
(3 of 17 confirmed umpire stats) are not scraped from the AFL's semi-public API; a defender-specific
model correction (Phase 4's single largest identified accuracy lever) was not attempted.

---

## Phase 4 summary (see docs/PHASE4_DECISIONS.md for the full decision log and all 14 question answers)

**Section A (Phase 3 issue resolution):** role-leakage fixed (strictly lagged, prior-games-only
inference; 17.3% of rows now honestly UNKNOWN rather than leaked), MIDFIELDER_FORWARD merged into a
broader hybrid category (accuracy unchanged, confirming the merge was safe), lagged form features
built, the two anomalous 2024 event-data matches root-caused (genuine incomplete source coverage,
excluded via a clean rule), three separate CORE/ADVANCED/EXPERIMENTAL datasets built.

**A major correctness bug found and fixed mid-phase**: `PlackettLuceModel` fit unstandardised,
mixed-scale features through a non-log-space likelihood, causing catastrophic cancellation and silently
corrupted results on the ADVANCED dataset (correct-3% ~10-20%, exact-3-2-1% exactly 0.000, log loss
~2.5 — all symptoms, not a real "ADVANCED loses" finding). Fixed via train-only feature standardisation
+ a fully log-space likelihood; 9 new regression tests added (`tests/test_plackett_luce.py`), all
passing; both the CORE backtest and ADVANCED comparison rerun cleanly after the fix. A second bug (error
analysis crashing when segmenting by player-level attributes) was also found and fixed. Full writeup:
`docs/PHASE4_DECISIONS.md`.

**Models built**: Model 0 (Benchmark logistic), Model 1 (Plackett-Luce, the primary structurally-correct
ranking model), Model 2a/2b (GBM utility / multiclass, using `HistGradientBoosting*` as an XGBoost
substitute after a native-library ABI issue). **Model 1 with a recent-8-season training window wins on
every tracked metric** (`docs/MODEL_BACKTEST.md`).

**Experiments completed**: full 22-fold rolling-origin backtest (`docs/MODEL_BACKTEST.md`), ADVANCED vs
CORE comparison (`docs/PHASE4_DECISIONS.md`), 8-season season-level pseudo-live backtest
(`docs/MODEL_BACKTEST.md` §7), feature ablation + 3 hypothesis-test pairs (`docs/FEATURE_ABLATION.md`),
reputation experiment (`docs/REPUTATION_EXPERIMENT.md`), game-state experiment, rerun and reconfirmed
after the fix (`docs/EXPERIMENTAL_GAMESTATE_MODEL.md`), calibration analysis (`docs/CALIBRATION.md`),
error analysis by segment (`docs/ERROR_ANALYSIS.md`), and an 8-fold model-stability analysis
(`docs/PHASE4_DECISIONS.md`).

### Headline findings

1. **Model 1 (Plackett-Luce), recent-8 window**: correct_3=57.5%, exact_321=9.1%, best log loss
   (0.183) and Brier score (0.088) of any model — a clean sweep, not a marginal win.
2. **Context (win/margin) is the single biggest feature-family lever** (+5.7pp correct_3) — bigger than
   any other addition in the whole ablation.
3. **Role adds large, stable value after controlling for statistics** — `role_KEY_DEFENDER` has the
   single largest, most stable coefficient of any feature in the model, yet the model still only
   correctly picks key defenders as the actual 3-vote winner 12.5% of the time (vs 62.0% for
   midfielders) — a coherent, well-quantified picture of a real, partially-but-not-fully-compensated
   bias.
4. **Teammate competition confirmed with a controlled, stable, negative coefficient** — not just a
   univariate pattern from Phase 3.
5. **Match-relative features add far less value than Phase 3's univariate analysis suggested** once a
   properly-structured ranking model is in place — a genuine, honest correction to an earlier finding.
6. **The win×margin interaction term is completely redundant** — confirmed twice, identical results
   with/without.
7. **Nonlinear terms and lagged-form features each trade one thing for another** (calibration vs
   exact-pick accuracy; ranking vs calibration respectively) rather than being unambiguous wins.
8. **Reputation (lagged prior vote-rate) helps consistently but modestly** (4 wins/2 ties/1 loss across
   7 seasons, better on all 4 aggregate metrics) — recommended as an optional, separately-labelled
   layer given the unresolved reputation-vs-quality interpretation ambiguity.
9. **ADVANCED beats CORE on ranking despite fewer training seasons**, but is very slightly worse
   calibrated — a genuine nuanced trade-off, not a clean win either way.
10. **Event/game-state features do not add meaningful signal** — confirmed twice (before and after the
    Plackett-Luce fix), with a small, consistent exception on exact-3-2-1-order accuracy specifically.
11. **GBM-utility is dramatically miscalibrated** (ECE 8-14x worse than other models) despite reasonable
    ranking — fixable via isotonic recalibration but not usable as-is.

### Recommendation for Phase 5

**Model 1 (Plackett-Luce), recent-8-season window, CORE feature set** as the primary model to convert
into calibrated match-level vote distributions and season simulations. Benchmark retained as the
standing baseline. ADVANCED variant carried forward as an alternate for 2015+. GBM variants, the
reputation layer, and the experimental game-state layer are not part of the core recommendation without
further work (see `docs/PHASE4_DECISIONS.md` Q14 for the full reasoning).

---

## Phase 3 summary (see docs/PHASE3_DECISIONS.md for the full decision log)

**New code:** `src/features/` — 4 `build_*.py` feature-construction modules (relative, context,
teammate, composite indices), 2 role modules (real-label reference + validated statistical proxy),
1 assembly script producing the versioned analytical dataset, 4 `analyze_*.py` descriptive-analysis
scripts, and `event_feasibility.py` (experimental, not merged into the trainable dataset).

**New data:** `data/processed/analytical_features_v1.parquet` (320,861 rows x 148 columns, additive —
the Phase 2 canonical CORE/ADVANCED tables are untouched), `player_season_role.parquet`,
`role_reference_2021_2025.parquet`.

**New reports (`reports/`):** `feature_availability.csv`, `univariate_vote_relationships.csv`,
`role_vote_summary.csv`, `season_vote_relationships.csv`, `correlation_matrix.csv`,
`candidate_feature_sets.csv`, `role_proxy_confusion_matrix.csv` + classification report,
`event_score_reconciliation_2024.csv`, `event_game_state_sample_match.csv`, plus 2 logged-for-review
exception files (role-reference surname collisions).

**New docs:** `docs/FEATURE_REGISTRY.md`, `docs/EXPLORATORY_ANALYSIS.md`, `docs/ROLE_ANALYSIS.md`,
`docs/TEMPORAL_DRIFT.md`, `docs/LEAKAGE_AUDIT.md`, `docs/PHASE3_DECISIONS.md`.

### Answering the Phase 3 required questions (section Q)

1. **Strongest raw-stat correlates:** SuperCoach/AFL Fantasy points (Spearman 0.37/0.36) — stronger
   than any single raw box-score stat — followed by `possession_impact_index`, disposals, effective
   disposals, score involvements, contested possessions.
2. **Match-relative vs raw:** match-relative disposal/contested-possession z-scores modestly but
   consistently beat their raw equivalents (~0.01-0.02 correlation units).
3. **Winner effect:** strong (8-14pp polling-rate gap).
4. **Winner effect x margin:** yes, dramatically and asymmetrically — winners poll more as margin
   grows (8.3%→13.5%), losers collapse to near-zero (5.2%→0.05%).
5. **Teammate competition:** substantial — near 3x reduction in polling rate (27.8%→~9-10%) for an
   identical own-performance band as more teammates also perform well.
6. **Nonlinearities:** disposals and goals both strongly convex — the 30→35 disposal jump is worth
   ~6x the 15→20 jump in polling-rate terms; 3-vote rate more than doubles per additional goal from 4
   to 6.
7. **Interactions worth testing:** performance x role, performance x win/margin, performance x
   teammate competition.
8. **Era drift:** winner effect flat across 1999-2025; clearances' vote-association has risen ~30-40%
   relatively since the 2000s; goals show a mild recent decline.
9. **Role effects:** very large — midfielders poll at 7.4x key defenders' rate and capture 64.5% of
   all 3-vote games.
10. **Redundant features:** 18 pairs at |r|>=0.85, notably clearances<->stoppage_clearances (0.91),
    disposals<->effective_disposals (0.92), afl_fantasy_points<->supercoach_points (0.86).
11. **CORE/ADVANCED/EXPERIMENTAL sets:** proposed in `docs/PHASE3_DECISIONS.md` and
    `reports/candidate_feature_sets.csv` (109 CORE+ADVANCED features, 19 ADVANCED-only, 6 excluded as
    metadata).
12. **Features to discard:** metadata-only companions, several redundant pairs — see
    `docs/PHASE3_DECISIONS.md` §4.
13. **Does torpdata support credible game-state features?** Yes, but only after fixing a real bug
    found during this audit (see below) — corrected chain-level reconciliation reaches 99.1% (goals)
    / 96.0% (behinds) exact match against official scores across 215 2024 matches.

### The most important Phase 3 finding: a real bug caught by "verify at scale"

The single-match spot check in Phase 2's `docs/EVENT_DATA_2021_AUDIT.md` implied roughly an 83% behind
reconstruction rate. Checking all 215 2024 matches (not just one) revealed the naive method actually
achieves only **11.6%** — because a chain's `final_state` field is copied onto every action row in that
chain, not just one, so naive row-counting massively over-counts rushed behinds. Root-caused and fixed
(chain-level deduplication + correct team-attribution rule for `rushedOpp`), bringing the corrected rate
to **96.0%**. This is now documented precisely in `docs/EXPLORATORY_ANALYSIS.md` §10 and the fixed logic
lives in `src/features/event_feasibility.py`. Two of 215 matches still show large, undiagnosed
discrepancies — flagged as an open item, not silently averaged away.

### Open items for Phase 4 (from docs/PHASE3_DECISIONS.md §5-6)

1. Role features must be rebuilt as season-to-date-only (not full-season aggregate) before any
   forward-looking model use — currently a documented CONDITIONAL leakage risk, safe only for the
   retrospective analysis done in Phase 3.
2. The MIDFIELDER_FORWARD role-proxy class is statistically unrecoverable pre-2021 (F1 = 0.06) —
   needs a decision on how to handle it.
3. Two 2024 matches with large event-data reconciliation failures warrant a quick follow-up look.
4. No multivariate model has been fit yet — every Phase 3 finding is univariate/descriptive; redundancy
   and interaction effects that look large individually may overlap once modelled jointly.

---

---

## What exists right now

- A working Python environment (`.venv/`, `requirements.txt` — pinned, minimal: pandas, pyarrow,
  pyreadr, requests, lxml/bs4/html5lib, pytest. No modelling libraries yet, per the "add only when
  needed" decision.)
- Real, validated data:
  - `data/raw/fitzroy_data/` — downloaded via `src/data/fetch_fitzroy_data.py`, with SHA-256 provenance
    in `PROVENANCE.json`.
  - `data/raw/torpdata_pilot/` — one season (2024) of the experimental event-level dataset, downloaded
    for validation only.
  - `data/interim/player_match_afltables_1984_2025.parquet` — unfiltered-column intermediate.
  - `data/processed/player_match_core_1984_2025.parquet` — **the canonical target-bearing dataset**,
    320,861 rows, 7,413 matches, 1984-2025, home-and-away only.
  - `data/processed/player_match_advanced_2010_2025.parquet` — CORE plus footywire advanced columns.
- `config/team_mapping.csv` — the explicit, documented team-identity canonicalisation table.
- `src/data/` — four reproducible scripts: `fetch_fitzroy_data.py`, `build_core_dataset.py`,
  `build_advanced_dataset.py`, `build_coverage_matrix.py`. Re-running them regenerates everything above
  from raw data.
- `tests/test_core_dataset_integrity.py` — 10 automated QC checks, **all passing**.
- `reports/` — machine-readable outputs: `coverage_matrix.csv` / `coverage_matrix_wide.csv` /
  `coverage_first_reliable_season.csv`, plus two logged-for-review exception files
  (`identity_review_missing_id.csv`, `advanced_join_surname_collisions.csv`).
- Nine documents: the four from Phase 1 (now with a Phase 2 addendum on `DATA_SOURCE_AUDIT.md`) plus
  five new ones: `docs/TARGET_VALIDATION.md`, `docs/DATA_COVERAGE.md`, `docs/2026_STATS_MIRROR.md`,
  `docs/EVENT_DATA_2021_AUDIT.md`, and an updated `DATA_DICTIONARY.md`.

Nothing has been committed to git yet (only requested when you ask).

---

## Answering the Phase 2 stop-condition questions

**1. Were Brownlow votes successfully joined to real player-match records?**
Yes. MATCH → PLAYER → PLAYER MATCH STATISTICS → BROWNLOW VOTES is established and reproducible for
**1984-2025 home-and-away matches** (320,861 rows, 7,413 matches). Earlier seasons exist in the raw data
but have no match-level vote detail (see #1 below) — this is a real historical limit, not a build choice.

**2. Validation examples**
Three matches spanning three eras (1991, 1992, 2024) were checked line-by-line against **live AFL
Tables pages**, not just re-reads of the same scrape — every field matched exactly, including a
same-team, same-surname pair (the Daniher brothers, 1992) correctly disambiguated by stable player ID.
Two independent season-total facts (Dangerfield 2016 = 35 votes, Neale 2020 = 31 votes from 17 games)
were cross-checked against cited public reporting and matched exactly. Full detail: `docs/TARGET_VALIDATION.md`.

**3. Discovered integrity problems**
- A pandas gotcha where missing player IDs (NaN) falsely appeared as "duplicate players" — found, fixed,
  covered by a regression test.
- Round numbering differs between afltables and footywire/the AFL's own API by up to one round per
  season (Opening-Round-as-Round-0 offset) — found via a real example, now a hard rule in all join code
  (always join on date, never round number).
- A genuine historical anomaly (a 1928 finals replay sharing one round label across two dates) — outside
  the modelling window, documented rather than silently absorbed.
- A confirmed ~4% (96.1% match rate) shortfall joining footywire's advanced stats onto the core table,
  concentrated in substitute-flagged players — quantified and logged, not hidden.
- A confirmed ~17% under-count reconstructing behinds (not goals) from the experimental event dataset's
  naive event descriptions, and a null-column bug in one of that dataset's convenience fields — both
  documented with the specific fix needed before anyone relies on them.

**4. Final source choices**
Bulk layer: the community `fitzRoy` data repository (MIT-adjacent "with permission" sourcing from AFL
Tables + Footywire), not bespoke scraping — per your Phase 1 decision. Every critical fact (target
variable, identifiers, scores) independently re-verified against afltables.com directly, not trusted
blindly. Experimental event layer: `peteowen1/torp`/`torpdata` (MIT-licensed, actively maintained),
which superseded the single-season candidate found in Phase 1.

**5. Feature coverage by season**
Full matrix in `reports/coverage_matrix.csv`. Clean, sharp transition seasons found for every feature —
no partial-coverage "in-between" seasons, only a lower-common-stat era (1984+) and progressively richer
eras arriving in 1987, 1998, 1999, 2003, 2010, and 2015. Full table in `docs/DATA_COVERAGE.md` §2.

**6. Recommended modelling start years**
- **CORE model: 2003-2025** (23 seasons) — every common box-score stat available with no gaps.
  1984-2002 usable as a longer, thinner-featured extension if wanted.
- **ADVANCED model: 2015-2025** (11 seasons) — the full extended stat set (score involvements, metres
  gained, intercepts, centre/stoppage clearance split, tackles inside 50, turnovers) only reliable from
  2015. This is a starting point for the rolling-origin comparison your Phase 1 decision already called
  for (recent-5/8/10, expanding window, recency-weighted) — not a final answer, since that requires
  actual backtesting in Phase 3.

**7. Is the 2021 event dataset credible?**
The original single-season candidate: no. A materially better replacement was found and validated
instead: `torp`/`torpdata`, MIT-licensed, actively maintained, covering **2021-2026** (six seasons, not
one). Validated against our own already-confirmed ground-truth match with exact agreement on
disposals/kicks/handballs/goals, a specific and fixable ~17% behind-undercount, and one confirmed
column-level bug. Verdict: **valid enough for the experimental pilot track**, not a CORE/ADVANCED
data source. Full detail: `docs/EVENT_DATA_2021_AUDIT.md`.

**8. Which 2026 umpire-visible statistics can we actually reproduce historically?**
13 of the 17 directly, with well-established multi-year/decade coverage. The remaining 4 (kick-ins,
intercept marks, intercept possessions, spoils) — which Phase 1 wrongly called fully unreproducible —
are exposed by the AFL's own public API, confirmed live for 2025-2026, but that API's *historical* depth
has not yet been established, so they should be treated as an enrichment layer pending that check, not
assumed available for backtesting yet. Full table: `docs/2026_STATS_MIRROR.md`.

**9. Remaining material limitations**
- ADVANCED-table join has a ~4% unresolved gap (root cause only partly diagnosed).
- The AFL public API's historical depth for the 4 previously-"unavailable" 2026 stats is unknown.
- The experimental event dataset only reaches back to 2021 and has two documented, specific defects.
- Player-quarter-level statistics remain entirely unavailable (unchanged from Phase 1) — still an open
  decision for you (drop the brief's quarter-by-quarter section, restrict to whatever the event-level
  pilot can approximate, or pause for a paid source).
- Umpire-appointment historical data availability (for hierarchical umpire-effect modelling) has not yet
  been investigated at all.
- No 2026 Brownlow votes exist yet — the regime-change question remains untestable until the season ends.

---

## Decisions needed from you before Phase 3 (modelling) begins

1. Accept the CORE (2003-2025) / ADVANCED (2015-2025) starting windows as the candidate set for
   rolling-origin backtesting, rather than a single fixed window?
2. Proceed with the `torp`/`torpdata` experimental event-level track as scoped in
   `docs/EVENT_DATA_2021_AUDIT.md` §6 (build Tier-2 leverage/WPA features on 2021-2026 only, compare
   against the CORE/ADVANCED-only model, report the answer either way)?
3. How to handle the still-unresolved player-quarter-level gap?
4. Any objection to closing out the ~4% ADVANCED-join gap and the AFL-API historical-depth question as
   background Phase 3 investigation rather than blocking modelling on them now?

## Next steps (historical note: this section predates Phase 3/4/5 and is retained for the record only —
see the Phase 5 summary at the top of this file for current status)

Model comparison per `docs/MODELLING_PLAN.md` §2 (baseline → boosting → ranking model → game-state
variant → hierarchical if needed), rolling-origin validation per §4, on the CORE/ADVANCED windows
established above. No feature weighting, model fitting, or predictions until you confirm the above.

## Actual next steps as of Phase 5 completion

Phase 5 (2026 production forecast) is complete and awaiting your review; nothing has been committed to
git yet. The natural first task for a future phase is to check the FINAL_ENSEMBLE forecast against the
real 2026 Brownlow count once it is revealed (see `docs/2026_STRUCTURAL_BREAK.md` §5) — this is the
first point at which the structural-break assumption becomes genuinely testable rather than a
sensitivity band.

## Post-Phase-5 audit and review dashboard (current status)

A targeted production audit (not a new modelling phase) found and fixed a 6-match coverage gap —
`docs/2026_FINAL_AUDIT.md` — bringing coverage to 207/207 matches and revising Nick Daicos's headline
projection from 44.9 to **47.2** EV (mechanical, from adding his own previously-missing Round 1 match;
survived a dedicated over-concentration audit). All figures elsewhere in this file that still say 44.9
or "201/207" predate that fix; `docs/2026_FINAL_AUDIT.md` and `reports/2026_leaderboard.csv` are the
current source of truth.

A read-only Streamlit review dashboard (`app.py`, `dashboard/`, `pages/`) was then built on top of the
audited outputs for use before/during Brownlow count night — see the "2026 Brownlow Review Dashboard"
section of `README.md` and `docs/2026_BROWNLOW_REVIEW_REPORT.md`. It changes no prediction and adds no
new modelling; 8 new integrity tests (`tests/test_dashboard_data.py`) confirm this.

A user-reported round-label bug (2026 matches shown one round late) was then traced, fixed, and
generalized — see `docs/2026_ROUND_INTEGRITY_AUDIT.md` (2026) and `docs/ROUND_NORMALIZATION.md`
(2023-2025 follow-up). Root cause: afltables labels the AFL's unnumbered "Opening Round" as
"Round 1" and numbers everything after sequentially, one higher than the AFL's own official
round number, for every season from **2024** onward (2023 was independently confirmed
unaffected — its Opening Round was itself labelled "Round 1" by both afltables and footywire).
Classified as display/label-only (Class A): `round` is never a join key, sort key, or model
feature anywhere in the codebase. Fixed at source in `src/data/build_core_dataset.py` via the
new `src/data/round_normalization.py`, with `player_match_core_1984_2025.parquet`,
`player_match_advanced_2010_2025.parquet`, `model_core.parquet`, and `model_advanced.parquet`
rebuilt (9,522 rows changed per affected season, 2024 and 2025 only; zero rows changed for 2023
or earlier). Three further match_id-keyed files were found stale mid-fix
(`player_match_role_lagged.parquet`, `oos_predictions_core.parquet`,
`experimental_gamestate_features.parquet`) and fixed with a targeted `match_id`-string patch
rather than a re-run of their source pipelines — no model was retrained. Verified via full
column-level diffs (every non-round/match_id column byte-identical), a checksum comparison of
the one column transiently affected mid-fix (`role`/`role_source` — a join-key staleness
artifact, fully restored, confirmed via SHA-256 hash match against the pre-fix baseline), an
empirical metric recomputation (2025 Model1_PlackettLuce match-level metrics reproduced to full
float precision from the corrected files), and the full test suite (37/37 passing, including
5 new/updated season-aware assertions). No Phase 4 report or model output required
regeneration. Nothing committed to git.
