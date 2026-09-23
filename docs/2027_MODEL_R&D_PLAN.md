# 2027 Brownlow model — R&D plan

Branch `2027-model-rd`. Goal: a reproducible, point-in-time-correct, Brownlow-specific modelling
system that can be frozen before 2027 Round 1 and evaluated honestly afterwards. Not "fit 2026
better": 2026 is now known and is treated as the latest out-of-sample fold that has *also*
informed this research.

## 0. Integrity rules (enforced in code and tests)

- Frozen 2026 Production / Objective predictions, Wheelo snapshot, betting snapshots, simulation
  outputs, the actual-vote dataset and the 2026 evaluation are immutable evidence.
  `tests/test_2027_rd.py::test_frozen_outputs_unchanged_by_rd` re-hashes them.
- 2026 actual votes are used only as **labels** (feature store `label_source = afl_actual_2026`).
- No random splits: `src/validation/walk_forward.py` builds train < test season folds only; the
  inner holdout for hyperparameters / temperature / calibration is the last *training* season.
- Every feature has a declared timing class in `data/features/feature_registry.json`
  (`same_match`, `prior_matches`, `prior_seasons`, `same_season_unrevealed`). The last class is the
  legacy reputation family (`brownlow_votes_prev*_mean`, `brownlow_votes_season_to_date_mean`),
  which averages votes from earlier rounds of the same season — votes that do not exist until
  count night. It is excluded from every 2027 candidate and evaluated only as a non-deployable
  upper bound in the ablation.

## 1. Architecture

```
data/features/      point-in-time feature store (src/features/point_in_time.py)
data/experiments/   registry.jsonl, metrics/, oof/, analysis/, logs/
src/validation/     metrics.py, walk_forward.py, registry.py, run_experiments.py, analyze.py
src/models/structural/     A: Plackett-Luce + analytic gradient (fast_pl.py)
src/models/performance_ml/ B: XGBoost LambdaMART ranker + inner-holdout temperature -> PL marginals
src/models/stats_only/     C: learned linear PL on strictly same-match features (guarded)
src/models/ensemble/       walk-forward log-linear stacking of OOF P3
src/simulation/season_sim.py  model-agnostic 3-2-1 season simulator (Gumbel-max PL draws)
pages/40_2027_Model_Lab.py    R&D dashboard (R&D nav section)
```

Legacy code is untouched; new modules import the validated Plackett-Luce marginalisation and
scaler from `src/models/plackett_luce.py`.

## 2. Candidates

| | Model | Features | Purpose |
|---|---|---|---|
| A | Structural PL | same-match + prior-match (role, lagged form, baseline-relative, team strength) + prior-season reputation | umpire behaviour + context, ordered 3-2-1 structure |
| B | Performance ML | same families, XGBoost rank:ndcg with match as query group, temperature fitted on inner holdout | nonlinear performance -> vote utility |
| C | Stats-only PL | same-match families only (raw, relative, team-relative, context, teammate, dominance, hinges) | independent learned benchmark, Objective successor |
| C-ML | Stats-only ranker | as C, nonlinear | does nonlinearity help without context/history? |
| Baseline | Phase 4 / Production Scenario A PL | legacy 68 CORE features incl. dropna behaviour | reproduce known numbers |
| Wheelo | external | — | evaluation only, never a training target |

Ensemble: weights learned by minimising OOF P3 log loss on strictly earlier seasons; compared to
the best single model and to equal weights; promoted only if it wins across seasons.

## 3. Feature families (data/features/feature_registry.json)

same_match: raw (22), match_relative z (8), match_relative_ext z/pct/share (66 incl. 2015+
advanced), team_relative (32), context (6), teammate (5), dominance / vote competition (11:
fixed composite `impact_z` rank, gap to match best, gap to best / 2nd teammate, strong-teammate
count, teammate concentration, match standout margin), nonlinear hinges (4).
prior_matches: role dummies + interactions (18), lagged form (16), baseline-relative (10), team /
opponent season-to-date and prior-season strength (7).
prior_seasons: reputation_pit (prior-season votes per game, last-season votes, total).
Umpire data: not present in any raw source held (fitzRoy player tables, AFL API match payloads
have no umpire field); sourcing would mean scraping ~4,500 afltables match pages — deferred and
documented rather than done in this run.

## 4. Validation

Test seasons 2012–2026, expanding and recent-8 windows (CORE window starts 2003). Match metrics:
3-vote accuracy, 3-voter in top 2 / top 3, exact 3-2-1, unordered top 3, P3 log loss (per match),
P3 Brier, ECE. Season metrics: MAE, RMSE, Spearman, rank MAE (actual top 30), winner rank, top
3/5/10 hit rates. Every metric is reported by season, pooled (2012–2026), recent (2022–2026) and
pre-2026, with bootstrap CIs over matches and paired bootstraps between models.

## 5. Selection and promotion

Hyperparameters: small fixed spaces, chosen on inner holdouts, never on the test season.
Promotion rules (`docs/2027_MODEL_R&D_RESULTS.md` §Champion): a challenger replaces the champion
only if it is better on P3 log loss AND 3-vote accuracy pooled, not worse on season MAE /
Spearman, better in a majority of seasons including the recent window, with paired-bootstrap
CIs excluding zero on at least one headline metric, and not more complex than justified.

## 6. Priorities executed

P0 audit, feature store, walk-forward + metrics, baseline reproduction; P1 candidates A/B/C with
OOF predictions; P2 ablation, disagreement, calibration, Error Lab, ensemble; P3 Model Lab page,
simulation framework, umpire/advanced-data audit. Betting research reuses the frozen 2026
settled-market evaluation as evidence only (no staking optimisation, no historical odds exist).
