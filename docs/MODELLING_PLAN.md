# Modelling Plan — AFL Brownlow Predictor

Status: **Phase 1 proposal — no model has been trained. This document is a plan to be reviewed and agreed,
not a report of results.**
Last updated: 2026-09-16

This plan should be read together with `docs/DATA_SOURCE_AUDIT.md` (what data we can actually get) and
`docs/FEATURE_CANDIDATES.md` (the feature registry). It deliberately does not specify final hyperparameters,
weights, or feature coefficients — those must come from fitting models to data, not from this document.

---

## 1. Framing the prediction problem

The fundamental unit of analysis is **player × match**. But the target is not independent per player: within
every applicable match, votes are allocated as exactly one 3, one 2, one 1, and zero to everyone else. This is
a **within-match ranking/choice problem**, not a set of independent 4-class classifications, and the modelling
approach must respect that constraint structurally, not just approximately.

We will treat this as nested problems at two levels:

1. **Match-level allocation model:** given all players who took part in a match and their match-level feature
   vectors, produce a probability distribution over "who gets the 3, who gets the 2, who gets the 1" that
   sums correctly within the match (a proper ranking/choice model, not independently-normalised per-player
   probabilities).
2. **Season-level aggregation:** simulate many draws from the match-level model across a whole season,
   respecting the one-3/one-2/one-1-per-match constraint in every simulated draw, and aggregate to get
   season-level expected votes, intervals, and leaderboards (Monte Carlo, detailed in §6).

This two-level structure is what allows the project to produce every deliverable in the brief (match
probabilities, season aggregates, intervals, confidence ratings) from one coherent process rather than bolting
independent pieces together.

---

## 2. Candidate modelling approaches (to compare, not to pre-select a winner)

| Approach | What it is | Why it's a candidate | Known risk |
|---|---|---|---|
| **A. Transparent baseline** | Simple, interpretable ordinal/multinomial logistic regression on a small, defensible feature set (disposals, goals, clearances, win/loss, margin) | Establishes the floor every other model must beat; fully interpretable | Ignores within-match ranking constraint; likely underfits |
| **B. Conditional/rank-ordered logit or Plackett-Luce** | Directly models "probability this player is the best-afield, second-best, third-best in this match" as a proper ranking likelihood | Structurally matches how votes are actually awarded (3-2-1 per match); well-established in econometrics/choice modelling literature | More complex to implement and diagnose than a boosting model; fewer off-the-shelf libraries than XGBoost |
| **C. Gradient-boosted trees (XGBoost/LightGBM/CatBoost)** | Multiclass or ordinal classifier over {0,1,2,3} votes, with within-match re-normalisation/re-ranking applied afterward to enforce the constraint | Captures nonlinearities and interactions with far less manual feature engineering; strong track record in similar sports-prediction problems (used by at least one public Brownlow model found in Phase 1 research) | Constraint-enforcement is a post-hoc patch rather than a structural property of the model; needs careful calibration |
| **D. Game-state-enhanced variant** | Any of B/C, extended with the context/leverage/quarterly features from §5, tested specifically for incremental value over the equivalent model without them | Directly tests the brief's central hypothesis that context matters, rather than assuming it | Feature availability is genuinely limited (see Data Audit §6) — the enhancement may be smaller than the brief anticipates, and that finding must be reported honestly if so |
| **E. Bayesian hierarchical model** | Partial pooling across players/teams/eras/(if usable) umpire groups, e.g. via PyMC | Naturally handles small samples (rare events like a 3-vote game), quantifies uncertainty as a first-class output rather than a bolt-on, well suited to hierarchical structure (player-in-team-in-season) | Slower to fit and iterate than tree models; requires more statistical care to specify sensibly |
| **F. Learning-to-rank (LambdaMART-style, or neural)** | Listwise ranking loss directly optimising for correct within-match ordering | Purpose-built for "rank items within a group" problems | Likely needs more data volume/engineering effort than justified until simpler approaches are exhausted; brief explicitly says neural methods only if data volume justifies — we do not currently believe it does, pending Phase 2 data volume confirmation |
| **G. Ensemble** | Combination of B/C/D/E (and F if it earns its place) | Brief explicitly wants this only "if evidence supports it" | Must not be built until component models are validated individually — building an ensemble first would hide which components are actually adding value |

**Working recommendation for Phase 3 build order:** A (baseline) → C (boosting, well-understood and fast to
iterate) → B (proper ranking model, to test whether structural correctness beats flexible-but-unconstrained
trees) → D (add game-state features to whichever of B/C wins) → E (only if uncertainty quantification from
simulation-on-top-of-C/B proves insufficient) → G (only if ≥2 independently-validated models show genuine,
non-redundant skill). This ordering is a proposal for discussion, not a locked-in sequence.

---

## 3. The 2026 regime change — treatment plan

This is a structural break in the data-generating process, not a cosmetic rule tweak: from 2026, umpires are
handed 17 numeric fields immediately before casting votes (see Data Audit §7). Historical (pre-2026) voting
reflects umpires' own in-game observation and memory; 2026+ voting reflects that observation filtered/anchored
through an explicit numeric scoreboard the umpire did not previously have. We will not assume these processes
are identical, and we will not assume they are completely different — both are testable claims, addressed as
follows:

1. **Baseline: historical (pre-2026) model.** Trained and validated entirely on 1978–2025 data (or whatever
   window Phase 2 verification confirms is usable), used as the reference point for everything else.
2. **Structural-break indicator.** Add an explicit `post_2026` indicator/interaction in models that support it,
   so that once 2026+ vote data exists, the model can estimate whether and how coefficients shifted, rather
   than silently pooling eras that may behave differently.
3. **Recency weighting.** As an alternative/complement to a hard indicator, down-weight older seasons in
   training so the model is more responsive to the most recent behaviour once some 2026 data exists, without
   discarding pre-2026 history entirely.
4. **Sensitivity analysis.** Before any 2026 votes exist, explicitly test how much the model's 2026 predictions
   would change under a range of assumed shifts (e.g. "what if the 4 stats umpires now see systematically get
   more weight than before") — reported as a range, not a point estimate, precisely because we have no data yet
   to pin this down.
5. **Bayesian priors informed by history.** For the hierarchical model variant (E), pre-2026 fitted
   distributions become informative priors for 2026+ parameters, which are then updated as real 2026 votes
   accumulate — a principled way to say "start from history, but let evidence move us."
6. **Post-2026 recalibration.** As soon as actual 2026 Brownlow votes are known (end of the 2026 season, or
   progressively if any in-season signal exists, e.g. Coaches Association votes as a leading indicator), refit
   and compare against the pre-2026 baseline explicitly, and report the delta.
7. **What we will NOT do:** invent a hypothesis about which of the 17 stats umpires weight most, or assume the
   4 stats we cannot obtain historically (kick-ins, intercept marks, intercept possessions, spoils — Data
   Audit §7) matter more or less than the 13 we can. Any claim about differential umpire weighting must come
   from actual 2026 voting data once available, or be clearly labelled as an untested hypothesis under
   sensitivity analysis, never presented as a finding.

**Until real 2026 votes exist, all 2026-season predictions from this project must be presented as
extrapolations from a pre-2026-fitted model under an explicitly stated assumption of continuity, with the
sensitivity-analysis range shown alongside — not as validated 2026 predictions.** This should be a standing
disclaimer on any 2026 output, not a one-time footnote.

---

## 4. Validation framework

### 4.1 Time-aware splitting (mandatory)

No random cross-validation across seasons. Rolling-origin ("expanding window") validation only:
train on seasons up to year *N*, validate on year *N+1*, then roll forward. This mirrors the brief's explicit
requirement and avoids leaking future voting-behaviour shifts backward into training.

### 4.2 Match-level metrics

- Exact 3-2-1 order accuracy
- Correct-3, correct-2, correct-1 identification rates (independently)
- "All three vote recipients identified regardless of order" rate
- Top-3 precision/recall
- Ranking correlation (e.g. Spearman) between predicted and actual within-match vote order
- Log loss and Brier score on the P(3)/P(2)/P(1)/P(0) distributions
- Calibration curves (predicted probability vs. observed frequency), not just point accuracy

### 4.3 Player-season-level metrics

- MAE / RMSE of total predicted vs. actual votes
- Bias (systematic over/under-prediction, checked by segment — see §4.5)
- Rank correlation between predicted and actual season order
- Prediction-interval coverage (does the stated 80%/95% interval actually contain the true outcome ~80%/95% of
  the time, historically? If not, intervals are miscalibrated and must be widened/narrowed accordingly, not
  left as-is)
- Error specifically among top-20 Brownlow contenders (the segment that matters most in practice)

### 4.4 Leaderboard-level metrics

- Top-5 and top-10 inclusion rate (not just exact winner accuracy)
- Full-season rank correlation
- Season-level probability calibration (e.g., among all players given a 30% chance of finishing top-10, do
  roughly 30% actually finish top-10 across seasons?)

### 4.5 Mandatory error segmentation

All of the above must be reported broken down by, at minimum: player position/role, winning vs. losing
players, close games vs. blowouts, high-disposal games, high-goal games, season, and team. A model with good
average performance but a large, consistent blind spot (e.g. systematically under-rating key forwards) is not
acceptable to ship silently — it must be surfaced.

### 4.6 Leakage controls (mandatory, checked explicitly, not just by convention)

- No feature computed using information not available before that match's votes were cast (e.g. season-end
  awards, later-season form, final ladder position at a date after the match).
- Historical-reputation features (prior votes, All-Australian history) are explicitly run as an ablation:
  model-with vs. model-without, both reported, so reputation is never silently substituting for current-game
  signal.
- A leakage checklist is applied to every new feature before it enters any model (see Feature Candidates doc —
  every feature has a "risk of leakage" field, not optional).

---

## 5. Game-state / clutch value framework — scoped to what the data actually supports

Given the Data Audit's finding that true play-by-play data is not available at scale (§6 of that document),
this framework is deliberately scoped down from the brief's ideal to what is honestly buildable, in three
tiers, tested independently for incremental value rather than assumed to help:

- **Tier 1 (buildable now, all usable seasons):** quarter-level *team* score margin trajectory (public back
  decades) combined with match-total player stats — e.g. "share of match played with the scoreline within one
  score", "was the player's team leading or trailing at three-quarter time", "final-quarter margin vs.
  half-time margin (comeback/fade indicator)". These are match-context features, not true event-level leverage,
  and will be labelled as such everywhere they appear.
- **Tier 2 (pilot only, 2021 season, contingent on verifying the play-by-play dataset's player IDs/timestamps):**
  a genuine event-level Leverage Index / WPA-style construction, built and validated on one season as a proof
  of concept, explicitly not used for historical training or backtesting beyond that season given the coverage
  gap.
- **Tier 3 (aspirational, contingent on future access):** if a Champion Data licensing relationship or a
  broader public event dataset ever becomes available, extend Tier 2's methodology across history.

Every derived "clutch" feature (Clutch Clearances, High-Leverage Disposals, etc.) will be built as a
**continuous weighting**, not an arbitrary binary threshold, per the brief's explicit instruction — but will
only ever be built at the tier the underlying data actually supports, and its incremental predictive value
will be tested (not assumed) before inclusion in any non-experimental model, exactly as required in §4.6.

---

## 6. Uncertainty and simulation

### 6.1 Match-level output

For each player in each match: `P(3), P(2), P(1), P(0)` and `Expected Votes = 3·P(3) + 2·P(2) + 1·P(1)`,
derived directly from whichever match-level model (B/C/D/E) is in use — never hand-set.

### 6.2 Season-level Monte Carlo

- For each simulated season, draw one 3-2-1 outcome **per match**, sampled jointly from the match-level model's
  output for that match (e.g. via a Plackett-Luce draw, or by sampling from the fitted rank-ordered
  probabilities) so that every simulated match obeys the one-3/one-2/one-1 constraint by construction — never
  by sampling each player's vote count independently and hoping totals happen to work out.
- Repeat for a number of simulations large enough for stable tail estimates (exact number to be determined by a
  convergence check in Phase 3 — e.g. running until the 95th-percentile season total estimate stabilises to
  within a small tolerance across additional simulation batches, not an arbitrarily chosen round number).
- Aggregate simulated season totals per player into: expected votes, median, 80% interval, 95% interval
  (reported only where statistically meaningful, i.e. not for players whose realistic range is trivially 0),
  and probability of exceeding relevant thresholds (e.g. medal contention thresholds).

### 6.3 Confidence ratings

Derived objectively from the shape of the match-level probability distribution — e.g. entropy of `(P(3), P(2),
P(1), P(0))` across the plausible candidates in a match, and the probability gap between the top two candidates
— mapped to VERY HIGH / HIGH / MEDIUM / LOW / VERY LOW bands via thresholds calibrated against how often each
band's "top pick" is actually correct historically (a calibration exercise, not an arbitrary cutoff choice).

### 6.4 Model disagreement

Once ≥2 independently-trained models exist (e.g. C and B, or C and D), compute a disagreement measure per
match/player (e.g. variance or range of each model's predicted vote/expected-votes for that player) and surface
it as an explicit uncertainty signal, per the brief — high agreement reinforces confidence, high disagreement
flags a case for manual review rather than blind trust in either model.

---

## 7. Explanation generation

Every "why is this player predicted to poll" explanation must be generated **programmatically from the actual
features and their fitted contributions** for that specific player-match (e.g. SHAP values for tree models, or
directly interpretable coefficients/terms for the ranking and hierarchical models) — never hand-written or
templated with generic boilerplate. This is an engineering requirement for Phase 3/4, not a modelling-method
decision, but is recorded here because it constrains which models are viable: any candidate model must support
some form of per-prediction feature attribution, which is one more point in favour of tree-based and linear/
hierarchical models over harder-to-interpret alternatives, all else equal.

---

## 8. External benchmarking

Once our models exist, compare (not tune to match) against publicly available Brownlow predictors identified in
the Data Audit: AFL.com.au's own Brownlow Predictor, WheeloRatings (Monte Carlo simulation approach), Stats
Insider (ordinal logistic regression using media/coaches votes), and others surfaced via the Squiggle API's
model-comparison endpoint where applicable. Differences will be investigated and documented, not eliminated by
adjusting our model to agree.

---

## 9. Technology stack proposal

Python-first, matching the brief's own default and the ecosystem's dominant tooling:

- **Data/ETL:** pandas or polars; requests/BeautifulSoup only if we must scrape directly (see Data Audit §5 —
  prefer reusing fitzRoy/its data repo over building our own scraper).
- **Feature engineering:** pandas/numpy, with a structured feature registry (see Feature Candidates doc) kept
  in sync with code, not just documentation.
- **Modelling:** scikit-learn (baseline + ordinal models), statsmodels (conditional logit, GLMs), xgboost/
  lightgbm/catboost (boosted trees), PyMC (hierarchical Bayesian, only if E is pursued).
- **Explanation:** SHAP for tree models.
- **Simulation:** numpy-based Monte Carlo; a `src/simulation` module enforcing the per-match constraint at
  the code level (not just conceptually).
- **Visualisation/reporting:** matplotlib/plotly.
- **Environment:** a pinned, reproducible environment (e.g. `pyproject.toml`/`requirements.txt` with pinned
  versions, or a conda/poetry environment) — to be created when Phase 2 begins, not before, so it reflects
  actual chosen libraries rather than a speculative list.

No installation happens in Phase 1. This section is a proposal for Phase 2 sign-off.

---

## 10. Explicit non-goals for now

- No hyperparameter values, feature weights, or model coefficients are specified anywhere in this document —
  all must come from fitting to real data.
- No 2026 Brownlow prediction will be produced until historical model validation (§4) is complete on pre-2026
  data.
- No ensemble (G) will be built before its components are independently validated.
- No claim of "true" leverage/WPA will be made outside the Tier-2 pilot scope in §5.
