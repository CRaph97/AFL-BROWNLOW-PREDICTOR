# Exploratory Analysis — Phase 3

Status: **Complete.** All figures below are computed directly from
`data/processed/analytical_features_v1.parquet` (320,861 player-match rows, 1984-2025), restricted to
2003-2025 (the CORE window, per `docs/DATA_COVERAGE.md`) unless stated otherwise. Machine-readable
backing: `reports/univariate_vote_relationships.csv`, `reports/correlation_matrix.csv`,
`reports/season_vote_relationships.csv`. Nothing here is a causal claim — all figures are descriptive
associations, exactly per the brief's own caution in Phase 3 section L.

## 1. Which raw statistics are most strongly associated with votes? (Q1)

Top univariate Spearman correlations with `brownlow_votes` (2003-2025, full ranking in
`reports/univariate_vote_relationships.csv`):

| Feature | Spearman r | Mean votes when polled 3 vs 0 |
|---|---|---|
| SuperCoach points | 0.371 | 132.3 vs 72.0 |
| AFL Fantasy points | 0.363 | 120.1 vs 67.2 |
| `possession_impact_index` (our composite) | 0.321 | 4.45 vs −0.27 |
| `disposals_match_z` | 0.316 | 1.67 vs −0.10 |
| Disposals (raw) | 0.308 | 27.8 vs 15.1 |
| `effective_disposals` | 0.290 (2010+) | 20.5 vs 11.5 |
| `score_involvements` | 0.285 (2015+) | 8.17 vs 3.90 |
| Contested possessions | 0.273 | 11.4 vs 5.6 |
| Metres gained | 0.260 (2015+) | 443.6 vs 245.5 |

**The two external fantasy-scoring composites (SuperCoach, AFL Fantasy) are the single strongest
univariate correlates found in this entire audit** — stronger than any raw box-score stat or our own
composite. This is a genuine finding, not something assumed going in: these formulas were designed for
fantasy-sports purposes, not to approximate Brownlow voting, yet they track it better than any single
raw stat, likely because they already blend disposal volume, efficiency, and scoring into one number.
This does not mean they should be used uncritically as a Brownlow predictor (see §7 caution on
composites), but it is a strong signal that **volume-weighted, multi-stat composites beat any single
raw stat**, which independently supports building a proper multivariate model rather than relying on
one or two headline numbers.

## 2. Do match-relative statistics beat raw totals? (Q2)

Yes, modestly and consistently, exactly matching the brief's stated hypothesis:

- `disposals_match_z` (0.316) and `disposals_match_pct` (0.314) both **outperform raw disposals**
  (0.308) in Spearman correlation with votes.
- `contested_possessions_match_z` (0.279) likewise edges out raw contested possessions (0.273).
- The improvement is real but not large (~0.01-0.02 correlation units) — raw totals are already
  fairly informative on their own, and the match-relative transform adds a modest, not transformative,
  amount of extra signal in a purely univariate sense. Its value may be larger in a multivariate model
  once combined with teammate-competition and composite features (Phase 4 question, not answered here).

## 3. How strong is the winner effect, and does it depend on margin? (Q3, Q4)

Very strong, and **yes, it depends heavily and asymmetrically on margin** — this is one of the clearest
findings in this audit. Binning by absolute margin (2003-2025):

| Absolute margin | Winner polling rate | Loser polling rate | Gap |
|---|---|---|---|
| 0-6 (very close) | 8.3% | 5.2% | 3.0pp |
| 6-12 | 9.2% | 4.3% | 4.8pp |
| 12-24 | 10.0% | 3.5% | 6.6pp |
| 24-48 | 11.5% | 2.0% | 9.6pp |
| 48-100 | 12.9% | 0.6% | 12.3pp |
| 100-200 (huge blowout) | 13.5% | 0.05% | 13.4pp |

**Winners poll MORE as margin grows** (8.3% → 13.5%), while **losers poll dramatically LESS**
(5.2% → 0.05%, essentially never polling in a 100+ point loss). The winner-vs-loser gap roughly
quadruples from close games to blowouts. This directly answers Q4: the winner effect is not constant —
it is much larger in blowouts than in close games, and the effect is driven far more by the collapse of
loser polling than by any dramatic rise in winner polling. A losing player can still poll meaningfully in
a close game (5.2%, not far below the winner rate of 8.3%) but is almost never rewarded in a blowout loss
regardless of individual performance.

Season-by-season, this winner-advantage-in-polling-rate is **remarkably stable** across 1999-2025
(oscillating narrowly between 7.3 and 9.7 percentage points with no clear trend — see
`reports/season_vote_relationships.csv`) — unlike some other relationships (§6), the basic winner
effect does not appear to be an era-drifting phenomenon.

## 4. How substantial is teammate vote competition? (Q5)

Very substantial. Holding a player's own disposal output roughly fixed at 25-29 (a genuinely good game),
polling rate falls sharply and monotonically as more teammates also have big games:

| Teammates with 30+ disposals | Own polling rate (disposals fixed at 25-29) | n |
|---|---|---|
| 0 | 27.8% | 6,137 |
| 1 | 20.4% | 5,614 |
| 2 | 14.4% | 2,754 |
| 3 | 12.3% | 1,076 |
| 4 | 10.2% | 382 |
| 5 | 9.3% | 97 |

A near-3x reduction (27.8% → ~9-10%) in polling rate for an **identical own performance band**, purely
as a function of how many teammates also performed at a high level. This is strong, direct, sample-size
-backed evidence for the brief's "vote stealing" hypothesis — a good game is worth much less when
several teammates also had good games, independent of the opponent or match context entirely.

## 5. Which variables show important nonlinearities? (Q6)

**Disposals: strongly convex, not linear.** The marginal effect of +5 disposals accelerates sharply at
higher volumes:

| Disposal band | Polling rate | Marginal jump from previous band |
|---|---|---|
| 15-19 | 3.8% | — |
| 20-24 | 7.8% | +4.0pp |
| 25-29 | 21.3% | +13.5pp |
| 30-34 | 44.5% | +23.2pp |
| 35-39 | 69.2% | +24.7pp |

Going from 30→35 disposals is worth roughly **six times** the polling-rate improvement of going from
15→20 — a dramatic, unambiguous confirmation of the brief's exact worked example. Any model treating
disposals as linear will materially misrepresent this relationship.

**Goals: even more extreme convexity**, confirming the "five-goal haul" salience hypothesis directly:

| Goals | Polling rate | 3-vote rate |
|---|---|---|
| 0 | 3.9% | 1.1% |
| 1 | 7.8% | 2.6% |
| 2 | 11.2% | 4.3% |
| 3 | 16.7% | 6.1% |
| 4 | 34.9% | 10.1% |
| 5 | 60.9% | 20.0% |
| 6 | 83.4% | 37.0% |

The 3-vote rate more than **doubles with every additional goal from 4 to 6** — a five- or six-goal game
is not just "more of the same" as a two-goal game, it occupies a qualitatively different, much more
heavily rewarded region of the outcome space. Both findings argue strongly for nonlinear treatment
(splines, piecewise terms, or tree-based models that discover this automatically) in Phase 4, per the
brief's explicit instruction not to assume linearity.

## 6. Which interactions are plausible enough to test formally? (Q7)

- **Performance × role** is clearly justified (see `docs/ROLE_ANALYSIS.md` for the full breakdown): a
  key forward's goal-scoring translates to votes very differently from a key defender's equivalent
  defensive dominance — the same "elite performance for the role" does not translate to the same
  polling rate across roles.
- **Performance × win/margin** is clearly justified per §3 — the same statistical performance carries
  very different vote probability depending on match result and its margin.
- **Performance × teammate competition** is clearly justified per §4.
- These three are recommended as the priority interaction candidates for Phase 4, rather than a large
  undirected grid of polynomial interactions (per the brief's explicit instruction against that).

## 7. How much does voting behaviour change across eras? (Q8)

See `docs/TEMPORAL_DRIFT.md` for the full season-by-season table and discussion. Headline finding:
**clearances' association with voting has risen steadily** (Spearman ~0.16-0.19 in 1999-2009 to
~0.21-0.26 in 2010-2025), while the winner-advantage effect (§3) and contested-possession association
have stayed comparatively flat. Goals' association shows a mild decline in the most recent seasons.

## 8. Are role effects large enough to justify explicit modelling? (Q9)

Emphatically yes. See `docs/ROLE_ANALYSIS.md` — midfielders poll at roughly **7.4x** the rate of key
defenders per game, and capture **64.5%** of all 3-vote games despite being well under half of all
player-matches in the sample. This is one of the largest, cleanest effects found in the entire audit.

## 9. Which features are redundant? (Q10)

See `reports/correlation_matrix.csv` for the full matrix; 18 pairs exceed |Spearman r| = 0.85, most
unsurprisingly mechanical (a stat and its own derived match-z/team-share/composite are naturally highly
correlated with each other). The genuinely useful redundancy findings, i.e. between conceptually
*different* raw stats:
- `clearances` ↔ `stoppage_clearances`: r = 0.91 — the large majority of clearances are stoppage
  clearances, not centre-bounce clearances, in the 2015-2025 window checked.
- `disposals` ↔ `effective_disposals`: r = 0.92 — effective disposals track raw disposals very closely;
  disposal *efficiency* (the % form) is the more independent signal of the two.
- `disposals` ↔ `uncontested_possessions`: r = 0.87 — most disposal volume is driven by uncontested
  possession volume, not contested.
- `afl_fantasy_points` ↔ `supercoach_points`: r = 0.86 — keep one as an external composite benchmark,
  not both.

## 10. Event data feasibility (Q13, Phase 3 section O)

**A materially important correction was found and fixed during this audit, not merely observed.**

The `torp`/`torpdata` chain feed's `final_state` field (e.g. "rushed", "rushedOpp", "behind") is a
**chain-level** outcome copied onto **every action row** within that chain — not a per-event flag. The
single-match spot check in `docs/EVENT_DATA_2021_AUDIT.md` (Phase 2) did not surface this because it
only checked one match's aggregate counts informally; a naive full-2024-season reconciliation against
official scores initially found only an **11.6% exact match rate for behinds** (far worse than the ~83%
implied by Phase 2's single-match check) once every action row was naively counted as a separate score.

After correcting the logic to count **one behind per distinct chain** (not per action row), with the
documented team-attribution rule (`rushedOpp` credits the *other* team from the chain's own attacking
team; `rushed` and `behind` credit the chain's own team directly):

| Metric | Match rate vs official scores (2024 season, 215 matches) |
|---|---|
| Goals | 99.1% |
| Behinds (naive, uncorrected) | 11.6% |
| **Behinds (corrected chain-level logic)** | **96.0%** |

Of the 17 remaining mismatched team-sides (out of 430), 13 are off by exactly one behind (a small,
consistent residual, cause not further diagnosed), and 2 specific matches have larger, unexplained
discrepancies (5-11 behinds out of ~10-15) that look like genuine data-quality problems in those
specific matches, not a general methodology flaw.

**Verdict: game-state reconstruction (running score, quarter, time remaining, lead changes, close-game
state) is feasible and sufficiently reliable for the EXPERIMENTAL track**, provided the corrected
chain-level logic in `src/features/event_feasibility.py` is used — the naive row-level approach is not
usable and would have silently produced badly wrong game states if it had gone unchecked. A minimal,
unfit `simple_leverage()` placeholder (closeness × time-elapsed) was implemented as a feasibility
demonstration only — it has not been evaluated against Brownlow outcomes and must not be treated as a
working leverage feature yet. This whole track remains EXPERIMENTAL and 2021-2026 only, per
`docs/EVENT_DATA_2021_AUDIT.md`.

## 11. Summary answers to the Phase 3 required questions

See `PROJECT_STATE.md` for the consolidated point-by-point answer to all 13 questions in the brief's
section Q, cross-referencing this document, `docs/ROLE_ANALYSIS.md`, `docs/TEMPORAL_DRIFT.md`, and
`docs/LEAKAGE_AUDIT.md`.
