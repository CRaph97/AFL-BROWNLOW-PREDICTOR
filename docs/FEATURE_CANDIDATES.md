# Feature Candidate Registry — AFL Brownlow Predictor

Status: **Phase 1 — candidate list only. No feature has been computed, tested, or selected.**
Last updated: 2026-09-16

This registry lists candidate features by category, per the brief's structure. Each entry records what it
would take to build the feature, not a pre-judgement of its usefulness — "expected relationship" is an
intuitive prior to be tested, not an assumed result, and "actual observed relationship" / "importance in final
model" are left blank everywhere because no model has been fit yet. Filling those two columns in prematurely
would be exactly the kind of fabrication the project brief prohibits.

**Data availability codes** (from `DATA_SOURCE_AUDIT.md`): **A** = available now, **D** = derivable, **P** =
partially available/unverified, **U** = unavailable. A feature marked U or P must not be built as if it were A.

Template per feature:

```
### <name>
- Definition:
- Category:
- Source / calculation:
- Data availability:
- Reason for inclusion (intuitive prior — A/B/C framing per brief):
- Missingness:
- Leakage risk:
- Expected relationship (hypothesis, not a finding):
- Actual observed relationship: [Phase 3 — not yet tested]
- Importance in final model: [Phase 3 — not yet tested]
```

---

## 1. Volume

### Disposals, kicks, handballs (raw match totals)
- Definition: Count of each action in a match.
- Category: Volume
- Source/calculation: Direct field, AFL Tables/Footywire.
- Data availability: A
- Reason for inclusion: (A) intuitively the most obvious Brownlow correlate; (B) historically the strongest
  univariate predictor in essentially every public model reviewed in Phase 1; (C) to be tested for genuine
  out-of-sample contribution once combined with quality/context features, since the brief explicitly warns
  raw volume may be over-weighted relative to quality.
- Missingness: None expected in-window.
- Leakage risk: None — fully determined during the match.
- Expected relationship: Positive, likely nonlinear/plateauing at high volumes (per brief's explicit
  hypothesis: marginal value of the 30th disposal ≠ marginal value of the 15th).

### Goals, behinds
- Category: Volume / Scoring
- Data availability: A
- Reason for inclusion: Scoring is highly salient to human voters (brief's "salience" hypothesis) and
  independently a core performance measure.
- Expected relationship: Strongly positive for goals, especially nonlinearly for hauls (e.g. 4+ goals), weaker
  or even slightly negative in isolation for behinds (inefficiency signal) — both are hypotheses to test.

### Hitouts
- Category: Volume / Ruck
- Data availability: A
- Reason for inclusion: Primary raw ruck-role stat; needed as a baseline before testing role-interaction terms.
- Expected relationship: Weak/inconsistent in isolation historically (commonly cited finding in public AFL
  analytics writing) — to be tested, not assumed, on our own data.

---

## 2. Efficiency

### Disposal efficiency (effective disposals / total disposals)
- Data availability: P — effective-disposal field not confirmed on standard public views (Data Audit §3);
  needs Phase 2 confirmation before this can move from candidate to buildable.
- Reason for inclusion: Brief explicitly wants quality over volume distinguished; this is the most direct such
  metric.
- Expected relationship: Positive, independent of raw volume.

### Contested vs. uncontested possession split / contested possession rate
- Data availability: A (season totals confirmed on AFL Tables; per-game and full-history start-year P)
- Reason for inclusion: Contested-ball is widely believed (media narrative — category A in the brief's A/B/C
  framing) to carry more voting weight than uncontested; must be tested against C (actual predictive value),
  not assumed from B (media narrative).
- Expected relationship: Contested possessions hypothesised to have higher marginal Brownlow value than
  uncontested, per common commentary — explicitly to be tested.

---

## 3. Contest

### Contested marks
- Data availability: A
- Reason for inclusion: Salient, visible action; also a distinguishing feature between key-position and
  midfield roles, relevant to §8 role interactions.

### One-percenters
- Data availability: A
- Reason for inclusion: Often defensive/team-service actions; hypothesis is this is one of the lowest-salience
  stats for human voters despite defensive value — a good test case for the "intuitive importance vs. actual
  voting predictiveness" distinction the brief asks us to separate explicitly.

### Ground ball gets, tackles inside 50, pressure acts, metres gained
- Data availability: U (Data Audit §3)
- Reason for inclusion: Listed in brief as desired quality/pressure indicators.
- Status: **Cannot be built** until/unless a source is found in Phase 2. Retained here as a documented gap,
  not silently dropped.

---

## 4. Scoring

### Score involvements
- Data availability: D — not confirmed as a direct scraped field; likely to be constructed as a defensible
  composite (e.g. goals + goal assists + other scoring-chain contributions the data supports), with the exact
  definition documented once built, since the brief explicitly warns against arbitrary composite definitions.
- Reason for inclusion: Named directly in both the brief's feature list and the AFL's own 17 umpire-facing
  stats — high priority once a clean definition is confirmed as buildable.

### Goal type distinctions (go-ahead goal, equalising goal, late close-game goal, blowout goal)
- Data availability: U at scale historically (requires event-level/timestamped scoring data — Data Audit §6);
  P for the single pilot season (2021) only.
- Reason for inclusion: Directly requested "salient performance" and "clutch" hypotheses in the brief.
- Status: Buildable only within the Tier-2 pilot scope defined in `MODELLING_PLAN.md` §5 — not for full
  historical training.

---

## 5. Defence

### Intercept possessions, intercept marks, spoils
- Data availability: U historically / P recently (Data Audit §7 — these are 3 of the 4 umpire-visible stats we
  could not confirm on free sources).
- Reason for inclusion: Directly on the AFL's own 2026 approved-stats list; a defensible proxy or alternate
  source should be actively sought in Phase 2 given their importance to the regime-change question.

### Rebound 50s
- Data availability: A
- Reason for inclusion: Best currently-available public proxy for defensive-team-service contribution given
  the unavailability of intercepts/spoils historically; to be tested as an imperfect substitute, clearly
  labelled as such rather than presented as equivalent.

---

## 6. Clearance

### Clearances (total)
- Data availability: A

### Centre clearances / stoppage clearances (split)
- Data availability: P — not confirmed on standard views; check footywire "advanced" tab in Phase 2.
- Reason for inclusion: Brief specifically distinguishes clearance types as differently salient (centre-bounce
  clearances arguably more visible/impactful than boundary-throw-in clearances).

### Clearances resulting in scores / high-leverage clearances
- Data availability: U at scale (requires possession-chain linkage — event-level data, Data Audit §6); P for
  2021 pilot only.

---

## 7. Territory

### Inside 50s
- Data availability: A

### Metres gained
- Data availability: U (Data Audit §3) — no public source found.

### Rebound 50s
- (see §5 — dual-purpose territory/defence feature)

---

## 8. Ruck

### Hitouts, hitouts to advantage
- Data availability: A (hitouts) / U (to advantage)
- Reason for inclusion: Role-specific baseline; ruck role interaction terms (§13) depend on this.

---

## 9. Team

### Team result (win/loss/draw), margin
- Data availability: A
- Reason for inclusion: Brief explicitly requires empirically estimating winner advantage rather than assuming
  a constant bonus, and testing whether it interacts with margin, role, and dominance — this is the base
  variable that analysis depends on.

### Team score involvement share / team clearance share / team contested-possession share
- Data availability: D (derivable from existing match totals across all players in a match)
- Reason for inclusion: Directly supports the brief's "teammate vote competition" and "relative match
  dominance" requirements.

---

## 10. Relative dominance (within-match)

### Player rank within match (per stat), player z-score within match, player value minus match mean, value /
next-best player
- Data availability: D — fully derivable from any match-level box-score dataset once every player in the match
  is included, no extra source needed.
- Reason for inclusion: Central to the brief's "Brownlow voting is inherently comparative" thesis; this
  category is high-priority precisely because it's both cheap to build (pure derivation, no new data) and
  directly targets a named core hypothesis of the project.
- Expected relationship: Hypothesised to outperform raw totals as predictors, per the brief's own worked
  example (30 disposals means different things in different matches) — to be tested head-to-head against raw
  totals, not assumed superior.

### Difference from best/second-best teammate, count of teammates above a performance threshold
- Data availability: D
- Reason for inclusion: Directly implements the brief's "teammate vote competition / vote stealing" concept.

---

## 11. Game state / clutch (scoped per Modelling Plan §5)

### Tier 1 — buildable now (quarter-level team score context)
- Share of match spent with scoreline within one score, all-quarter margin trajectory, half-time-to-full-time
  margin change, three-quarter-time context (leading/level/trailing).
- Data availability: A (team quarter scores are reliably public — Data Audit §1) combined with match-total
  player stats. **Not** true event-level leverage — labelled as match-context features throughout.

### Tier 2 — 2021 pilot only
- Leverage Index / Win Probability Added-style per-event weighting, Clutch Clearances/Contested
  Possessions/Tackles/Goals/Score Involvements, High-Leverage Disposals/Effective Disposals/Metres Gained.
- Data availability: P, single season, contingent on verifying player IDs and timestamps exist in the
  candidate dataset (Data Audit §2.6).
- Reason for inclusion: This is the brief's central distinguishing idea; even a single-season pilot is valuable
  to prove or disprove the concept's value before any claim about it is made across history.
- Explicit constraint: any "clutch" feature must be built as a continuous weighting, never an arbitrary
  binary threshold, per the brief.

---

## 12. Quarterly

### Q1/Q2/Q3/Q4 impact scores, Q4 share of total impact, best-quarter impact, consistency across quarters,
second-half-minus-first-half performance
- Data availability: **U for individual players** (Data Audit §1 — no public source publishes player stats by
  quarter). Only team-level quarter scores are public.
- Status: **Cannot be built as specified in the brief with current data.** This is one of the most important
  gaps to flag back to the user: the entire "quarter-by-quarter player performance" section of the brief is
  not buildable from any source found in Phase 1. Possible mitigations to evaluate in Phase 2: (a) the
  single-season 2021 play-by-play pilot dataset, if it actually contains per-player, per-event data, could be
  aggregated into per-quarter player splits for that one season only; (b) no historical mitigation currently
  identified. This should be raised explicitly as a scoping decision point, not silently dropped.

---

## 13. Role

### Player position/role (static, listed)
- Data availability: P — a position field exists on public sources; whether it reflects dynamic in-game role
  vs. a static team-sheet label needs Phase 2 confirmation.
- Reason for inclusion: Base variable for every role-interaction term below.

### Role × stat interactions (clearances × midfielder, goals × key forward, hitouts × ruck, intercepts ×
defender, contested possessions × role)
- Data availability: D, contingent on the position field above.
- Reason for inclusion: Brief explicitly wants historical umpire behaviour modelled accurately by role rather
  than assuming midfield bias — these interaction terms are how that gets tested empirically rather than
  asserted.

---

## 14. Opponent

### Opponent strength / opponent ladder position at time of match
- Data availability: D (derivable from results history — a defensible strength-of-schedule proxy, e.g. rolling
  ladder position or point differential, to be defined precisely in Phase 3, not invented here).

### Tagging matchup
- Data availability: U (Data Audit §3) — no public source found.

---

## 15. Match outcome

### Win/loss, margin, margin × dominance interactions, close-loss vs. blowout-loss distinction
- Data availability: A/D
- Reason for inclusion: Brief explicitly wants winner-advantage tested as non-constant (varying by margin,
  role, dominance, era) rather than a flat bonus — this category is how that gets estimated rather than
  assumed.

---

## 16. Historical / reputation

### Prior-season votes, career votes per game, All-Australian history
- Data availability: A/D (derivable from historical vote and award records)
- Reason for inclusion: Brief explicitly requires testing this, with a strong warning about leakage/bias risk.
- Leakage risk: **High relative to other categories** — must always be run as an explicit with/without
  ablation (Modelling Plan §4.6), never included by default without that comparison being reported alongside.

---

## 17. Umpire

### Umpire-panel identity / hierarchical umpire effects
- Data availability: P — depends on whether umpire appointments per match are publicly recorded with enough
  historical depth and consistency to support hierarchical modelling; not yet confirmed in Phase 1.
- Reason for inclusion: Brief explicitly raises this as worth investigating, with an explicit caution about
  small samples and personnel turnover.
- Status: Lower priority than other categories until umpire-appointment data availability is confirmed in
  Phase 2; must use shrinkage/hierarchical modelling if pursued at all, per the brief.

---

## 18. External human-judgement signals (not in the brief's explicit category list, but surfaced by Phase 1
research)

### AFL Coaches Association votes
- Data availability: P (Data Audit §2.9 — publicly reported weekly, no confirmed clean structured archive)
- Reason for inclusion: Independently identified in Phase 1 research as a strong predictor used by multiple
  existing public Brownlow models; timing means it is not leakage for in-season prediction.
- Caution: This is a human-judgement proxy, not a raw performance stat — must be modelled and reported as its
  own clearly labelled category, and the model's performance with vs. without it should both be reported, by
  direct analogy to the reputation-feature ablation requirement above, since leaning on it changes what the
  model is actually doing (aggregating other judges' opinions vs. modelling performance directly).

### Market-implied win probability / odds
- Data availability: A (solid from ~2013 per one free source; paid options for fuller/more recent coverage)
- Reason for inclusion: A non-fabricated, defensible prior for pre-game competitiveness expectation, usable
  in game-state/context features (Modelling Plan §5) without inventing our own win-probability model from
  nothing for the pre-game state.

---

## 19. Interaction and nonlinear-transform candidates (cross-cutting, not a separate data category)

To be tested, per the brief's explicit examples, once a modelling framework exists (Modelling Plan §2):
splines or piecewise transforms on disposal/goal counts (marginal value likely non-constant), log/quantile
transforms where distributions are skewed, and the role × stat interactions in §13. None of these are
committed to a specific functional form here — the brief is explicit that nonlinear thresholds should be
discovered empirically, not hand-specified.

---

## 20. Explicitly excluded / deferred candidates

- **Tagging matchups, metres gained, ground ball gets, pressure acts, tackles inside 50, hitouts to
  advantage:** no public data source found (Data Audit §3). Not built. Not proxied with a fabricated stand-in.
  Revisit only if Phase 2 finds a source.
- **Player-quarter-level features (§12):** not buildable with current public data outside a single-season
  pilot. Flagged to the user as a brief-vs-data-availability conflict requiring an explicit decision, not
  quietly scoped down without discussion.
- **True event-level Leverage Index / WPA across full history (§11 Tier 2/3):** not buildable beyond a
  single-season pilot without a Champion Data licensing relationship.

---

## 21. Feature-selection discipline (to apply once features are actually computed)

Per the brief: avoid uncontrolled feature explosion. Once this registry moves from "candidates" to "computed
features" in Phase 3, apply regularisation (e.g. L1/L2 for linear models, feature importance + pruning for
trees) and report which features survive selection and why, alongside — not instead of — the intuitive-prior
vs. historical-predictor vs. out-of-sample-value distinction (categories A/B/C from the brief) for each one
that is seriously considered for the final model.
