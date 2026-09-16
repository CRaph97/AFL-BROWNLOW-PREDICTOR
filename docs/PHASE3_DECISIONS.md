# Phase 3 Decisions — Feature Sets for Phase 4

Status: **Proposal, grounded in the exploratory findings, for user sign-off before Phase 4.**
Machine-readable backing: `reports/candidate_feature_sets.csv` (134 features classified).
Last updated: 2026-09-17

## 1. CORE feature set (2003-2025 window)

All features with confirmed >=95% coverage from 2003 onward (109 of 134 audited features qualify —
see `reports/candidate_feature_sets.csv`, `proposed_set == "CORE + ADVANCED"`). Includes:
- Every CORE raw stat (disposals, kicks, handballs, marks, goals, behinds, hitouts, tackles,
  rebound_50s, inside_50s, clearances, clangers, frees for/against, contested/uncontested possessions,
  contested marks, marks inside 50, one-percenters, bounces, goal assists, TOG%).
- All 64 match/team-relative features (§2 of `docs/FEATURE_REGISTRY.md`) on the 8 curated base stats.
- All win/margin context features, all teammate-competition features.
- Role (with the leakage caveat from `docs/LEAKAGE_AUDIT.md` §5 — must be rebuilt season-to-date before
  Phase 4 forward-looking use).
- All 6 composite indices (using only their CORE-available components).

**Recommended priority features for Phase 4's first model**, based on the strongest, cleanest
exploratory signal found: `disposals_match_z`, `contested_possessions_match_z`, `possession_impact_index`,
`goals` (with explicit nonlinear/piecewise treatment per §5 of `docs/EXPLORATORY_ANALYSIS.md`), `margin`
and `absolute_margin` (with the win × margin interaction), `n_teammates_disposals_ge_30` or
`disposals_gap_best_team`, and `role`.

## 2. ADVANCED feature set (2015-2025 window)

CORE set plus the 19 footywire-extended-stat-era features (`proposed_set == "ADVANCED only"`):
`effective_disposals`, `disposal_efficiency_pct`, `centre_clearances`, `stoppage_clearances`,
`score_involvements`, `metres_gained`, `turnovers`, `intercepts`, `tackles_inside_50`, plus their
match-relative/composite derivatives where built, and `afl_fantasy_points`/`supercoach_points` (pick
**one**, not both — r=0.86 redundant per `docs/EXPLORATORY_ANALYSIS.md` §9).

`score_involvements` and `metres_gained` are recommended high priority for this set — both showed
strong, clean univariate association (Spearman 0.285 and 0.260 respectively) and represent genuinely
new information not available in the CORE set.

## 3. EXPERIMENTAL feature set (2021-2026, pilot only)

Not yet built as trainable features — feasibility only, per Phase 3's explicit instruction. Confirmed
feasible with the corrected chain-level logic (`docs/EXPLORATORY_ANALYSIS.md` §10): running score
differential, quarter, time remaining, lead-change flags, tied-score flags, and a placeholder leverage
function. **Recommended Phase 4 next step for this track**: build the actual "Clutch
Clearances"/"High-Leverage Disposals"-style features described in `docs/MODELLING_PLAN.md` §5 Tier 2,
fit a model with vs. without them on the 2021-2025 overlap window, and report the answer to "does
intra-match timing information materially improve prediction?" honestly either way — this has still not
been tested, only shown to be technically buildable.

## 4. Features to exclude or de-duplicate before Phase 4

- **6 metadata-only columns** (`*_n_components` companions to the composite indices) — informational,
  not predictive signal, excluded from modelling but kept in the table for interpretability.
- **Redundancy-flagged pairs** (`docs/EXPLORATORY_ANALYSIS.md` §9): keep `clearances` over
  `stoppage_clearances` (or use the split deliberately if centre-vs-stoppage distinction matters
  theoretically, not both plus the total blindly); keep `disposal_efficiency_pct` alongside disposals
  rather than also including `effective_disposals` (r=0.92 with raw disposals — likely redundant once
  disposals and efficiency are both in the model); keep one of `afl_fantasy_points`/`supercoach_points`.
- **`bounces`, `clangers`, `frees_for`, `frees_against`** — retained in the table but not given
  match-relative/composite treatment; deprioritise for Phase 4's first model pass based on low
  exploratory salience, revisit only if a specific football hypothesis calls for them.

## 5. What Phase 3 could NOT resolve (open items for Phase 4 or beyond)

1. **Role leakage fix** (`docs/LEAKAGE_AUDIT.md` §5) — must be done before role is used in any
   forward-looking model, not after.
2. **MIDFIELDER_FORWARD proxy is unreliable** (F1 = 0.06) for 1999-2020 — decide whether to collapse it
   into a broader category for that era or accept the noise.
3. **Two 2024 matches showed large, undiagnosed event-data score-reconciliation failures** — worth a
   quick follow-up look before building real experimental features on the full 2021-2026 window, in
   case they represent a broader pattern rather than two isolated bad matches.
4. **The mild recent decline in goals' vote-association** (`docs/TEMPORAL_DRIFT.md` §4) was not
   significance-tested — worth confirming formally if goal-related features are weighted heavily in
   Phase 4.
5. **No formal multivariate model has been fit** — every finding in this phase is univariate or simple
   cross-tabulation. Redundancy, interaction, and role effects that look large univariately may
   partially or fully explain each other once combined; only Phase 4's actual model fitting (with proper
   regularisation and importance analysis) will resolve which features carry genuinely independent
   signal, per the brief's own repeated caution not to confuse univariate association with final
   predictive value.

## 6. Decisions needed from the user before Phase 4

1. Accept the CORE (2003-2025) / ADVANCED (2015-2025) feature-set proposals as the Phase 4 starting
   point?
2. Accept fixing the role-leakage issue as a Phase 4 prerequisite task (not optional)?
3. How to handle the MIDFIELDER_FORWARD proxy unreliability — collapse the category pre-2021, exclude
   it, or accept the noise with a documented caveat?
4. Proceed with building real (fitted, tested) EXPERIMENTAL leverage features in Phase 4 per §3's
   recommendation, or defer that track further?
5. Any features from §4's exclusion list you want reinstated for a specific reason not captured here?
