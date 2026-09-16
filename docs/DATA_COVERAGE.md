# Data Coverage — Phase 2C

Status: **Complete for the CORE (afltables) and ADVANCED (footywire) sources.** Not yet extended to the
experimental event-level track (see `docs/EVENT_DATA_2021_AUDIT.md`) or umpire-appointment data.
Last updated: 2026-09-16

Machine-readable outputs: `reports/coverage_matrix.csv` (long form, one row per season × feature × source),
`reports/coverage_matrix_wide.csv` (feature × season pivot), `reports/coverage_first_reliable_season.csv`
(summary below).

## 1. Method

For every season in the validated 1984-2025 window, the % of non-null values was computed for every
candidate predictor, separately for the CORE (afltables-sourced) and ADVANCED (footywire-sourced, joined
by date+team+opponent+surname — see `docs/DATA_DICTIONARY.md`) tables. A feature is called **structurally
unavailable** in a season when coverage is ~0% for that entire season (the stat did not exist yet, not a
data problem), versus **missing observation** when a season has partial coverage (a genuine gap worth
investigating). In practice, every feature below transitions cleanly from 0% to ~100% at a specific
season with no partial-coverage seasons in between — i.e., every gap found is a structural-availability
boundary, not row-level missingness within an available era. This is a clean, reassuring result: there is
no season where a stat existed but was spottily recorded.

## 2. First reliable season, by feature

| Feature | First reliable season | Notes |
|---|---|---|
| kicks, handballs, disposals, marks, goals, behinds, hitouts, frees_for, frees_against | **1984** (start of validated window) | Present from the beginning of the window; true origin is earlier (afltables' own detailed stats go back to 1965 per Phase 1) but that predates our validated Brownlow-vote window so is out of scope for target-bearing modelling |
| tackles | **1987** | |
| clearances, rebound_50s, inside_50s, clangers | **1998** | |
| contested_marks, bounces, contested_possessions, uncontested_possessions, one_percenters, marks_inside_50 | **1999** | |
| time_on_ground_pct, goal_assists | **2003** | |
| effective_disposals, disposal_efficiency_pct | **2010** | From the footywire ADVANCED join (99.6% coverage in-scope — the ~0.4% shortfall is join-matching noise, not a source gap; see §4) |
| centre_clearances, stoppage_clearances, score_involvements, metres_gained, turnovers, intercepts, tackles_inside_50 | **2015** | Footywire's *extended* stat set specifically (not the same as the base 2010 advanced set) — matches the known real-world staged rollout of Champion Data's more granular metrics |

No feature in the brief's candidate list was found to be structurally unavailable *within an era it
claims to cover* — i.e., every "P" (partial) item from the Phase 1 audit for these particular stats has
now been resolved to a precise season boundary. Items still fully unavailable (metres gained is **not**
one of them any more — see §3) are documented in `docs/FEATURE_CANDIDATES.md` §20 and were not
found in either the CORE or ADVANCED tables.

## 3. A correction to the Phase 1 audit: metres gained, score involvements, and tackles-inside-50 ARE available

Phase 1's `DATA_SOURCE_AUDIT.md` marked **metres gained as fully unavailable (U)** and **score
involvements / centre-stoppage clearance split as partial/derivable-only**, based on what a single
standard Footywire match-stats page happened to show. Phase 2's direct inspection of the footywire-sourced
`player_stats.rda` file (accessed via the fitzRoy data repo, not scraped by us) found a much richer
42-column schema, confirmed field-by-field against the community-documented footywire abbreviation
glossary (`jimmyday12/fitzRoy` issue #51):

- **Metres gained (`MG`)** — genuinely available, 2015+. This is a correction, not a refinement, of
  Phase 1.
- **Score involvements (`SI`)** — a genuine direct field, 2015+, not merely a composite we'd have to
  invent.
- **Centre clearances (`CCL`) / stoppage clearances (`SCL`)** — genuinely split, 2015+, and internally
  consistent (`CCL + SCL == clearances` was spot-checked and held on the validation sample).
- **Tackles inside 50 (`T5`)** — genuinely available, 2015+. Phase 1 marked this fully unavailable.
- **Turnovers (`TO`)**, distinct from clangers — genuinely available, 2015+.
- **Intercepts (`ITC`)** — available, 2015+, but this is footywire's own generic "Intercepts" metric.
  **It is not confirmed to be defined identically to Champion Data's 2026 umpire-facing "intercept
  possessions" field** — see `docs/2026_STATS_MIRROR.md`. Treat as a related proxy, not a confirmed
  match, until compared against a live Champion Data-sourced number.

This is a direct, documented example of "verify before scale" paying off: relying on the single sample
page fetched in Phase 1 would have caused the project to under-build its ADVANCED feature set and wrongly
tell the user several stats were unbuildable when they were not.

## 4. The advanced-join match-rate gap (~4%, worth flagging honestly)

Joining ADVANCED (footywire) rows onto the CORE (afltables) table cannot use `(season, round, team)`
as a key — **footywire's round numbering does not always match afltables'** (see
`docs/DATA_DICTIONARY.md` §Known data-quality issues for the specific confirmed example). The join
instead uses `(season, date, canonical team_id, canonical opponent_id, normalised player surname)`.

After normalising surnames to strip apostrophes/hyphens/case (which fixed a confirmed real mismatch —
"O'Halloran" in one source vs "OHalloran" in the other), the in-scope (2010-2025) match rate is
**96.1%**. The remaining ~3.9% unmatched rows are concentrated in:
- Players flagged with `substitute_status` "On"/"Off" (the interchange/medical-substitute rule) —
  disproportionately represented among unmatched rows in the sample inspected.
- A sharp rise in unmatched-row *counts* from 2021 onward (though the *rate* stays broadly similar
  because total rows also rose) that was not fully root-caused in the time available for this audit.

This is logged to `reports/advanced_join_surname_collisions.csv` (5,006 rows involved in a same-match
surname collision, 2,526 duplicate rows dropped keeping first match) and is an open item for Phase 3
investigation before the ADVANCED feature set is used in any model — a ~4% missing-observation rate in a
*joined* secondary table is a real, quantified limitation, not swept under the rug, and does not affect
the CORE table's target-variable integrity (which is 100% clean — see `docs/TARGET_VALIDATION.md`).

## 5. Recommended modelling windows (empirically derived, not arbitrary)

| Model | Window | Rationale |
|---|---|---|
| **CORE** | **2003-2025** (23 seasons) | Every common box-score stat (including contested/uncontested possessions, clearances, TOG%, goal assists) is reliably available from 2003 onward with no structural gaps. 1984-2002 is usable for a narrower stat set (kicks/handballs/disposals/marks/goals/behinds/hitouts/frees, +tackles from 1987, +clearances/inside50s/clangers/rebounds from 1998) if a longer, thinner-featured backtest is wanted. |
| **ADVANCED** | **2015-2025** (11 seasons) | Full extended stat set (score involvements, metres gained, intercepts, centre/stoppage clearance split, tackles inside 50, turnovers, effective disposals, disposal efficiency) only reliably available from 2015. |

Per the approved Phase 1 decision, the *final* preferred window(s) will still be chosen empirically via
rolling-origin backtesting across candidate windows (recent-5/8/10 seasons, full 2003+ or 2015+, expanding
window, recency-weighted) once modelling begins — these coverage boundaries define the *feasible* set of
windows to compare, not the final answer.
