# Data Leakage Audit — Phase 3

Status: **Complete for all features built through Phase 3.** Must be re-run for every new feature
family added in Phase 4, especially any historical-reputation or coaches-vote feature per
`docs/FEATURE_CANDIDATES.md` §16/§18.
Last updated: 2026-09-17

Classification: **SAFE** (uses only information available at/before the moment votes are cast for that
match), **CONDITIONAL** (safe for the specific use made of it so far, but would leak if reused
differently — the condition is stated explicitly), **LEAKAGE RISK** (identified but not yet built),
**EXCLUDE** (leaks and should not be used as specified).

## 1. Raw performance features (CORE/ADVANCED tables, Phase 2)

**SAFE.** Every stat is this match's own final box score — exactly the information an umpire has
available at full-time before casting votes. No feature here uses any other match, past or future.

## 2. Match-relative and team-relative features (`build_relative_features.py`)

**SAFE.** Every rank/percentile/z-score/share/gap is computed using only players who played in **this
same match** — this is not "future" information relative to the vote-casting moment, since the votes
for a match are cast only after the whole match (including every other player's final stats) has been
played. This mirrors exactly how umpires actually observe the game: they see everyone's performance
before voting, not a live in-progress subset.

## 3. Winning / margin features (`build_context_features.py`)

**SAFE.** Final score and result are known at vote-casting time by definition.

## 4. Teammate competition features (`build_teammate_features.py`)

**SAFE.** Same reasoning as §2 — computed from this match's own final team-mate stats only.

## 5. Role / position (`build_role_reference.py`, `build_role_proxy.py`)

**CONDITIONAL — the one genuine leakage risk found in this audit.**

Both the real (2021-2025) and proxy (1999-2020) role labels are built from a **full-season average**
(`season_player_profile()` in `build_role_proxy.py` aggregates every game the player played that
season, and the real-label join is similarly season-level). For a match played in, say, Round 5, this
average includes rounds 6 through 23/24 — information that does not exist yet at the time Round 5's
votes are cast.

- **Safe for how it was used in Phase 3:** all analysis in `docs/ROLE_ANALYSIS.md` and
  `docs/EXPLORATORY_ANALYSIS.md` is retrospective/descriptive (e.g. "how did midfielders poll on
  average across full seasons"), where using the full season's data to characterise that season's
  players is not leakage — it's simply the unit of analysis.
- **Would be leakage if used as-is in a forward-looking Phase 4 model** trained to predict a specific
  match's votes without knowledge of the rest of that season.
- **Required fix before Phase 4 use:** rebuild as either (a) a season-to-date-only rolling average
  (using only games up to and including the round being predicted), or (b) a fixed pre-season
  assignment (e.g. the player's role as listed at the start of the season, or their previous season's
  role), whichever the Phase 4 team judges more realistic for how role would actually be "known" at
  prediction time. **This is an explicit, tracked action item, not a silent gap.**

## 6. Composite indices (`build_composite_indices.py`)

**SAFE.** Pure functions of this match's own box-score stats (via within-match z-scores, §2's
reasoning applies identically). No Brownlow-vote information is used in their construction at any point
— confirmed by inspecting `build_composite_indices.py` directly: `brownlow_votes` never appears as an
input.

## 7. External composite benchmarks (`afl_fantasy_points`, `supercoach_points`)

**SAFE.** Both are calculated from this match's own box score by the respective external providers,
available at (indeed, typically before) full-time. Not derived from Brownlow votes or later information.

## 8. Event-level / game-state features (EXPERIMENTAL, `event_feasibility.py`)

**SAFE, by construction, for the feasibility work done so far** — every reconstructed game-state
variable (running score, quarter, time remaining) uses only events that occurred during the match being
analysed, before that match's final siren. The `simple_leverage()` placeholder likewise uses only
within-match state.

**Not yet a leakage question in practice**, since this experimental track has not been fit to any
Brownlow outcome yet (per the Phase 3 stop condition) — this classification will need re-confirming once
real leverage-weighted features (e.g. "high-leverage disposals") are actually constructed and fit in
Phase 4.

## 9. Features explicitly NOT built (named here so their absence is a decision, not an oversight)

- **Historical reputation** (prior-season votes, career votes/game, All-Australian history) — **LEAKAGE
  RISK if built carelessly**, per `docs/FEATURE_CANDIDATES.md` §16. Not built in Phase 3. If built in
  Phase 4, must use only information available strictly before the match being predicted (e.g. prior
  *completed* seasons' totals, never the current season's future rounds), and must be run as an explicit
  with/without ablation per `docs/MODELLING_PLAN.md` §4.6.
- **AFL Coaches Association votes** — not leakage in principle (published within days of each match,
  well before the Brownlow count), but not built in Phase 3 due to the unresolved data-availability gap
  in `docs/DATA_SOURCE_AUDIT.md` §2.9.
- **Season-end awards, final ladder position, media Brownlow predictions** — never used anywhere in this
  project's data pipeline. Confirmed absent from every `src/features/build_*.py` and
  `src/data/build_*.py` script by direct inspection while writing this audit.

## 10. Summary table

| Feature family | Classification | Action needed |
|---|---|---|
| Raw performance (CORE/ADVANCED) | SAFE | None |
| Match/team-relative | SAFE | None |
| Win/margin context | SAFE | None |
| Teammate competition | SAFE | None |
| **Role/position** | **CONDITIONAL** | **Rebuild as season-to-date/pre-season before Phase 4 forward-looking use** |
| Composite indices | SAFE | None |
| External fantasy composites | SAFE | None |
| Event/game-state (experimental) | SAFE (nothing fit yet) | Re-audit once leverage-weighted features are fit to Brownlow outcomes |
| Historical reputation | Not built | Build only with strict past-only windowing + mandatory ablation, if pursued |
| Coaches votes | Not built | Blocked on data availability, not a leakage issue |
