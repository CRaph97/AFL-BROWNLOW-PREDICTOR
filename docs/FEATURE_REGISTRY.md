# Feature Registry — Phase 3

Status: **Complete for the features actually built.** Machine-readable companion:
`reports/feature_availability.csv` (134 features, per-feature first/last reliable season) and
`reports/candidate_feature_sets.csv` (proposed CORE/ADVANCED/EXPERIMENTAL membership).
Last updated: 2026-09-17

This registry documents what was actually built in `src/features/`, not the full aspirational list
from `docs/FEATURE_CANDIDATES.md` (Phase 1) — some Phase 1 candidates remain unbuildable (see that
document's §20) and are not repeated here.

## How to read this registry

Every feature below has: **definition**, **source**, **transformation**, **family**, **years
available**, **expected interpretation** (a stated prior, not a result), **leakage risk**, and which of
CORE / ADVANCED / EXPERIMENTAL it's proposed for. Actual observed relationships and Phase-4 importance
are **not** filled in here — see `docs/EXPLORATORY_ANALYSIS.md` for what was actually found, kept
separate so a reader can't confuse "why we tried this" with "what we found."

## 1. Raw performance features (`data/processed/player_match_core_1984_2025.parquet` /
`player_match_advanced_2010_2025.parquet`, unmodified from Phase 2)

All 24 CORE columns and 17 ADVANCED columns from `DATA_DICTIONARY.md` are retained as-is at the
player-match grain — no re-derivation needed, since Phase 2 already built them correctly. Family
tags and exact first-reliable-season per feature: see `reports/feature_availability.csv`. Highlights:
- **Excluded from further transformation:** `clangers`, `frees_for`, `frees_against`, `bounces`,
  `jumper_number`, `substitute_status` — kept in the analytical table as raw context but not carried
  into match-relative/composite treatment (§2/§6), since exploratory review found them low-salience
  and not clearly theoretically distinct from stats already covered (documented exclusion, not an
  oversight).
- `hitouts` is deliberately **not** given match-relative treatment (§2) — it is extreme and
  positionally concentrated (rucks vs everyone else) rather than generally informative in a rank/z-score
  sense; it is instead a direct input to the role-proxy classifier (§5).

## 2. Match-relative and team-relative features (`src/features/build_relative_features.py`)

Applied to 8 curated base stats — **disposals, contested_possessions, clearances, tackles, goals,
inside_50s, contested_marks, marks** — chosen for plausible Brownlow relevance and mutual
non-redundancy; not applied to every available stat, per the brief's explicit warning against feature
explosion (kicks/handballs excluded as largely redundant with disposals; one_percenters/bounces
excluded as sparse and low-salience; hitouts excluded per §1).

| Feature suffix | Definition | Family | Leakage risk |
|---|---|---|---|
| `_match_rank` | Rank of this player's stat among all players in the match (1 = highest) | match_relative | Safe — uses only this match's own final stats, available post-match exactly as umpires see them |
| `_match_pct` | Percentile rank within the match, (0,1] | match_relative | Safe |
| `_match_z` | (value − match mean) / match std | match_relative | Safe |
| `_team_rank` | Rank within the player's own team only | team_relative | Safe |
| `_team_share` | value / team total | team_relative | Safe |
| `_gap_best_team` | value − best teammate's value (excl. self) | teammate_competition | Safe |
| `_gap_2nd_team` | value − second-best teammate's value (excl. self) | teammate_competition | Safe |
| `_opp_best_gap` | value − best value on the opposing team | opponent_relative | Safe |

Expected interpretation (stated prior, tested in `docs/EXPLORATORY_ANALYSIS.md`): these should
outperform raw totals because "32 disposals when no one else exceeded 24" and "32 disposals in a
shootout with six players at 30+" are different signals that raw totals cannot distinguish.

## 3. Winning / margin features (`src/features/build_context_features.py`)

`is_win`, `is_loss`, `is_draw` (from the CORE `win_loss_draw` field), `is_close_game`
(`absolute_margin <= 12`, one exploratory cut — a two-scoring-shot game), `is_blowout`
(`absolute_margin >= 50`). Both thresholds are explicitly **exploratory, not sacred** per the brief,
and are re-examined via continuous margin bins in `docs/EXPLORATORY_ANALYSIS.md` rather than only via
these two cut-offs. Family: `match_outcome`. Leakage risk: Safe (final score is known at vote time).

## 4. Teammate competition features (`src/features/build_teammate_features.py`)

`n_teammates_disposals_ge_25`, `n_teammates_disposals_ge_30`, `n_teammates_goals_ge_2`,
`n_teammates_goals_ge_3` (counts EXCLUDING self), and `team_disposal_concentration` (a
Herfindahl-style index: sum of each team-mate's squared share of team disposals — higher means
output is concentrated in fewer players). Thresholds (25/30 disposals, 2/3 goals) are explicitly
exploratory per the brief; continuous alternatives (`_gap_best_team` etc., §2) are the
non-threshold comparison point. Family: `teammate_competition`. Leakage risk: Safe.

## 5. Role / position (`src/features/build_role_reference.py`, `build_role_proxy.py`)

- **2021-2025: real, officially-listed AFL position** (`KEY_FORWARD`, `MEDIUM_FORWARD`, `MIDFIELDER`,
  `MIDFIELDER_FORWARD`, `RUCK`, `KEY_DEFENDER`, `MEDIUM_DEFENDER`), sourced via the `torp`/`torpdata`
  ecosystem already validated in Phase 2, joined by normalised surname + team + season.
- **1999-2020: a statistically-derived proxy** — a shallow (max_depth=4) decision tree trained on
  season-averaged, within-season-z-scored box-score profiles (hitouts, marks_inside_50, goals,
  rebound_50s, one_percenters, inside_50s, clearances, contested_possessions, disposals,
  uncontested_possessions), fit on the 2021-2025 real-label seasons and applied backward.
  **Grouped (by player) 5-fold cross-validated accuracy: 76.8%** — see
  `reports/role_proxy_confusion_matrix.csv` and `reports/role_proxy_classification_report.txt` for the
  full per-class breakdown. This is a genuinely useful but imperfect proxy, not ground truth — treated
  and reported as such throughout `docs/ROLE_ANALYSIS.md`.
- **1984-1998 and 2026: no role classification** — the input stats aren't structurally available
  before 1999 (see `docs/DATA_COVERAGE.md`), and 2026 role labels weren't downloaded for this audit.
- **Family:** `role`. **Leakage risk: CONDITIONAL** — both the real and proxy labels are built from a
  **full-season aggregate**, which for a genuinely forward-looking (in-season) Phase 4 model would use
  information from rounds after the one being predicted. Safe for the retrospective exploratory
  analysis in this document; **must be rebuilt as a season-to-date or pre-season-only calculation**
  before any real-time Phase 4 use. Flagged explicitly in `docs/LEAKAGE_AUDIT.md`.

## 6. Composite indices (`src/features/build_composite_indices.py`)

Six transparent, EXPLORATORY-only indices, each the sum of its components' **within-match** z-scores
(never raw sums, so they're comparable across eras/paces automatically): `possession_impact_index`
(disposals, contested_possessions, uncontested_possessions), `contest_index` (contested_possessions,
tackles, clearances), `clearance_index` (clearances [+ centre_clearances where available]),
`scoring_index` (goals, 0.5×behinds, goal_assists), `territory_index` (inside_50s, rebound_50s [+
metres_gained where available]), `defensive_index` (one_percenters, rebound_50s, tackles). Each carries
a `_n_components` companion column recording how many inputs were actually summed (so an era-thin index
value is distinguishable from a full one). **Not** final Brownlow predictors — explicitly built and
labelled as exploratory dominance/concentration tools per the brief. Family: `composite`. Leakage risk:
Safe (pure box-score function, no Brownlow information used in construction).

## 7. External composite benchmarks (already in the ADVANCED table, not separately built)

`afl_fantasy_points`, `supercoach_points` — retained and analysed (they turned out to be the single
strongest univariate correlates found, see `docs/EXPLORATORY_ANALYSIS.md`) but flagged: highly
correlated with each other (r=0.86) and are themselves external composite formulas, not Brownlow-vote
information — using one as a feature is legitimate (contemporaneous, no leakage) but including both is
redundant.

## 8. Explicitly NOT built in Phase 3 (deferred, not forgotten)

- **Historical reputation / prior-season votes** — deliberately deferred; the brief's own leakage
  caution (Phase 1 `FEATURE_CANDIDATES.md` §16) applies, and Phase 3's brief did not ask for these to be
  built yet.
- **AFL Coaches Association votes** — data-availability gap identified in Phase 1, unresolved, not
  built.
- **True event-level leverage/WPA features** — deliberately kept OUT of the analytical dataset per
  Phase 3 section O's explicit instruction; feasibility-only work is in
  `src/features/event_feasibility.py` and documented separately (§EXPERIMENTAL below and
  `docs/EXPLORATORY_ANALYSIS.md` §Event data).

## 9. EXPERIMENTAL feature family (not merged into the analytical dataset)

Game-state reconstruction from the `torp`/`torpdata` chains feed (2021-2026) — running score
differential, quarter, time remaining, lead changes, tied-score state — and a minimal, unfit
`simple_leverage(absolute_margin, period, period_seconds)` placeholder function (closeness × time
elapsed, not fit to any Brownlow outcome). See `docs/EXPLORATORY_ANALYSIS.md` §Event data feasibility
for the reconciliation-accuracy findings that determine whether this family is trustworthy enough to
build real leverage features from in Phase 4.
