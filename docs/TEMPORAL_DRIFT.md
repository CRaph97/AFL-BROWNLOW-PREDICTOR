# Temporal Drift Analysis — Phase 3

Status: **Complete.** Machine-readable backing: `reports/season_vote_relationships.csv` (27 seasons,
1999-2025).
Last updated: 2026-09-17

## 1. Method

For every season 1999-2025 (the window where the full common stat set is available, per
`docs/DATA_COVERAGE.md`), computed: Spearman correlation of disposals/contested possessions/
clearances/goals/`possession_impact_index` with `brownlow_votes`; the winning-team polling-rate
advantage; the winning-team 3-vote-rate advantage; and (where role data exists) the midfielder share of
3-vote games. This directly informs whether/how much recency weighting matters for Phase 4, per the
brief's explicit purpose for this analysis.

## 2. Has the winner advantage changed over time?

**No — it is remarkably stable.** The winner-vs-loser polling-rate gap has stayed within a narrow
7.3-9.7 percentage-point band across all 27 seasons with no visible trend (e.g. 8.9pp in 1999, 8.2pp in
2012, 7.9pp in 2019, 8.5pp in 2025). This is a genuinely flat relationship — one of the few in this
audit — and argues against needing season-specific or heavily recency-weighted treatment for the winner
effect specifically, whatever else is decided for other features.

## 3. Has contested-ball/clearance importance changed?

**Clearances: yes, a real and fairly steady rise.** Spearman correlation with votes:

| Era | Clearances-votes Spearman (approx. range) |
|---|---|
| 1999-2009 | 0.13-0.21 |
| 2010-2019 | 0.20-0.26 |
| 2020-2025 | 0.23-0.26 |

Roughly a 30-40% relative increase in association strength from the early 2000s to the current era.
**Contested possessions themselves show a much flatter pattern** (oscillating narrowly 0.24-0.30 across
the whole period with no clear direction) — so this is specifically a **clearances** story, not a
broader "the contested game matters more now" story. A plausible football explanation (not confirmed
here, worth testing formally in Phase 4) is that clearance-differential has become a more heavily
emphasised coaching/broadcast narrative over the 2010s-2020s, which may be reflected in what umpires
notice and reward.

## 4. Has scoring importance changed?

**A mild decline in goals' association with voting** in the most recent seasons: Spearman
~0.19-0.22 through the 2000s and early 2010s, drifting down to ~0.15-0.17 in 2022-2025. This is a
smaller, less certain pattern than the clearances finding (no formal significance test was run — see
§6 caveats) but is consistent enough across the last 4-5 seasons to flag as worth monitoring, not
dismiss as noise.

## 5. Has role/position polling changed over time?

**Volatile, without a single clean trend.** Midfielder share of 3-vote games ranged from 0.46 (1999) up
to a peak around 0.71-00.75 (2013-2021), then back down to 0.56-0.68 in 2022-2025. Given only ~176-207
matches (and correspondingly few 3-vote games) per season, year-to-year swings of this size are
plausibly within normal sampling noise rather than a real structural shift — this figure should be read
as directional context alongside the much larger-sample, more reliable role finding in
`docs/ROLE_ANALYSIS.md`, not as its own strong claim.

## 6. Caveats

- No formal trend-significance testing (e.g. a fitted linear-in-season-index regression with confidence
  intervals) was performed here — patterns are described as "observed" based on visual/tabular
  inspection of `reports/season_vote_relationships.csv`, which is appropriate for Phase 3's descriptive
  purpose but should be tested formally (e.g. as an explicit `season × feature` interaction term) in
  Phase 4 before being relied upon for model design decisions.
- 2020's short, COVID-affected season (fewer, shorter matches) is included in the season-by-season table
  without special adjustment — its individual-season numbers should be read with that context in mind,
  though it does not visibly stand out as an outlier in any of the tracked relationships.

## 7. Implication for Phase 4 recency weighting

- The **winner effect** needs no special recency treatment — it is stable across the full 1999-2025
  window.
- **Clearances** show genuine drift and may benefit from recency-weighted training or a
  structural-break/era term, consistent with the brief's general caution about not pooling eras that
  behave differently.
- This is independent evidence (not assumed) supporting the brief's broader instruction (Phase 1
  `docs/MODELLING_PLAN.md` §2026 regime change) to test recency weighting / rolling-origin windows
  empirically rather than picking one window by assumption — the same principle applies even before
  the 2026 structural break specifically.
