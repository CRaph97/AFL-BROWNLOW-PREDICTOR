# 2026 Data Validation — Phase 5

Status: **Complete.**
Last updated: 2026-09-17

## 1. Internet / source availability check (done first, per the brief's instruction to check this
before committing to a 2026 forecast)

The same fitzRoy data mirror used throughout this project (`jimmyday12/fitzroy_data`, updated
nightly, sourced from afltables/footywire "with permission") already contains the **complete, real
2026 home-and-away season** at the time this phase ran (2026-09-17) — the AFL home-and-away season
had already finished and finals were underway (`EF`/`QF`/`SF`/`Wildcard Final` rows present, dated up
to 2026-09-12; the home-and-away rounds run 2026-03-05 to 2026-08-23). **No fallback to a proxy season
was necessary.** This is genuine 2026 data, not a synthetic stand-in — see `reports/2026_data_extension_validation.json`
for the raw counts below.

## 2. Home-and-away match coverage

- **207 matches**, 25 rounds, 18 teams, 2026-03-05 to 2026-08-23.
- Every one of the 18 teams played **exactly 23 matches** (no team missing a bye-adjusted share of
  the season).
- Every match has **exactly 46 player rows** (std = 0 across all 207 matches) — no partial-roster or
  truncated-box-score matches.
- No finals rounds (`EF`, `QF`, `SF`, `Wildcard Final` — 2026 introduced this fifth finals round
  format) are included; Brownlow votes are never awarded for finals in any season.

## 3. Missing statistics / identity resolution

- `player_id` (afltables numeric ID): **82 of 9,522 rows (0.86%)** have no ID yet in the afltables
  crosswalk (consistent with the historical pattern of very recent debutants not yet catalogued — 79
  rows were found in this same situation for 2025 during Phase 2). Each gets a unique
  `NOID2026_<row>` placeholder, never silently merged with another player.
- Core box-score stats (kicks, handballs, disposals, marks, contested marks, tackles, goals, behinds,
  goal assists, clearances, contested possessions): **zero missing values** across all 9,522 2026 rows.
- ADVANCED (footywire-sourced) stats join match rate for 2026: **98.1%** (matches the historical
  footywire-join reliability documented in `docs/DATA_COVERAGE.md`).
- Brownlow votes for every 2026 row: confirmed all null (asserted programmatically in
  `build_2026_extension.py` — the count has genuinely not happened yet).

## 4. Team and round identity integrity

All 2026 team names (`Adelaide`, `Brisbane Lions`, `Carlton`, `Collingwood`, `Essendon`, `Fremantle`,
`Geelong`, `Gold Coast`, `Greater Western Sydney`/`GWS`, `Hawthorn`, `Melbourne`, `North Melbourne`,
`Port Adelaide`, `Richmond`, `St Kilda`, `Sydney`, `West Coast`, `Western Bulldogs`) already exist in
`config/team_mapping.csv` from prior seasons — no new club names or relocations to map. Round
numbering is 1-25, no gaps, consistent with afltables' scheme used throughout the project.

## 5. Role information for 2026

`role_reference_2021_2025.parquet` (the real, torpdata-sourced position roster) does not extend to
2026. **Every 2026 row's `role` feature is produced by the lagged, prior-games-only proxy classifier**
(Tier B of `build_role_lagged.py`'s hierarchy) — never a real label. Players with fewer than 3 games
of 2026-season history (i.e. the first 3 rounds of each player's 2026 season) have role = `UNKNOWN`.
See `reports/2026_role_source_breakdown.txt` for the exact real/proxy/unknown split.

## 6. 2026 umpire-visible-stat availability (recap, full detail in `docs/2026_STRUCTURAL_BREAK.md`)

14 of the 17 confirmed umpire-visible statistics are available for 2026 (directly or via a documented
proxy); kick-ins, intercept marks and spoils are not (they require live-scraping the AFL's
semi-public API, out of scope for this run). This directly bounds what Scenario C ("stats-assisted")
can represent — disclosed, not hidden.

## 7. Feature completeness at inference time

Unlike TRAINING data (where an incomplete-feature row is safely dropped because the match still has
plenty of other players), a 2026 PREDICTION row cannot simply be dropped without silently removing a
real player from that match's forecast. See `reports/2026_feature_completeness.txt` for the exact
count of 2026 rows with complete vs. incomplete lagged-form/role features (typically only affects
players in their first 1-2 games of the season, where a `_prev3_mean`/`_prev5_mean`/`_prev10_mean`
window cannot yet be filled) — these players are excluded from the probability tables with the
reason logged, not silently guessed.

## 8. Verdict

**No blocking data-integrity issue was found.** The 2026 home-and-away season is complete, genuinely
real, and of comparable quality to any historical season in this project. The production forecast in
`docs/2026_FINAL_REPORT.md` proceeds on this data with the documented, non-blocking limitations above
(role proxy-only, 3 of 17 umpire stats unavailable, a small number of early-season debutant rows
excluded from probability tables for insufficient history).
