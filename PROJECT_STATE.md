# Project State — AFL Brownlow Predictor

Last updated: 2026-09-16
Current phase: **Phase 2 (Target + Data Foundation) complete, pending user review. Modelling has not
started — per the stop condition, no feature weighting, model fitting, predictions, or simulations have
been done.**

---

## What exists right now

- A working Python environment (`.venv/`, `requirements.txt` — pinned, minimal: pandas, pyarrow,
  pyreadr, requests, lxml/bs4/html5lib, pytest. No modelling libraries yet, per the "add only when
  needed" decision.)
- Real, validated data:
  - `data/raw/fitzroy_data/` — downloaded via `src/data/fetch_fitzroy_data.py`, with SHA-256 provenance
    in `PROVENANCE.json`.
  - `data/raw/torpdata_pilot/` — one season (2024) of the experimental event-level dataset, downloaded
    for validation only.
  - `data/interim/player_match_afltables_1984_2025.parquet` — unfiltered-column intermediate.
  - `data/processed/player_match_core_1984_2025.parquet` — **the canonical target-bearing dataset**,
    320,861 rows, 7,413 matches, 1984-2025, home-and-away only.
  - `data/processed/player_match_advanced_2010_2025.parquet` — CORE plus footywire advanced columns.
- `config/team_mapping.csv` — the explicit, documented team-identity canonicalisation table.
- `src/data/` — four reproducible scripts: `fetch_fitzroy_data.py`, `build_core_dataset.py`,
  `build_advanced_dataset.py`, `build_coverage_matrix.py`. Re-running them regenerates everything above
  from raw data.
- `tests/test_core_dataset_integrity.py` — 10 automated QC checks, **all passing**.
- `reports/` — machine-readable outputs: `coverage_matrix.csv` / `coverage_matrix_wide.csv` /
  `coverage_first_reliable_season.csv`, plus two logged-for-review exception files
  (`identity_review_missing_id.csv`, `advanced_join_surname_collisions.csv`).
- Nine documents: the four from Phase 1 (now with a Phase 2 addendum on `DATA_SOURCE_AUDIT.md`) plus
  five new ones: `docs/TARGET_VALIDATION.md`, `docs/DATA_COVERAGE.md`, `docs/2026_STATS_MIRROR.md`,
  `docs/EVENT_DATA_2021_AUDIT.md`, and an updated `DATA_DICTIONARY.md`.

Nothing has been committed to git yet (only requested when you ask).

---

## Answering the Phase 2 stop-condition questions

**1. Were Brownlow votes successfully joined to real player-match records?**
Yes. MATCH → PLAYER → PLAYER MATCH STATISTICS → BROWNLOW VOTES is established and reproducible for
**1984-2025 home-and-away matches** (320,861 rows, 7,413 matches). Earlier seasons exist in the raw data
but have no match-level vote detail (see #1 below) — this is a real historical limit, not a build choice.

**2. Validation examples**
Three matches spanning three eras (1991, 1992, 2024) were checked line-by-line against **live AFL
Tables pages**, not just re-reads of the same scrape — every field matched exactly, including a
same-team, same-surname pair (the Daniher brothers, 1992) correctly disambiguated by stable player ID.
Two independent season-total facts (Dangerfield 2016 = 35 votes, Neale 2020 = 31 votes from 17 games)
were cross-checked against cited public reporting and matched exactly. Full detail: `docs/TARGET_VALIDATION.md`.

**3. Discovered integrity problems**
- A pandas gotcha where missing player IDs (NaN) falsely appeared as "duplicate players" — found, fixed,
  covered by a regression test.
- Round numbering differs between afltables and footywire/the AFL's own API by up to one round per
  season (Opening-Round-as-Round-0 offset) — found via a real example, now a hard rule in all join code
  (always join on date, never round number).
- A genuine historical anomaly (a 1928 finals replay sharing one round label across two dates) — outside
  the modelling window, documented rather than silently absorbed.
- A confirmed ~4% (96.1% match rate) shortfall joining footywire's advanced stats onto the core table,
  concentrated in substitute-flagged players — quantified and logged, not hidden.
- A confirmed ~17% under-count reconstructing behinds (not goals) from the experimental event dataset's
  naive event descriptions, and a null-column bug in one of that dataset's convenience fields — both
  documented with the specific fix needed before anyone relies on them.

**4. Final source choices**
Bulk layer: the community `fitzRoy` data repository (MIT-adjacent "with permission" sourcing from AFL
Tables + Footywire), not bespoke scraping — per your Phase 1 decision. Every critical fact (target
variable, identifiers, scores) independently re-verified against afltables.com directly, not trusted
blindly. Experimental event layer: `peteowen1/torp`/`torpdata` (MIT-licensed, actively maintained),
which superseded the single-season candidate found in Phase 1.

**5. Feature coverage by season**
Full matrix in `reports/coverage_matrix.csv`. Clean, sharp transition seasons found for every feature —
no partial-coverage "in-between" seasons, only a lower-common-stat era (1984+) and progressively richer
eras arriving in 1987, 1998, 1999, 2003, 2010, and 2015. Full table in `docs/DATA_COVERAGE.md` §2.

**6. Recommended modelling start years**
- **CORE model: 2003-2025** (23 seasons) — every common box-score stat available with no gaps.
  1984-2002 usable as a longer, thinner-featured extension if wanted.
- **ADVANCED model: 2015-2025** (11 seasons) — the full extended stat set (score involvements, metres
  gained, intercepts, centre/stoppage clearance split, tackles inside 50, turnovers) only reliable from
  2015. This is a starting point for the rolling-origin comparison your Phase 1 decision already called
  for (recent-5/8/10, expanding window, recency-weighted) — not a final answer, since that requires
  actual backtesting in Phase 3.

**7. Is the 2021 event dataset credible?**
The original single-season candidate: no. A materially better replacement was found and validated
instead: `torp`/`torpdata`, MIT-licensed, actively maintained, covering **2021-2026** (six seasons, not
one). Validated against our own already-confirmed ground-truth match with exact agreement on
disposals/kicks/handballs/goals, a specific and fixable ~17% behind-undercount, and one confirmed
column-level bug. Verdict: **valid enough for the experimental pilot track**, not a CORE/ADVANCED
data source. Full detail: `docs/EVENT_DATA_2021_AUDIT.md`.

**8. Which 2026 umpire-visible statistics can we actually reproduce historically?**
13 of the 17 directly, with well-established multi-year/decade coverage. The remaining 4 (kick-ins,
intercept marks, intercept possessions, spoils) — which Phase 1 wrongly called fully unreproducible —
are exposed by the AFL's own public API, confirmed live for 2025-2026, but that API's *historical* depth
has not yet been established, so they should be treated as an enrichment layer pending that check, not
assumed available for backtesting yet. Full table: `docs/2026_STATS_MIRROR.md`.

**9. Remaining material limitations**
- ADVANCED-table join has a ~4% unresolved gap (root cause only partly diagnosed).
- The AFL public API's historical depth for the 4 previously-"unavailable" 2026 stats is unknown.
- The experimental event dataset only reaches back to 2021 and has two documented, specific defects.
- Player-quarter-level statistics remain entirely unavailable (unchanged from Phase 1) — still an open
  decision for you (drop the brief's quarter-by-quarter section, restrict to whatever the event-level
  pilot can approximate, or pause for a paid source).
- Umpire-appointment historical data availability (for hierarchical umpire-effect modelling) has not yet
  been investigated at all.
- No 2026 Brownlow votes exist yet — the regime-change question remains untestable until the season ends.

---

## Decisions needed from you before Phase 3 (modelling) begins

1. Accept the CORE (2003-2025) / ADVANCED (2015-2025) starting windows as the candidate set for
   rolling-origin backtesting, rather than a single fixed window?
2. Proceed with the `torp`/`torpdata` experimental event-level track as scoped in
   `docs/EVENT_DATA_2021_AUDIT.md` §6 (build Tier-2 leverage/WPA features on 2021-2026 only, compare
   against the CORE/ADVANCED-only model, report the answer either way)?
3. How to handle the still-unresolved player-quarter-level gap?
4. Any objection to closing out the ~4% ADVANCED-join gap and the AFL-API historical-depth question as
   background Phase 3 investigation rather than blocking modelling on them now?

## Next steps (Phase 3, not started)

Model comparison per `docs/MODELLING_PLAN.md` §2 (baseline → boosting → ranking model → game-state
variant → hierarchical if needed), rolling-origin validation per §4, on the CORE/ADVANCED windows
established above. No feature weighting, model fitting, or predictions until you confirm the above.
