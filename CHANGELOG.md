# Changelog

## Unreleased

- Phase 1: repository scaffolding created, no data or code yet.
- Phase 1: `docs/DATA_SOURCE_AUDIT.md`, `docs/MODELLING_PLAN.md`, `docs/FEATURE_CANDIDATES.md`,
  `PROJECT_STATE.md` written.
- Phase 2: canonical CORE (1984-2025) and ADVANCED (2010-2025) player-match datasets built and
  validated against live AFL Tables pages; target-variable integrity confirmed with zero exceptions
  across 7,413 matches; identity-resolution and round-numbering bugs found and fixed; experimental
  2021-2026 event-level dataset (`torp`/`torpdata`) discovered, validated, and scoped as pilot-only;
  2026 umpire-stats mirror audit completed, correcting several Phase 1 "unavailable" findings.
  New docs: `docs/TARGET_VALIDATION.md`, `docs/DATA_COVERAGE.md`, `docs/2026_STATS_MIRROR.md`,
  `docs/EVENT_DATA_2021_AUDIT.md`. Updated: `DATA_DICTIONARY.md`, `PROJECT_STATE.md`,
  `docs/DATA_SOURCE_AUDIT.md` (Phase 2 addendum).
