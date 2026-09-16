# Data Dictionary

Status: **Populated for the CORE and ADVANCED processed tables (Phase 2).** The experimental
event-level (chains) schema is documented separately in `docs/EVENT_DATA_2021_AUDIT.md` since it has not
been integrated into these tables.
Last updated: 2026-09-16

## Tables

- `data/processed/player_match_core_1984_2025.parquet` — canonical target-bearing table. Grain: one row
  per player per home-and-away match, 1984-2025. Built by `src/data/build_core_dataset.py`.
- `data/processed/player_match_advanced_2010_2025.parquet` — CORE table plus footywire-sourced advanced
  columns (null before each column's confirmed start year — see `docs/DATA_COVERAGE.md`). Built by
  `src/data/build_advanced_dataset.py`.

## CORE table columns

| Column | Type | Source | Notes |
|---|---|---|---|
| `season` | int | afltables | 1984-2025 |
| `round` | str | afltables | Numeric string for home-and-away rounds. **Not directly comparable across sources** — see Known data-quality issues below. |
| `match_id` | str | derived | `{season}_R{round}_{home_team_id}_v_{away_team_id}_{date}`. Includes `date` deliberately — a bare (season, round, teams) key is not always unique across all of AFL history (see `docs/TARGET_VALIDATION.md` §5, the 1928 replay case), so date is always part of the key even though it turned out not to be strictly required within 1984-2025. |
| `date` | str (YYYY-MM-DD) | afltables | |
| `venue` | str | afltables | |
| `player_id` | str | afltables numeric `ID`, or `NOID_<row>` placeholder | Stable across trades/seasons (verified against Lachie Neale's 2019 trade). ~79 rows (all 2025 debutants at time of writing) lack a real ID and get a **never-colliding, row-unique placeholder** — see `reports/identity_review_missing_id.csv` for the list requiring manual review before those specific rows are used in any name-based join. |
| `player_name` | str | derived (`first_name + " " + surname`) | |
| `team_id` / `opponent_id` | str | derived via `config/team_mapping.csv` | See Identity resolution below |
| `home_away` | str | afltables | |
| `team_score` / `opponent_score` / `margin` / `absolute_margin` / `win_loss_draw` | numeric/str | derived | |
| `kicks`, `marks`, `handballs`, `disposals`, `goals`, `behinds`, `hitouts`, `tackles`, `rebound_50s`, `inside_50s`, `clearances`, `clangers`, `frees_for`, `frees_against`, `contested_possessions`, `uncontested_possessions`, `contested_marks`, `marks_inside_50`, `one_percenters`, `bounces`, `goal_assists`, `time_on_ground_pct` | numeric | afltables | Null before each stat's confirmed first-available season — see `docs/DATA_COVERAGE.md` §2. Never zero-filled: a null here means "not recorded that era", which is a different fact from "recorded as zero". |
| `substitute_status` | str/null | afltables | Medical-substitute interchange flag where recorded |
| `jumper_number` | numeric | afltables | |
| `brownlow_votes` | Int64, {0,1,2,3} | afltables (via the Brownlow round-by-round page) | **The target variable.** Zero nulls within this table by construction — see `docs/TARGET_VALIDATION.md`. |

## ADVANCED table: additional columns (footywire-sourced, joined on)

| Column | Footywire code | First reliable season |
|---|---|---|
| `effective_disposals` | ED | 2010 |
| `disposal_efficiency_pct` | DE | 2010 |
| `centre_clearances` | CCL | 2015 |
| `stoppage_clearances` | SCL | 2015 |
| `score_involvements` | SI | 2015 |
| `metres_gained` | MG | 2015 |
| `turnovers` | TO | 2015 |
| `intercepts` | ITC | 2015 — **footywire's generic "Intercepts"; not confirmed identical to Champion Data's 2026 "intercept possessions"** (see `docs/2026_STATS_MIRROR.md`) |
| `tackles_inside_50` | T5 | 2015 |
| `adv_contested_possessions`, `adv_uncontested_possessions`, `adv_contested_marks`, `adv_marks_inside_50`, `adv_one_percenters`, `adv_bounces`, `adv_time_on_ground_pct`, `adv_goal_assists` | CP, UP, CM, MI5, One.Percenters, BO, TOG, GA | 2010+ | Prefixed `adv_` because a CORE-table equivalent already exists (afltables-sourced); kept both rather than silently preferring one, since they come from independent scrapes and occasionally could disagree — cross-checking them against each other is a natural QC step for Phase 3. |
| `afl_fantasy_points`, `supercoach_points` | AF, SC | 2010+ | Not in the original feature registry; retained as an available composite-quality proxy. |

**Join method** (see `src/data/build_advanced_dataset.py` for the exact code): `(season, date,
canonical team_id, canonical opponent_id, normalised player surname)`. Match rate in-scope
(2010-2025): **96.1%** after surname punctuation normalisation. See `docs/DATA_COVERAGE.md` §4 for the
known ~4% shortfall and its likely causes (interchange substitutes, a handful of root-cause-unconfirmed
recent-season cases).

## Identity resolution

- **Player identity:** the afltables numeric `ID` field, confirmed stable across team changes and
  distinct for same-surname teammates (the 1992 Daniher brothers test case — see
  `docs/TARGET_VALIDATION.md` §4c). Never fuzzy-matched: rows with a missing ID get a row-unique
  placeholder and are logged for manual review (`reports/identity_review_missing_id.csv`), never merged
  with another player on name similarity.
- **Team identity:** `config/team_mapping.csv` — an explicit, documented table mapping every historical
  and cross-source team-name spelling to one canonical `team_id`. Encodes real historical decisions
  (Fitzroy kept separate and permanently defunct rather than folded into Brisbane Lions; Footscray →
  Western Bulldogs and Kangaroos ↔ North Melbourne treated as pure renames) with a `note` column
  explaining each non-trivial case.

## Known data-quality issues (found and fixed or documented, not swept under the rug)

1. **Round numbering differs by source.** afltables numbers the season-opening round "Round 1";
   footywire (and the official AFL API used by the `torp` project) number it "Round 0"/"Opening Round",
   shifting every later round number by one in some seasons. Confirmed concretely: the 2024-05-12
   Adelaide v Brisbane Lions match is "Round 10" in afltables-derived data and "Round 9" in both the
   footywire-derived `player_stats` table and the independent `torpdata` chains dataset. **Never join
   across sources on round number — always use date.** This is why every cross-source join in this
   project's code uses `date`, not `round`.
2. **NaN player IDs collide under pandas' default duplicate detection.** `DataFrame.duplicated()`
   treats multiple `NaN` values as equal, which produced false-positive "duplicate player" flags across
   several genuinely distinct 2025-debutant players during Phase 2A/2B integrity checking. Fixed in
   `src/data/build_core_dataset.py` by assigning each missing-ID row a unique placeholder before any
   duplicate check, and covered by `tests/test_core_dataset_integrity.py::test_unique_player_match_rows`.
3. **A pre-1984 finals replay** (1928 Collingwood v Melbourne Semi Final, played twice under an
   identical Season/Round/Home/Away label eleven days apart) is out of scope for Brownlow modelling but
   is the reason `match_id` always includes `date`.
