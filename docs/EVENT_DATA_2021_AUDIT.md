# Event-Level (Play-by-Play) Data Audit — Phase 2E

Status: **Complete. Verdict: a genuinely usable candidate found and validated — see §5.**
This module is **EXPERIMENTAL**. Nothing here has been integrated into the primary (CORE/ADVANCED)
dataset, and no finding here should be treated as established beyond the specific seasons checked.
Last updated: 2026-09-16

## 1. Two candidates investigated

Phase 1 identified one candidate (`alittlefitness/afl_play_by_play`). Phase 2 investigation of that
candidate's own source code led to a second, substantially better-maintained candidate
(`peteowen1/torp` / `peteowen1/torpdata`). Both are documented below; **the second is recommended**.

## 2. Candidate A: `alittlefitness/afl_play_by_play`

- **Creator/source:** GitHub user `alittlefitness`; data "sourced from afl.com.au for the 2021 AFL
  season" per its own README.
- **Licence/status:** **No LICENSE file.** Repository created 2021-05-14, last *content* push
  2022-10-03 — effectively abandoned, not actively maintained (16 stars).
- **Coverage:** **2021 season only.** 8 games explicitly documented as missing (spanning rounds 4-24).
- **Schema:** Two CSV files (~53MB, ~54MB), reportedly 800,000+ rows, 34 listed event/stat types (Kick,
  Handball, Mark, Tackle, Goal, Behind, Disposal, Turnover, Clanger, Centre Clearance, Stoppage
  Clearance, Effective Disposal, Spoil, Intercept Mark, Ground Ball Get, Score Launch, etc.).
- **Player attribution / timestamps / quarter / scoreboard reconstruction:** **Not confirmed** — the
  repository's own documentation does not state whether rows carry player IDs or in-match timestamps,
  and inspecting a live sample was not pursued once Candidate B was found to be a strictly better,
  actively-maintained superset covering the same underlying data source.
- **Verdict on this candidate alone: NOT RECOMMENDED** — no license, unmaintained, single season, and
  its own methodology script (`Score Sources.R`, found inside the repo) turned out to reference a
  *different*, since-renamed/relocated data pipeline — which is what led to Candidate B.

## 3. Candidate B: `peteowen1/torp` (R package) + `peteowen1/torpdata` (data releases) — RECOMMENDED

### Identity and provenance
- **Creator:** GitHub user `peteowen1`. `torp` is a proper R package (DESCRIPTION, NAMESPACE, tests,
  pkgdown site, CI badge for automated package checks and test coverage). `torpdata` is a companion
  repository distributing processed data as **Parquet files via GitHub Releases**.
- **Licence: MIT**, explicit `LICENSE`/`LICENSE.md` files on both repositories — a materially stronger,
  unambiguous licensing position than every other source examined in this project (afltables/footywire's
  informal norms included).
- **Maintenance:** Created July 2023; **last data release push 2026-09-15 — one day before this audit**,
  i.e. actively maintained and currently tracking the in-progress 2026 season.
- **Data source:** The AFL's own public/semi-public JSON API (`aflapi.afl.com.au`, `api.afl.com.au/cfs`,
  `api.afl.com.au/sapi`) — accessed via a documented client-side token flow, not a private Champion Data
  commercial credential. The project's own 878-line `AFL-API-REFERENCE.md` documents every endpoint,
  field, and caveat in detail, including explicit self-critical notes about what the API does and does
  **not** expose (see §4). This level of documentation and self-scrutiny was treated as a strong
  reliability signal in this audit.

### Schema (from the `matchPlays/{match_id}` chain endpoint, distributed as `chains-data` / `pbp-data`)
One row per **action** (event) within a possession **chain**, with:
- `period` (quarter), `period_seconds` (seconds elapsed in that quarter) — **a genuine in-match clock**.
- `player_id`, team identifiers (`team_id` at both chain and action level).
- `description` (event type, e.g. "Kick", "Handball", "Goal", "Behind", "Centre Bounce", "Loose Ball
  Get"), `disposal` (effective/ineffective), `shot_at_goal` flag.
- `x`, `y` on-ground coordinates.
- `initial_state` / `final_state` per chain (e.g. `centreBounce` → `turnover`, `possGain` → `goal`).
- Match-level context attached to every row: venue, both teams' running/final scores, round, date.

### Number of matches / events (2024 season, downloaded and inspected directly)
426,754 action rows in the 2024 `chains_data_2024_all.parquet` file alone. Six full seasons available
as separate files: **2021, 2022, 2023, 2024, 2025, 2026 (in progress)** — a genuinely multi-season,
current dataset, not a single-season snapshot.

## 4. Validation performed (against our own already-verified ground truth)

The 2024-05-12 Adelaide v Brisbane Lions draw (90-90) — the same match independently verified against
live AFL Tables pages in `docs/TARGET_VALIDATION.md` §4a — was located in this dataset and checked
event-by-event for Izak Rankine (the match's 3-vote getter):

| Check | Result |
|---|---|
| Match found, correct final score (90-90) | **Yes**, `match_id = CD_M20240140909` |
| Handballs (afltables: 11) | **11** `Handball` events — exact match |
| Kicks (afltables: 13) | 12 `Kick` events + 1 `Ground Kick` event = **13** — exact match, but only once
  "Ground Kick" is correctly understood as a kick sub-type. This is a real, useful, previously-undocumented
  mapping detail for anyone reconstructing box-score totals from this event feed. |
| Goals (afltables: 3) | **3** `Goal` events — exact match |
| Behinds, personal (afltables: 1) | **1** `Behind` event attributed to Rankine — exact match |
| Match total goals (known: 13+13=26) | **26** `Goal` events — exact match |
| Match total behinds (known: 12+12=24) | **20** `Behind`-described events — **short by 4 (~17%)** |

**Round-number mismatch, a third independent confirmation:** this dataset labels the match
`round_number = 9`, matching footywire's numbering (see `docs/DATA_DICTIONARY.md`), not afltables'
"Round 10" — the same Opening-Round-as-Round-0 offset found earlier, now confirmed in a third,
independently-sourced dataset. This should be treated as settled: **never join across these sources on
round number**; always use date.

**Scoreboard reconstruction:** goals reconstruct perfectly by counting `description == "Goal"` events
per team (via the chain-level `team_id`, which correctly and consistently identified the scoring team
using the fixture's `CD_T10`/`CD_T20`-style codes cross-checked against `home_team_team_abbr` /
`away_team_team_abbr`). **Behinds do not** — a naive count of `description == "Behind"` events
undercounts true behinds by roughly one-sixth in the one match checked, almost certainly because rushed
behinds (`final_state` of `rushed`/`rushedOpp`) are encoded as a chain outcome rather than always
producing their own dedicated `"Behind"` action row. **Any scoreboard-reconstruction code built on this
data must additionally count `rushed`/`rushedOpp` final-states, not just `Behind`-described rows** — this
is now a documented, specific requirement, not a guess.

**A confirmed data-quality bug in the joined convenience columns:** the derived `team_team_abbr` /
`team_team_name` columns were **100% null** throughout the 2024 file, even though the underlying raw
`team_id` codes (e.g. `CD_T10`) were present and correct (92.7% non-null). Any use of this dataset
**must use the raw `team_id` joined against the fixture-level `home_team_id`/`away_team_id`**, not the
apparently-broken derived team-name columns, at least for the 2024 file checked.

## 5. Verdict

**VALID ENOUGH FOR EXPERIMENTAL PILOT — Candidate B (`torp`/`torpdata`) only.** Candidate A is
superseded and not recommended for any use.

This verdict is deliberately scoped:
- Valid for building and testing a **Tier-2 experimental game-state/leverage model** (per
  `docs/MODELLING_PLAN.md` §5) across **2021-2026** — six seasons, not one, a meaningfully stronger
  basis for the "does intra-match timing information help?" research question than Phase 1 anticipated.
- **Not** valid as a substitute for the CORE/ADVANCED box-score tables, and **not** a source of
  historical depth before 2021 — it does not extend the project's usable history, only adds a richer
  *layer* on top of the recent portion of it.
- **Not** to be treated as perfectly accurate: the confirmed ~17% behind-undercount (if naively counted)
  and the null `team_team_abbr` bug are real, specific defects that any pilot code must handle explicitly,
  not silently work around.
- Findings from this single validated match must not be generalised as "the dataset is accurate" without
  further spot-checks across seasons — this audit checked one match in detail plus aggregate schema
  inspection; a broader multi-match validation pass is recommended before this data drives any reported
  result, experimental or otherwise.

## 6. Recommendation for Phase 3

If the experimental track proceeds: ingest `chains-data` (2021-2026) from `torpdata`'s GitHub Releases
(clean Parquet, no scraping required, MIT-licensed), build the Tier-2 leverage/WPA features on top of it
per `docs/MODELLING_PLAN.md` §5, and explicitly compare a game-state-enhanced model against the
equivalent CORE/ADVANCED-only model on the 2021-2025 overlap window to answer the brief's stated
research question: *does intra-match timing/game-state information materially improve Brownlow vote
prediction beyond normal match-level statistics?* Report the answer honestly whichever way it comes out,
per `docs/MODELLING_PLAN.md`'s explicit non-goal of assuming the answer in advance.
