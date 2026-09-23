# 2026 Brownlow Medal — ACTUAL votes (post-event ground truth)

Extracted 2026-09-23 (UTC) from the AFL's official Brownlow live tracker after the count
concluded. This is **post-event ground truth only**: it is stored under `data/actual/`, is
never read by any feature, model or simulation code, and every frozen 2026 prediction file
was hash-verified unchanged before and after the build.

## Source

Page: `https://www.afl.com.au/brownlow-medal/live-tracker?Season=85&Round=1367`
(`#leaderboard` and `#round-by-round`).

A real headless Chromium (Playwright, no login / stealth / CAPTCHA bypass) loaded the public
page and we observed its own network requests. The tracker is driven by structured JSON, which
we use instead of visual scraping:

| Feed | Endpoint | Meaning on the page |
|---|---|---|
| ACTUAL votes per match | `GET api.afl.com.au/cfs/afl/bfawards/season/CD_S2026014` | LARGE number in each round cell |
| ACTUAL leaderboard | `GET api.afl.com.au/cfs/afl/bfawards/leaderboard/season/CD_S2026014` | Total column, eligibility, winner |
| AFL predictor | `GET aflapi.afl.com.au/afl/v2/compseasons/85/award/brownlow?players=…` | SMALL number underneath, `B` bye, did-not-play |
| Fixture | `GET aflapi.afl.com.au/afl/v2/matches?competition=1&compSeasonId=85&pageSize=250` | home / away / UTC start / venue timezone |
| Teams | `GET aflapi.afl.com.au/afl/v2/teams?compSeasonId=85` | AFL team codes |

The two `bfawards` feeds need an `x-media-mis-token` header that the page itself obtains from
`GET api.afl.com.au/cfs/afl/WMCTok` (a public per-page client token, not a credential; not
persisted). Both feeds report `status: CONCLUDED`.

The page semantics in the task brief were verified in the DOM before trusting the feeds:
each round cell holds `.stats-table__cell-button` (large = actual) and
`.stats-table__cell-predicted` (small = predictor); bye cells carry `--bye` / `B`, did-not-play
cells `--not-played`. Nick Daicos Round 1 renders large 3 / small 1, matching `bfawards` = 3 and
predictor = 1, so the two feeds genuinely differ per round.

The predictor feed is requested once per leaderboard page of 15 rows, so the leaderboard's
"Show next 15 results" button was clicked until it disappeared (12 clicks, last one labelled
"Show next 3 results"), yielding 183 DOM rows = 183 API leaderboard entries = 13 predictor pages
covering all 183 players. Per-cell DOM values were then compared against the API tables (0
mismatches).

## Outputs (`data/actual/`)

| File | Rows | Content |
|---|---|---|
| `2026_brownlow_match_votes.csv` | 621 | **Canonical source of truth.** One row per awarded vote: round, canonical `match_id`, AFL match id, date, home/away, canonical `player_id`, AFL `afl_player_id`, name, team, `actual_brownlow_votes` (3/2/1), `eligible`, `identity_status`. |
| `2026_brownlow_player_round.csv` | 4,575 | 183 players × 25 rounds (0 = Opening Round … 24): `status` ∈ played / bye / did_not_play, `actual_votes`, `afl_predicted_votes`, match ids. |
| `2026_brownlow_leaderboard.csv` | 183 | Every player on the AFL leaderboard: rank, eligible rank, totals (AFL total, reconstructed total, predictor total), 3/2/1 counts, games played / byes / did-not-play, identity status. |
| `2026_brownlow_validation.json` | — | Machine-readable result of every check below plus frozen-file hashes. |
| `raw/` | — | Provenance snapshots: the two `bfawards` responses verbatim, all 13 predictor pages, trimmed fixture and teams, the post-pagination DOM snapshot, and `fetch_status.json` (timestamps, pagination log, response log). |

Code: `src/actual/fetch_afl_tracker.py` (network capture + pagination + DOM snapshot) and
`src/actual/build_actual_votes.py` (offline build + validation from the raw snapshots).
Tests: `tests/test_2026_actual_votes.py`.

## Round normalisation and match mapping

AFL numbers the Opening Round as `roundNumber 0` and Rounds 1–24 normally, which is exactly
this project's official scheme (`src/data/round_normalization.py`, round 0 = Opening Round).
Per-round match counts are identical to the canonical season (5, 9, 7, 7, 8, 9, 9, 9, 9, 9, 9,
9, 7, 8, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9 = 207). Finals (AFL rounds 25–28) carry no votes and
are excluded.

Each AFL match was mapped to the canonical `match_id` on (round, home team, away team); the
local-venue date (UTC start converted with the venue's timezone) equals the date embedded in
every canonical id. All 207 mapped 1:1.

## Player identity

Mapped with the existing external-identity logic (`src/external/identity.py`):

1. Round-level key (round, full surname, first initial, team) — 618 of 621 rows.
2. Exact full-name match within round + team, used only where step 1 was *ambiguous*:
   Chad Warner, Round 4 (Corey Warner also played for Sydney that round) → `12797`. 1 row.
3. Season-level team-keyed fallback — not needed this year.

Name normalisation: the AFL writes "Bailey J. Williams" to separate West Coast's Bailey
Williams from the Bulldogs'; a lone middle initial is stripped before lookup only (raw AFL name
kept in the output). The AFL provider id (`CD_I…`) is kept on every row so the mapping is
auditable, and it is 1:1 with canonical `player_id` across all rows.

**Unresolved (flagged, not guessed):** Jack Ross (Richmond, `CD_I1006133`; votes in Rounds 8
and 22, 2 rows, 2 total votes). The frozen 2026 CORE build never resolved an afltables id for
him — every canonical row is a `NOID2026_*` placeholder — so there is no stable id to map to.
He is on the explicit `KNOWN_UNRESOLVED_AFL_PLAYERS` allowlist in the builder; any *new*
unresolved player fails the build.

## Validation (all hard checks pass)

| Check | Result |
|---|---|
| Every match has exactly one 3, one 2, one 1; total 6; no duplicate positions | 207 / 207 |
| Vote-getter's team is the home or away team of that match | 621 / 621 |
| Reconstructed player totals = AFL leaderboard totals | 183 / 183, sum 1,242 = 207 × 6 |
| Every vote-getter is on the leaderboard and vice versa | yes |
| Nick Daicos total | 47 (13 × 3, 3 × 2, 2 × 1), winner |
| Leaderboard pagination exhausted | button gone; DOM 183 = API 183 |
| DOM per-cell and total values = API | 0 mismatches |
| Round coverage / Opening Round = round 0 | rounds 0–24, 15 Opening Round vote rows |
| Identity 1:1 and unique on leaderboard | yes |
| Unresolved rows only on documented allowlist | 2 rows (Jack Ross) |
| Frozen prediction files unchanged (SHA-256 before/after) | 13 files unchanged |
| Cross-check vs published results (ABC News / AFL.com.au, 2026-09-21) | Daicos 47 (record, 13 × 3), Smith 36, Bontempelli 34 — all match |

### Top 10 (AFL order)

| rank | player | team | actual | AFL predictor | 3s | 2s | 1s |
|---|---|---|---|---|---|---|---|
| 1 | Nick Daicos | collingwood | 47 | 47 | 13 | 3 | 2 |
| 2 | Bailey Smith | geelong | 36 | 34 | 9 | 4 | 1 |
| 3 | Marcus Bontempelli | western_bulldogs | 34 | 32 | 8 | 4 | 2 |
| 4 | Max Gawn | melbourne | 28 | 25 | 5 | 5 | 3 |
| 5 | Will Ashcroft | brisbane_lions | 27 | 27 | 5 | 6 | 0 |
| 5 | Patrick Cripps | carlton | 27 | 19 | 5 | 5 | 2 |
| 5 | Harry Sheezel | north_melbourne | 27 | 22 | 5 | 4 | 4 |
| 8 | Zak Butters | port_adelaide | 26 | 19 | 4 | 5 | 4 |
| 8 | Isaac Heeney | sydney | 26 | 23 | 5 | 3 | 5 |
| 10 | Izak Rankine | adelaide | 25 | 23 | 6 | 3 | 1 |

### Ineligible players (reported separately; votes retained with `eligible = False`)

14 players. `rank` is the all-player rank; `eligible_rank` in the leaderboard CSV excludes them.

| player | team | actual | rank |
|---|---|---|---|
| Jason Horne-Francis | port_adelaide | 22 | 12 |
| Harley Reid | west_coast | 15 | 23 |
| Zac Bailey | brisbane_lions | 13 | 30 |
| Jacob van Rooyen | melbourne | 10 | 38 |
| Riley Thilthorpe | adelaide | 5 | 68 |
| Darcy Fogarty | adelaide | 4 | 77 |
| James Sicily | hawthorn | 3 | 87 |
| Taylor Walker | adelaide | 3 | 87 |
| Tristan Xerri | north_melbourne | 3 | 87 |
| Paul Curtis | north_melbourne | 2 | 112 |
| Dylan Moore | hawthorn | 2 | 112 |
| Darcy Byrne-Jones | port_adelaide | 1 | 140 |
| Bailey Humphrey | gold_coast | 1 | 140 |
| Ben Long | gold_coast | 1 | 140 |

### Anomalies

- None structural. No actual vote landed on a round the predictor feed marked as bye or
  did-not-play; every leaderboard player has a predictor record.
- The AFL predictor (small numbers) is **not** actual data and is stored only for reference:
  it agrees with the actual vote in 86.1% of played player-rounds among leaderboard players,
  and its season totals differ from actual by 1.83 votes on average (max under-call: Patrick
  Cripps 19 vs 27).
- Player "did-not-play" status comes from the predictor feed's explicit marker; a round with
  no predictor record at all is treated as played with no votes (this is how the AFL page
  renders it — blank cell, no `--not-played` class).

## Training integrity

- No frozen 2026 prediction / model / simulation output was modified (SHA-256 guard in the
  builder and in `test_frozen_prediction_files_not_contaminated`).
- Nothing under `src/features`, `src/models`, `src/data`, `src/simulation` reads
  `data/actual/` (grep-enforced by the same test).
- No retraining was performed. Scoring the frozen forecasts against this ground truth is a
  separate, later step.

## Re-running

```
python -m src.actual.fetch_afl_tracker      # network: refresh data/actual/raw/
python -m src.actual.build_actual_votes     # offline: rebuild CSVs + validation
python -m pytest tests/test_2026_actual_votes.py -q
```
