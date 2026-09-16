# 2026 Umpire Statistics — Historical Mirror Audit (Phase 2D)

Status: **Complete for the 17 confirmed stats.** Based only on reliable reporting of what the AFL
actually announced — nothing here is inferred beyond that.
Last updated: 2026-09-16

## 1. What is confirmed public information (recap from `docs/DATA_SOURCE_AUDIT.md` §7)

From the 2026 AFL and AFLW seasons, the four field umpires are given **17 approved statistics**,
delivered via a secure Champion Data link to an AFL-issued device, after the match and before voting:
kicks, handballs, disposals, marks, contested marks, tackles, goals, behinds, goal assists, score
involvements, clearances, contested possessions, hitouts, kick-ins, intercept marks, intercept
possessions, and spoils. No weighting or presentation order has been publicly disclosed, and none is
assumed here.

## 2. Classification

Classification key: **AVAILABLE DIRECTLY** (a confirmed field, same definition, in a source we can use),
**AVAILABLE FOR LIMITED YEARS** (confirmed field, but only for part of the historical window),
**DERIVABLE** (buildable from other confirmed fields via a defensible, documented definition),
**PROXY ONLY** (a related field exists but is not confirmed to share the same definition as the Champion
Data field umpires see), **UNAVAILABLE** (no public source found).

| # | Umpire stat | Classification | Historical source & coverage | Notes |
|---|---|---|---|---|
| 1 | Kicks | AVAILABLE DIRECTLY | afltables, 1984+ | |
| 2 | Handballs | AVAILABLE DIRECTLY | afltables, 1984+ | |
| 3 | Disposals | AVAILABLE DIRECTLY | afltables, 1984+ | |
| 4 | Marks | AVAILABLE DIRECTLY | afltables, 1984+ | |
| 5 | Contested marks | AVAILABLE DIRECTLY | afltables, 1999+ | |
| 6 | Tackles | AVAILABLE DIRECTLY | afltables, 1987+ | |
| 7 | Goals | AVAILABLE DIRECTLY | afltables, 1984+ | |
| 8 | Behinds | AVAILABLE DIRECTLY | afltables, 1984+ | |
| 9 | Goal assists | AVAILABLE DIRECTLY | afltables, 2003+ | |
| 10 | Score involvements | AVAILABLE DIRECTLY | footywire (`SI`), **2015+** | Phase 1 wrongly classified this as derivable-only; Phase 2 found it as a direct field (see `docs/DATA_COVERAGE.md` §3) |
| 11 | Clearances | AVAILABLE DIRECTLY | afltables, 1998+ (total); footywire splits centre/stoppage, 2015+ | |
| 12 | Contested possessions | AVAILABLE DIRECTLY | afltables, 1999+ | |
| 13 | Hitouts | AVAILABLE DIRECTLY | afltables, 1984+ | |
| 14 | Kick-ins | **AVAILABLE FOR LIMITED YEARS** | Not on afltables/footywire at all. **Confirmed present** in the official AFL public API's `playerStats/match/{match_id}` endpoint (`extendedStats.kickins`, `kickinsPlayon`) and season/round aggregate endpoints — see §3. Coverage-year depth of this API not yet established beyond "current and recent seasons"; needs Phase 3 investigation before relying on it historically. | This corrects Phase 1's "UNAVAILABLE" verdict |
| 15 | Intercept marks | **AVAILABLE FOR LIMITED YEARS** | Same AFL public API, `extendedStats.interceptMarks`, confirmed live 2025-2026. Not found on afltables/footywire. | Corrects Phase 1's "UNAVAILABLE" verdict |
| 16 | Intercept possessions | **PROXY ONLY** (footywire) / **AVAILABLE FOR LIMITED YEARS** (AFL API) | Footywire's `ITC` ("Intercepts", 2015+) is a *related but unconfirmed-identical* metric — see caution below. The AFL public API separately exposes `intercepts` (`playersStats/seasons` / `playerSeasonRoundStats`), confirmed live for 2025 AFLM and AFLW, which is plausibly the actual Champion Data field name, though still not confirmed identical to whatever the umpires literally see. | Two independent partial routes now known instead of zero |
| 17 | Spoils | **AVAILABLE FOR LIMITED YEARS** | Not on afltables/footywire. Confirmed present in the AFL public API (`extendedStats.spoils`), live 2025-2026. | Corrects Phase 1's "UNAVAILABLE" verdict |

## 3. The AFL public API discovery (a Phase 2 correction to Phase 1's "hard constraint" framing)

Phase 1 stated flatly that Champion Data's feed — including the fields behind kick-ins, intercept
marks/possessions, and spoils — "is not accessible to this project." Phase 2 research (prompted by
investigating the 2021 event-data candidate, see `docs/EVENT_DATA_2021_AUDIT.md`) found this was **too
pessimistic**: the AFL's own public-facing website/app is backed by a JSON API
(`aflapi.afl.com.au/afl/v2/...` for some endpoints, `api.afl.com.au/cfs/afl/...` for others behind a
client-side bearer token obtained via an unauthenticated `POST` — i.e., the same access any visitor to
afl.com.au's own site effectively has, not a private commercial credential). This API's
`playerStats/match/{match_id}` and `playersStats/seasons` / `playerSeasonRoundStats` endpoints
**do expose kick-ins, intercept marks, intercepts, and spoils directly**, confirmed live against the 2025
and 2026 seasons by a third-party open-source project (`peteowen1/torp`, MIT-licensed, actively
maintained — see `docs/EVENT_DATA_2021_AUDIT.md` for full provenance).

This is a genuine, material correction to Phase 1, made because Phase 2 diligence didn't stop at "we
couldn't find it on footywire." It changes the practical 2026-readiness picture from "4 of 17 stats
completely unreproducible" to "13 of 17 fully available historically, 4 available via a *different*
official source whose historical depth we have not yet confirmed" — a meaningfully better starting
position for building a 2026-aligned feature set.

**What is still genuinely unresolved, and must not be overstated:**
- The historical depth of this AFL API's per-match/per-round stats endpoints (how many past seasons
  they actually serve, versus only recent ones) has **not** been established in this audit — the source
  project's own documentation only confirms live 2025/2026 behaviour, not a historical backfill.
- Whether this API's field values are numerically identical to what Champion Data hands the umpires
  (same definition, same rounding, same edge-case handling) is **not verified** — it is the AFL's own
  public consumer-facing mirror of Champion Data's numbers, which is a reasonable proxy but not proven
  identical.
- This does **not** change the separate, independently-confirmed finding that true event-level/chain
  (play-by-play) data is not exposed on any public endpoint — see `docs/EVENT_DATA_2021_AUDIT.md` §4.
  The 17 umpire stats are aggregate counts, and aggregate counts being available says nothing about
  event-level access.

## 4. Practical implication for modelling

- A **pre-2026 historical model** can be built using 13 of the 17 umpire-visible stats directly (via
  afltables/footywire) with well-established multi-decade or multi-year coverage (§2), and the remaining
  4 via a *newer, shallower, less-verified* source (the AFL public API) whose historical depth needs
  confirming in Phase 3 before it's relied upon for backtesting.
- Until that depth is confirmed, the safest default for any pre-2026 backtest is to build and validate
  models using only the 13 directly-available stats, then treat the 4 AFL-API-sourced stats as an
  **enrichment layer** added once their historical coverage is separately confirmed and integrated —
  not blocking the CORE/ADVANCED modelling work, but not silently assumed complete either.
- This table, not a guess about umpire weighting, is what should inform any 2026 "stats-assisted" model
  variant discussed in `docs/MODELLING_PLAN.md` §3.
