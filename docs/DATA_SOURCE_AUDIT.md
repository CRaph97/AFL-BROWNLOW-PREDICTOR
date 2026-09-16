# Data Source Audit — AFL Brownlow Predictor

Status: **Phase 1 audit, with a Phase 2 addendum below correcting several findings that direct data
inspection overturned.** Phase 2 has since collected and validated real data — see
`docs/TARGET_VALIDATION.md`, `docs/DATA_COVERAGE.md`, `docs/2026_STATS_MIRROR.md`, and
`docs/EVENT_DATA_2021_AUDIT.md`. The body of this document below is left as originally written for an
honest record of what Phase 1 could and couldn't establish from research alone.
Last updated: 2026-09-16 (Phase 2 addendum added same day)

## Phase 2 addendum — corrections to this document

Phase 2's "verify before scale" principle caught several places where this Phase 1 audit was too
pessimistic, based on a single sample page rather than the full underlying data. Recorded here rather
than silently edited into the sections below, so the record shows what was wrong and why:

1. **Target variable (§1, §8):** Confirmed available, cleanly, for **1984-2025** (not a range estimate —
   an exact, evidence-based boundary). See `docs/TARGET_VALIDATION.md`.
2. **Metres gained, score involvements, tackles inside 50, centre/stoppage clearance split, turnovers**
   (marked U/P in the variable matrix in §3): all **confirmed genuinely available** from **2015** via the
   footywire-sourced `player_stats.rda` file (not scraped by us — pulled from the fitzRoy data repo).
   Metres gained in particular was wrongly marked fully unavailable. See `docs/DATA_COVERAGE.md` §3.
3. **Kick-ins, intercept marks, spoils** (§7, marked fully UNAVAILABLE): Phase 2 found these **are**
   exposed by the official AFL's own public JSON API (distinct from Champion Data's private commercial
   feed), confirmed live for 2025-2026 by a third-party open-source project. Historical depth of that API
   is not yet established. See `docs/2026_STATS_MIRROR.md` §3 for the full correction and its limits.
4. **Event-level/play-by-play data (§6):** the single-season, unlicensed, abandoned 2021 dataset
   originally found is superseded by a materially better candidate (`peteowen1/torp`/`torpdata`):
   MIT-licensed, actively maintained (updated within a day of this audit), covering **six seasons
   (2021-2026)**, with a genuine timestamped, player-attributed, coordinate-level event schema, validated
   against our own already-confirmed ground truth. This does **not** change the core conclusion that true
   event data isn't available *before* 2021, or that it must be treated as experimental — see
   `docs/EVENT_DATA_2021_AUDIT.md` for the full audit and its own documented limitations (a confirmed
   ~17% under-count on naive behind-event counting, a null-column bug in one convenience field).
5. **Round numbering (not flagged at all in Phase 1):** confirmed, concretely, that afltables' and
   footywire's round numbers can differ by one within the same season (the AFL's "Opening Round"/Round 0
   convention vs. afltables' Round 1 start). This was not anticipated in Phase 1 and is now a hard rule in
   all our join code: **never join across sources on round number, always use date.**

Nothing else in this document has been found to need correction; the licensing posture, Champion Data
inaccessibility, and general source inventory in the sections below all held up under Phase 2 scrutiny.

This document inventories every candidate data source investigated for this project, states what is actually
confirmed vs. still unverified, and flags legal/licensing constraints. Nothing here has been scraped or
downloaded. Where a claim below could not be directly verified via a fetched page during this audit, it is
labelled **[UNVERIFIED — confirm in Phase 2]**. We do not fabricate coverage or fields that could not be
confirmed.

---

## 1. Summary verdict

| Need | Status |
|---|---|
| Match results, scores, fixtures, 1897–present | **Available**, free, multiple mirrors |
| Season-total player box-score stats, 1965–present (basic), ~1998–2010+ (advanced, varies by stat) | **Available**, free |
| Match-level (player × match) basic box-score stats | **Available**, free, from ~1965 |
| Match-level advanced stats (contested poss., clearances, CM, MI5, TOG%) | **Available**, free, from roughly late 1990s/2000s depending on stat — exact start year per stat **unverified** |
| Match-level Brownlow votes (who got 3/2/1 each game) | **Likely available** from 1931 onward via AFL Tables game logs — **not yet directly verified**; season-total votes definitely available all years |
| Quarter-level *player* stats (disposals etc. by quarter) | **Not available** from any public source found. Only quarter-level *team scores* (goals/behinds) are public. |
| Event-level / play-by-play data (timestamped, chain-of-possession) | **Not available historically.** One free but limited public dataset exists for the 2021 season only. |
| The exact 17 stats umpires see from 2026 | **Partially available.** 13 of 17 map to fields on free public sources; 4 (kick-ins, intercept possessions, intercept marks, spoils) are **not confirmed available** on free sources for most of history — see §7. |
| Win probability / leverage inputs | **Not available pre-built.** Must be derived in-house from score margin + time (see Modelling Plan). No public AFL win-probability model output found. |
| Official Champion Data feed (the umpires' actual data) | **Not accessible.** Proprietary, licensed to AFL-accredited organisations only. |
| Coaches Association votes (per round, per match) | **Partially available** — publicly reported in media weekly, but no confirmed clean structured historical archive found. |
| Betting odds / market-implied probabilities | **Available**, free for ~2009–2013 depth (one source), paid APIs for full modern coverage. |

**Headline implication:** the project can be built on real historical data back to at least 1931 for the
target variable (3-2-1 votes) and to the mid/late 1990s–2000s for a genuinely rich statistical feature set. The
event-level/leverage layer of the brief (Section "GAME-STATE / CLUTCH VALUE MODEL") **cannot be built from true
play-by-play data at the granularity the brief describes**, because that data is not publicly available for
enough seasons. We will need a defensible proxy (see §6 and Modelling Plan) rather than pretending we have
Champion Data's chain data. This is the single most important scoping finding of Phase 1.

---

## 2. Source-by-source inventory

### 2.1 AFL Tables (afltables.com)

- **What it is:** A free, long-running, community-maintained statistical archive of VFL/AFL history, run
  independently of the AFL/Champion Data.
- **Coverage years:** Match results and scores 1897–present. Detailed individual player stats from 1965
  onward (per fitzRoy package documentation and site structure). Season stat pages confirmed (directly fetched
  2023 season page) to include: games, kicks, marks, handballs, disposals, disposal average, goals, behinds,
  hitouts, tackles, rebound 50s, inside 50s, clearances, clangers, frees for/against, contested possessions,
  uncontested possessions, contested marks, marks inside 50, one-percenters, bounces, goal assists, time-on-ground %,
  and a season-total Brownlow votes (BR) column.
- **Granularity:** Season totals per player (confirmed). Per-player, per-game logs exist as separate
  "Game by Game" pages per fitzRoy/community tooling and general site structure — **[UNVERIFIED — confirm exact
  URL pattern and whether a per-game Brownlow-votes column exists in Phase 2]**. Community packages (fitzRoy,
  akareen/AFL-Data-Analysis) both claim to extract match-level Brownlow votes from this site, which is
  corroborating but not first-hand confirmation.
- **Brownlow-specific pages:** A dedicated "Brownlow Records" section exists, explicitly covering detailed
  data from **1984–2025** (title of page: "Brownlow Records 1984–2025"). This suggests some detailed
  breakdowns may be richer/easier from 1984 onward, even though the underlying 3-2-1 voting *system* has been
  unchanged since 1931 (see §4). Whether full match-by-match 3-2-1 detail is obtainable for 1931–1983 needs
  direct verification.
- **Reliability:** High. This is the de facto standard reference used by essentially every public AFL analytics
  project found in this audit (fitzRoy, akareen repo, Apify scraper, etc.).
- **Access method:** HTML scraping (no API). robots.txt could not be retrieved during this audit (404) — must
  be re-checked before any automated collection in Phase 2, and scraping should be polite (low request rate,
  identify as a research bot) regardless.
- **Licensing:** No formal terms-of-use found. It is a hobby/community site. fitzRoy documentation states its
  authors scrape "with permission." We should not assume blanket permission for our own separate scraper;
  recommended path is to reuse an already-permitted aggregator (see §2.3/§2.4) rather than scrape afltables.com
  directly ourselves.
- **Missingness:** Advanced stats (contested possessions, contested marks, etc.) do not exist for older seasons
  because the AFL/Champion Data did not record them yet — this is a genuine historical absence, not a scraping
  gap.
- **Automation suitability:** Good, well-trodden by existing open-source tooling.

### 2.2 Footywire (footywire.com)

- **What it is:** Free public AFL stats and news site, historically a popular source of Champion-Data-derived
  match stats for a fantasy-football audience.
- **Coverage years:** Match-level advanced stats available for a substantial modern era; exact starting season
  **[UNVERIFIED]** — commonly cited informally as "2010 onward" by community scrapers (e.g. a GitHub scraper
  description says "from 2012 onward" for its own combined afltables+footywire tool), but this is not an
  official statement from Footywire itself and must be confirmed empirically once we pull data.
- **Granularity:** Match totals per player. Directly fetched a sample match-stats page and confirmed standard
  columns: kicks, handballs, disposals, marks, goals, behinds, tackles, hitouts, goal assists, inside 50s,
  clearances, clangers, rebound 50s, frees for/against, an "AF"-style advanced/fantasy metric, SuperCoach
  points. The page itself notes "advanced stats available" via a separate tab/view, which was **not** loaded in
  this fetch — contested possessions/uncontested possessions/contested marks/TOG% likely live there and need
  direct confirmation.
- **Important negative finding:** Spoils, intercept possessions, kick-ins, and intercept marks — four of the 17
  stats now shown to umpires — **did not appear** on the standard match-stats page fetched during this audit.
  This is a material finding for the 2026 regime-change section (§7).
- **Reliability:** Good; widely used, but has occasionally changed page structure (a known scraping pain point
  reported by community package maintainers).
- **Access method:** HTML scraping only, no public API.
- **Licensing:** robots.txt was retrieved (via WebFetch summary) and disallows a small number of paths
  associated with "fantasy" rankings/comparisons and specific bot user-agents (msnbot crawl-delay; several bots
  fully blocked). Core match-stat pages were not explicitly listed as disallowed in what we could retrieve, but
  the exact robots.txt text should be re-fetched and read verbatim (not summarised) before building any
  automated scraper, and specific disallowed paths must be respected.
- **Missingness:** Same historical-absence pattern as AFL Tables for stats not yet invented in older eras.
- **Automation suitability:** Moderate — usable, but page structure fragility means a scraper needs defensive
  parsing and monitoring, and the "advanced stats" tab's real availability needs hands-on confirmation.

### 2.3 fitzRoy (R package, CRAN + GitHub: jimmyday12/fitzRoy)

- **What it is:** A maintained, actively-updated (CRAN release found dated May 2026) open-source R package that
  provides a consistent `fetch_*` API over multiple underlying sources: AFL Tables, Footywire, the official AFL
  website, and Squiggle/"fryzigg" data.
- **Why it matters for us:** It has already solved most of the scraping fragility problem for the sources
  above, is community-maintained, and states its data is sourced "with permission" from AFL Tables and
  Footywire. Using it (or its companion pre-scraped data repo, §2.4) meaningfully de-risks our own legal and
  engineering exposure compared to writing our own scrapers from scratch.
- **Coverage:** Confirmed functions include `fetch_player_stats` (source ∈ {afl, footywire, afltables,
  fryzigg}, competition ∈ {AFLM, AFLW, VFL, VFLW, WAFL, U18B, U18G}), `fetch_fixture`, `fetch_results`,
  `fetch_ladder`, `fetch_lineup`. Exact per-source column lists and year ranges were **not** fully resolved from
  documentation alone in this audit and should be confirmed empirically by calling the package in Phase 2.
- **Reliability:** High relative to raw scraping — it is the closest thing this ecosystem has to a de facto
  standard library, used across multiple downstream public projects found in this audit.
- **Access method:** R package; would require either (a) an R subprocess/rpy2 bridge from our Python pipeline,
  (b) reimplementing equivalent request logic in Python against the same endpoints, or (c) consuming its
  companion pre-built data repository directly (§2.4).
- **Licensing:** Open source (package itself); underlying data licensing inherits from AFL Tables/Footywire/AFL
  website terms.
- **Automation suitability:** High. **Recommended primary ingestion path**, in preference to writing bespoke
  scrapers against afltables.com/footywire.com HTML ourselves.

### 2.4 fitzRoy companion data repo (github.com/jimmyday12/fitzroy_data)

- **What it is:** A GitHub repository of pre-scraped CSV/data files backing the fitzRoy package, kept up to
  date via a scheduled GitHub Actions job.
- **Why it matters:** This could let us pull already-cleaned historical data via simple file download instead
  of scraping anything ourselves — the lowest-risk ingestion option identified in this audit.
- **Status:** Located but contents/schema not yet enumerated — **[Phase 2 action: clone/inspect this repo
  directly before deciding on final ingestion architecture]**.

### 2.5 akareen/AFL-Data-Analysis (GitHub)

- **What it is:** A public GitHub repository of pre-scraped AFL data plus Python (BeautifulSoup) scraping
  scripts.
- **Claimed coverage (per repo README, not yet independently verified by us):** 15,000+ matches, 1897–2025;
  profiles for 5,700+ players; ~682,000 player-game rows; match records include per-quarter **team** goals/
  behinds (e.g. `home_q1_g`), not per-quarter player stats; player-game rows reportedly include disposals,
  marks, goals, tackles, hitouts, rebound 50s, inside 50s, clearances, clangers, frees, **Brownlow votes**,
  contested possessions, marks inside 50, bounces, goal assists; historical betting odds 2009–2024 (from
  AusSportsBetting).
- **Significance:** If the claimed inclusion of a per-game Brownlow-votes field is accurate, this repository (or
  its underlying afltables source, re-derived ourselves) directly supplies our target variable at the
  player × match level, which is the single most important data requirement in the whole project.
  **[Phase 2 action: verify this claim directly against a known historical match with a documented 3-2-1
  result before relying on it.]**
- **Licensing:** Not checked — **must read the repository's LICENSE file before use.**
- **Reliability:** Unknown/unverified provenance beyond the README description; treat as a candidate
  accelerant, not a source of truth, until cross-checked against AFL Tables directly.

### 2.6 Play-by-play / event-level data (github.com/alittlefitness/afl_play_by_play)

- **What it is:** A public dataset of AFL event-level (play-by-play) data scraped from afl.com.au.
- **Coverage:** **2021 season only**, ~800,000 rows, with 8 games missing across rounds 4–24.
- **Fields (per repo listing):** kick, handball, mark, tackle, goal, behind, disposal, turnover, clanger, and
  ~34 total event/stat types including centre clearances, stoppages, effective disposals, one-percenters,
  spoils, intercept marks, F50 marks, shots at goal, possession gains, score launches, etc.
- **Critical limitation:** No confirmed timestamps within matches, no confirmed player identifiers per event,
  and no license statement, based on what could be retrieved in this audit — **all three must be directly
  verified** before this dataset can be used for anything beyond a rough one-season proof of concept.
- **Verdict:** Useful only as a single-season pilot/validation set for prototyping a game-state/leverage
  feature framework — **not sufficient** as the backbone of the "event/time-series level" data described in the
  brief, because one season cannot support the historical model training or backtesting the project requires.
  This is a hard constraint, not a temporary gap: no broader public multi-season event-level dataset was found.

### 2.7 Champion Data (official AFL stats provider)

- **What it is:** The AFL's official and exclusive statistics provider since 1999. Operates the AFL Data
  Platform / Sports Data Warehouse, which is the actual source of the 17 stats now shown to Brownlow umpires
  (per the AFL's own 2026 announcement, delivered "through a secure link from Champion Data").
- **Access:** Confirmed **not publicly accessible** — restricted to AFL-accredited organisations, requiring a
  commercial/licensing relationship. This includes any tracking/GPS data and the authoritative event-level
  chain-of-possession data.
- **Implication:** We cannot obtain the ground-truth version of the data umpires themselves see. Every feature
  we build that maps to one of the 17 umpire-visible stats is, at best, a public-source approximation of the
  Champion Data field of the same name, not the literal number Champion Data supplies to the AFL. This should
  be stated explicitly in any model documentation and is core to the 2026 regime-change risk (§7).

### 2.8 Squiggle API (api.squiggle.com.au)

- **What it is:** A free, well-governed public API providing fixtures, results, ladder/standings, and — notably
  — the **published predictions of many other public AFL forecasting models** (a "sources" endpoint).
- **What it does NOT provide:** Player-level stats or advanced stats of any kind (explicitly stated in its own
  documentation as out of scope — those remain Champion Data's).
- **Use case for us:** Not a stats source. Its real value is as an **external benchmark comparison layer** (see
  brief's "External Benchmarks" section) — we can pull other public models' outputs for comparison without
  needing to scrape each one individually.
- **Terms of use:** Free but governed — requires a descriptive User-Agent with contact email, caching, no bulk
  simultaneous requests, no redirecting end-users through our own proxy of it. These are easy to comply with
  and should be written into any client code as hard constraints.

### 2.9 AFL Coaches Association votes

- **What it is:** After each match, the two participating clubs' opposition coaches each cast informal "coach's
  votes" for best players, which are reported in AFL media each round — this is a genuinely predictive signal
  used by at least two public Brownlow models identified in this audit (Stats Insider, WheeloRatings).
- **Availability:** Publicly reported, round by round, in mainstream AFL media. **No single clean structured
  historical archive was found** during this audit — assembling a historical panel would likely require
  scraping round-by-round news articles, which is more fragile and source-fragmented than the stats sources
  above.
- **Timing/leakage status:** Not leakage for in-season, forward-looking prediction, since it is published
  within days of the match it concerns — well before end-of-season Brownlow results. However, it is *itself* a
  human-judgement proxy for Brownlow voting rather than a raw performance statistic, and its predictive power
  partly reflects that other judges are doing some of our modelling job for us. Must be evaluated and reported
  as a clearly separate feature category (see Feature Candidates doc), and results with vs. without it should
  both be reported, exactly as the brief requires for historical-reputation variables.

### 2.10 Betting markets / odds

- **AusSportsBetting.com:** Free downloadable spreadsheet with AFL results and odds; opening/min/max/closing
  head-to-head, line, and total-score markets confirmed from 2013 onward per the site's own description found
  in search results (earlier years may have partial coverage — not confirmed).
- **The Odds API:** Paid service, historical odds from roughly mid-2020 onward on paid tiers.
- **Use case:** Pre-game market-implied win probability and expected margin as a defensible, non-fabricated
  prior for game-context features (favourite/underdog, expected competitiveness) — not a substitute for
  in-game win probability, which must be modelled separately from score/time (see Modelling Plan §Game State).

### 2.11 Historical Brownlow vote totals (season level)

- **Wikipedia** has a season-by-season Brownlum Medal article for every year, each with a final vote-tally
  table for medal contenders. This is a solid cross-check source for season totals but is **not** a source of
  match-by-match voting detail, and does not (per search results) appear to publish full round-by-round
  breakdowns for all vote-getters — only final tallies for players who received votes.
- **AFL Tables Brownlow Records section** (see §2.1) is the more promising path for match-level detail.
- **Voting system history (directly confirmed from Wikipedia's Brownlow Medal article):**
  - 1924–1930: one vote per match to a single best player (not a 3-2-1 system).
  - **1931–1975: 3-2-1 votes per match, single field umpire** — i.e., the exact ranking structure the brief
    requires already existed from 1931.
  - 1976–1977: an experimental two-umpire system where *each* umpire independently cast a 3-2-1 (two sets of
    votes per game), abandoned after two seasons.
  - **1978–present: the modern conferring system** — multiple field umpires confer and agree on a single 3-2-1
    per match. This is the cleanest, most directly comparable historical regime to what the model will predict
    going forward.
  - 1981 onward: tied vote counts at season level are now shared (multiple medallists), a season-level rule
    change with no effect on match-level modelling.
  - 1991: a rule change on how reported-and-later-cleared players are handled, relevant for a small number of
    historical edge cases in vote eligibility.
  - **2026: approved statistics shown to umpires** (this project's central structural-break event, detailed in
    §7).

### 2.12 AFL.com.au official site, ESPN, Zero Hanger, WheeloRatings, Brownlow Tracker

- Modern public-facing stats/media pages and independent prediction models. Useful as (a) possible additional
  free stat mirrors for recent seasons, and (b) **external benchmarks** for comparing our model's output once
  built (per the brief's explicit requirement, e.g. AFL.com.au's own official "Brownlow Predictor" tool,
  WheeloRatings' Monte Carlo model, Stats Insider's ordinal logistic approach, a Towards Data Science /
  Medium walkthrough, and academic-adjacent efforts like bradgreig.github.io's predictor). None of these should
  be scraped for training features about *our* target players' opponents' stats without separately confirming
  their own terms; their primary value is documented published predictions to compare against.

---

## 3. Variable-by-variable availability matrix

Legend: **A** = Available now from a public source, **D** = Derivable from available data, **P** = Partially
available / source exists but coverage or exact fields unconfirmed, **U** = Unavailable from any public source
found.

| Variable | Status | Notes |
|---|---|---|
| Games played, season totals, ladder position | A | AFL Tables / Squiggle |
| Brownlow season vote totals | A | AFL Tables / Wikipedia |
| Brownlow match-level 3-2-1 (target variable) | P | Strong circumstantial evidence it's derivable from AFL Tables game logs from 1931(from 1978 for the modern conferred system); needs direct, first-match verification in Phase 2 |
| Kicks, handballs, disposals | A | From ~1965 |
| Effective disposals, disposal efficiency | P | Not confirmed on AFL Tables/Footywire standard views; may require footywire's "advanced" tab — verify |
| Contested / uncontested possessions | P | Confirmed present in AFL Tables season totals; per-game and start-year unconfirmed |
| Clearances (total) | A | Confirmed |
| Centre clearances / stoppage clearances (split) | U/P | Not seen split on public sources checked; may exist in footywire "advanced" tab — verify |
| Tackles | A | |
| Tackles inside 50 | U | Not observed in any confirmed public field |
| Pressure acts (Champion Data proprietary metric) | U | No public source found |
| Goals, behinds | A | |
| Goal assists | A | Confirmed on both AFL Tables and Footywire |
| Score involvements | P | Some sources reference it (umpire stat list, Stats Insider methodology); not confirmed as a directly scraped field vs. a derived metric — likely **D** (derivable as goals+assists+contested marks inside 50 style composite, needs a defensible definition) |
| Inside 50s | A | |
| Metres gained | U | No public source found (Champion Data proprietary GPS/tracking-adjacent metric) |
| Intercept possessions / intercept marks | U (historical), P (recent) | Not on standard Footywire page; possibly on footywire "advanced" tab or afl.com.au modern match centre for recent seasons only — verify |
| Turnovers, clangers | A (clangers) / P (turnovers as distinct) | Clangers confirmed; "turnovers" as a distinct category needs verification vs. clangers |
| Marks, contested marks, marks inside 50 | A | |
| Ground ball gets | U | No public source found |
| One-percenters | A | |
| Hitouts, hitouts to advantage | A (hitouts) / U (to advantage) | |
| AFL Player Ratings (Champion Data's own composite) | P | Confirmed to exist historically (from 2004 per methodology paper; widely reported 2012–2019 coverage collected by fitzRoy); public accessibility/terms for bulk use unconfirmed |
| Time on ground % | A | |
| Frees for/against | A | |
| Quarter-by-quarter *player* stats | U | Not found anywhere; only team quarter scores are public |
| Event/timestamped play-by-play | U (historical) / P (2021 only) | See §2.6 |
| Match state / win probability | U (raw) / D | No public AFL win-probability feed exists; must be modelled in-house from score margin, time, and possibly market odds |
| Home/away, venue, margin, result | A | |
| Weather | P | Public historical weather APIs exist generically (e.g. Bureau of Meteorology data) but matching to venue/kickoff time is extra work — not yet investigated in depth |
| Opponent strength / ladder strength | D | Derivable from results history |
| Player position/role | P | AFL Tables lists a position field of some kind; dynamic (per-game) role vs. static listed position needs verification — likely only static position is public |
| Tagging matchups | U | No public source found |
| Substitutions | P | Modern-era substitute (medical sub) rule is publicly known and recorded from ~2011 onward; historical interchange data less certain |
| Injuries / reduced TOG | D | Reduced TOG is directly observable from the TOG% field; injury *cause* is not structured data, only inferable from TOG drops or news text |
| Coaches Association votes | P | See §2.9 |
| Betting odds / market win probability | A (2013+ solidly) | See §2.10 |
| AFL Coaches votes, media votes/ratings | P | See §2.9 |

---

## 4. The 3-2-1 structure: how far back is it valid?

Confirmed from Wikipedia's Brownlow Medal history: the current-style 3-2-1-per-match structure has existed,
with one important caveat, since **1931**. The 1976–1977 seasons used a non-comparable dual-independent-umpire
variant and should be treated as a special case or excluded from the primary training window. From **1978**
onward the voting process (multiple umpires conferring to a single 3-2-1) is structurally the same process the
model must predict, all the way to the 2026 structural break (§7).

**Recommendation:** treat 1897–1930 as out of scope entirely (wrong award structure), treat 1976–1977 as a
flagged anomaly (exclude from training or model separately if included at all), and treat **1978–2025** as the
core historical training/validation window, with 1931–1977 as a possible extension only if match-level detail
and box-score coverage from that era prove usable during Phase 2 verification — practically likely to be
box-score-poor (pre-1965 there are no detailed player stats at all per §2.1), so the realistic usable window is
probably closer to **1965/1978–2025** once data richness is accounted for, not just voting-structure validity.

---

## 5. Legal / licensing summary

- No source reviewed grants an explicit, written public license for bulk redistribution of scraped data; the
  ecosystem operates on a longstanding informal norm (community packages describing access "with permission",
  robots.txt-based restrictions rather than blanket bans, and years of tolerated hobbyist scraping).
- The lowest-risk path is to **reuse an existing, actively-maintained, already-permitted aggregator (fitzRoy /
  its data repo)** rather than write our own scraper against afltables.com or footywire.com HTML directly. This
  should be the default recommendation unless Phase 2 investigation finds a specific reason it's unsuitable.
- Champion Data's actual proprietary feed (the umpires' real data) is definitively **not accessible** without a
  commercial licensing arrangement; this project does not have one and none is proposed.
- robots.txt for afltables.com could not be retrieved in this audit (404) and must be rechecked directly before
  any automated access; footywire.com's robots.txt disallows some specific analytics paths and bot user-agents
  that must be read verbatim and respected.
- Squiggle API terms are explicit and easy to comply with (attribution UA, no bulk simultaneous hits, no
  proxying to end users).
- No data source in this audit should be treated as licensed for commercial redistribution; this project should
  be scoped as personal/research use unless a specific commercial-use question arises later, at which point
  legal advice — not this audit — is the right next step.

---

## 6. The event-level / game-state data gap (read together with Modelling Plan)

The brief asks for genuine possession-chain, timestamped event data to build Leverage Index / WPA-style
features. **This does not exist publicly at the scale needed.** The only public event-level dataset found
covers a single season (2021) with unconfirmed timestamps and player IDs. This is a hard data-availability
constraint, not a modelling choice, and materially changes what "game-state value" can mean in this project:

- We can reconstruct an approximate **time-and-margin state** at quarter-level granularity (since team
  quarter-by-quarter scores are reliably public back many decades), and combine it with match-total player
  stats to build match-level, not truly play-by-play-level, context features (e.g., "how much of the match was
  scoreboard-close", "was Q4 close", "second-half margin trajectory").
- A genuine within-quarter, event-timestamped Leverage Index / WPA (as in the brief's ideal) can only be
  approximated for the recent handful of seasons at best, and even then only using the unverified,
  single-season play-by-play dataset as a pilot, or by acquiring a Champion Data license (out of scope here).
- This should be stated plainly to the user as a scoping limitation before any game-state modelling begins, so
  expectations are set correctly: **quarter-resolution context, not true play-by-play leverage, is the
  realistic ceiling on public data.**

---

## 7. The 2026 voting-process change — what is actually publicly known

Directly confirmed from the AFL's own announcement and corroborating reporting (ABC News, ESPN, AFL.com.au):

- From the **2026 AFL and AFLW seasons**, the four field umpires are given **17 approved player-performance
  statistics** after each match, before casting Brownlow votes: kicks, handballs, disposals, marks, contested
  marks, tackles, goals, behinds, goal assists, score involvements, clearances, contested possessions,
  hitouts, kick-ins, intercept marks, intercept possessions, and spoils.
- Data is delivered via **a secure link from Champion Data**, viewed on an **AFL-issued device only**; umpires
  cannot access any other statistics, and no personal mobile devices are permitted before voting is complete.
- The AFL states voting **remains a unanimous, subjective decision** of the umpires — the stats are explicitly
  positioned as an aid, not a formula.
- The change was prompted by controversy in the **2025** Brownlow count (cited cases: Matt Rowell, Nasiah
  Wanganeen-Milera).
- We found **no public disclosure of relative weighting, order of presentation, or any algorithmic aggregation**
  of these 17 stats — umpires reportedly just see the raw numbers. We must not invent a weighting scheme and
  attribute it to the AFL.

**Modelling-relevant finding from our own audit:** of these 17 stats, **13 map cleanly to fields we can obtain
from free public sources** (kicks, handballs, disposals, marks, contested marks, tackles, goals, behinds, goal
assists, clearances, contested possessions, hitouts, and — derivable — score involvements). The remaining
**4 (kick-ins, intercept marks, intercept possessions, spoils)** were **not found on the standard free
Footywire match-stats page** checked in this audit, and no confirmed alternative free historical source was
found for them. This means:

- Our historical feature set (pre-2026 training data) can closely mirror **most** of what umpires now see, but
  not all of it, and the historical mirror-quality itself needs per-stat, per-era verification (a stat existing
  today does not mean it existed publicly in 2005).
- Going forward, if these 4 stats materially matter to 2026+ voting, our model may be structurally missing
  information that the umpires literally have in front of them — a distinct risk from the "structural break in
  the coefficients" risk the brief already raises, and one that argues for actively hunting for a proxy or
  additional source for these 4 stats in Phase 2 (e.g. checking afl.com.au's modern match centre directly,
  which was not exhaustively checked in this audit) rather than dropping them silently.
- The structural-break treatment options the brief lists (pre-2026 historical model, structural-break
  indicator, recency weighting, sensitivity analysis, Bayesian priors, post-2026 recalibration) are all still
  viable strategies and are detailed in `MODELLING_PLAN.md` §2026 Regime Change — nothing here rules any of them
  out. This section only documents the underlying facts and the specific 4-stat coverage gap discovered.

---

## 8. Open Phase-2 verification tasks (do not treat any of these as resolved)

1. Fetch a specific, known historical match (e.g. a Brownlow Medal winner's best game) from AFL Tables and
   confirm a per-game Brownlow votes field exists and matches the known public record.
2. Retrieve afltables.com's actual robots.txt (not a summarised version) and footywire.com's in full text.
3. Install/run fitzRoy against a small date range and inspect actual returned columns per source, per
   competition, to replace every "[UNVERIFIED]" tag above with a confirmed answer.
4. Clone and inspect `jimmyday12/fitzroy_data` and `akareen/AFL-Data-Analysis` contents and licenses directly
   (not just their README descriptions).
5. Check footywire's "advanced stats" tab directly for CP/UP/CM/MI5/TOG% coverage and start year, and check
   whether it or afl.com.au's modern match centre carries kick-ins/intercept possessions/intercept marks/spoils
   for any historical range.
6. Determine the actual earliest season with reliable, complete advanced-stat coverage (currently only
   loosely bounded as "late 1990s–2010s depending on stat").
5. Investigate whether a structured historical archive of AFL Coaches Association votes exists anywhere before
   committing to using this feature.
7. Confirm data volumes are within what's reasonable for polite scraping (row counts, request counts) before
   any collection begins, and draft a rate-limited, cached collection plan.

No modelling or bulk collection should start until the target-variable verification (#1) is complete — it is
the single load-bearing fact the rest of the project depends on.
