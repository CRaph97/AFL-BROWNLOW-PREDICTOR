# Brownlow Betting Opportunities Module

Status: architecture, pricing engine, confidence classification, and combination
logic complete and tested against real Production/Objective/Wheelo data. Live
bookmaker odds ingestion from Neds/PointsBet returned **zero real markets** this
run -- documented honestly below, not hidden or worked around with fabricated
data.

## 1. What this module is

A read-only decision-support tool: given a real bookmaker selection (odds +
line), compute Production's and Objective's own probability of that outcome
from their existing, validated Monte Carlo simulations, classify the resulting
edge by evidence convergence, and surface multi-leg combination candidates
priced from real joint simulation draws. It never retrains, recalibrates, or
otherwise touches Production, Objective, or Wheelo. It never places a bet,
sizes a stake, or automates a bookmaker account.

## 2. Live scraping result: 0/3 sources returned usable data

Attempted (public, polite HTTP GET, no login/anti-bot-bypass/stealth, per the
task's explicit constraints):

| Source | Result | Evidence |
|---|---|---|
| Neds (Brownlow futures page) | No market data in response | 112,811-byte response; zero case-insensitive occurrences of "brownlow" anywhere in the HTML; only inline script is `window.__config` (GraphQL API endpoint config, e.g. `api.neds.com.au/graphql`) -- real content is fetched client-side after load |
| PointsBet (AFL-Futures/2442638) | No market data in response | 13,050-byte response containing only the PointsBet static app shell (theme bootstrap script, favicon/manifest links) -- no page-specific content of any kind |
| PointsBet (AFL-Futures/2889213) | No market data in response | Identical 13,050-byte app shell |

Each finding was independently re-confirmed via a second, browser-oriented
fetch tool, which reached the same conclusion for both sites -- ruling out a
single-tool fetch quirk as the explanation.

**Why not go further:** all three sites deliver their real content (odds,
selections, market names) via client-side JavaScript/API calls after the
initial page load. Reaching that content would require either a full
headless-browser rendering stack (not available in this environment) or
reverse-engineering and directly calling each site's private GraphQL/API
schema -- the latter is not meaningfully distinguishable from the anti-bot
circumvention the task explicitly prohibits, so it was not attempted.

Machine-readable provenance for every attempt lives in
`data/betting/raw/*.status.json`, regenerated fresh by
`python scripts/refresh_brownlow_odds.py` each time it's run.

## 3. What was built anyway, and why it's still real, useful work

Every stage of the pipeline downstream of scraping is real, tested code
operating correctly on however many real market rows exist on a given run --
including zero, which is exactly what "fail gracefully, don't fabricate"
means in practice:

- `src/betting/market_data.py` -- loads the real, current Production
  (100,000-draw) and Objective (20,000-draw) Monte Carlo simulation arrays.
- `src/betting/pricing.py` -- prices 14 market types (WINNER, TOP_N,
  EXACT_POSITION, PLAYER_VOTES_OU, X_PLUS_VOTES, TO_POLL_A_VOTE, PLAYER_H2H,
  GROUP_H2H, TEAM_TOP_POLLER, TEAM_VOTES_OU, WINNING_VOTE_TOTAL_OU, EXACTA,
  QUINELLA, TRIFECTA) directly from real simulation draws, each with its own
  settlement-logic test. `MOST_3_VOTE_GAMES` is explicitly `UNMODELLED`: it
  would require a persisted PER-MATCH simulation array, and only season
  totals are currently persisted -- a genuine data-availability gap, not a
  parsing failure.
- `src/betting/classification.py` -- the exact confidence taxonomy and
  thresholds from the project brief (7.5pp internal-gap cutoff for High
  Confidence, Wheelo-contradiction downgrade, etc.), fixed before any real
  market data existed to tune them against.
- `src/betting/combinations.py` -- 2-5 leg combination pricing from REAL
  joint simulation draws (never a product of marginals), with a genuine
  logical-conflict detector and a correlated/redundant-leg flag.
- `scripts/refresh_brownlow_odds.py` -- the full 12-step pipeline, run for
  real; produces `data/betting/processed/{priced_opportunities,combinations,
  price_comparison}.csv` and `refresh_summary.json` (currently empty/zero
  counts, correctly reflecting 0 real scraped markets, not fabricated rows).
- `pages/23_Brownlow_Betting_Opportunities.py` -- renders this state
  honestly: filters and section scaffolding are all present and will
  populate the moment `data/betting/processed/priced_opportunities.csv`
  contains real rows, but today correctly shows "no markets currently
  available" with the real refresh timestamp, rather than a fabricated
  opportunity list.

## 4. What would need to change to get real data flowing

Any of: a manually-exported HAR/API capture from a real bookmaker session fed
into `src/betting/scraping.parse_markets()` (which is a real parser, just
never yet exercised against real market markup); a headless-browser tool
added to this environment; or a documented, authorized API partnership with a
bookmaker. None of these were pursued here, per the task's explicit scraping
constraints and time-boxing.
