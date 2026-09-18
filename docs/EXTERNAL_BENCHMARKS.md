# External Brownlow Benchmarking Layer

Status: **Complete.** Read-only evaluation layer -- never modifies Production or Objective.
Last updated: 2026-09-18.

## 1. Sources

| Source | Status | Coverage | Granularity |
|---|---|---|---|
| **Wheelo** (local CSV) | **Integrated -- primary source** | 663/745 canonical players (98.8% of match-level rows resolved) | Match-level EV + P(3)-equivalent, all 24 rounds |
| **ESPN** (remote, snapshot) | Integrated -- partial | 28 players (ESPN's own published top-N) | Season total + own round-by-round columns (OR, R1-R24) |
| **Betfair** (remote, snapshot) | Integrated -- partial | 54 players, 22 of 207 matches | Season total only (round attribution unreliable -- see below) |
| **AFL.com.au** | **Unavailable** | none | JS-rendered navigation shell; no prediction data in static HTML |
| **Wheelo live site** | **Unavailable** (local CSV used instead) | none | Single-page app; static HTML has only a player-name dropdown, data loads via an uncaptured JS/API call |

## 2. A provenance lesson worth recording

The first pass at fetching the 4 remote sources used the WebFetch tool's built-in summarization
model, which returned a plausible-looking table for **every** source, including Wheelo's live site
and AFL.com.au. Direct `curl` fetches of the same URLs proved two of those four tables were
**hallucinated** (no such data exists anywhere in the real static HTML) and two were **real**
(ESPN's round-by-round table and Betfair's per-match tables genuinely exist, verified by finding
them in the raw HTML byte-for-byte). Every source in this document was independently confirmed
against raw HTML before being trusted -- a WebFetch summary alone was not treated as ground truth.

## 3. Identity resolution

External names are resolved to canonical `player_id` via a composite key: full hyphen-normalised
surname + first-initial + team (+ round, for match-level sources). This deliberately differs from
the internal footywire fix (`src/data/build_2026_extension.py`), which truncates a surname to its
last hyphen segment because footywire itself abbreviates one side of a hyphenated surname --
external sources here spell surnames in full, so truncating would be unnecessary and would
reintroduce the exact Byrne-Jones/Jones-style collision already fixed once internally. See
`src/external/identity.py`'s module docstring for the full reasoning.

**A real, previously-undiscovered bug was found while building this**: the Objective leaderboard
contains dozens of duplicate rows for several real players (e.g. 26 rows for "Charlie Cameron", 14
for "Jack Graham") under distinct `NOID2026_*` placeholder ids -- an existing "unresolved identity"
convention, but one that should collapse to a single placeholder-excluded player, not many synthetic
ones. Not fixed here (Objective is out of scope for this task); excluded from the canonical
identity table so it doesn't manufacture false collisions against real external players, and flagged
for separate attention.

Chad Warner vs. Corey Warner (real Sydney teammates, same surname/first-initial) is the deliberate
hard case: season-level resolution alone cannot disambiguate them, so Wheelo's match-level data is
resolved per-round instead, using Production's own already-validated match-level identity
resolution as ground truth for "who genuinely held this identity key in this round" -- correctly
resolving 21 of 26 rounds and correctly leaving the 5 rounds where both players are genuinely active
that week as ambiguous, never guessed.

## 4. Comparability rules

- `our_midpoint = mean(Production EV, Objective EV)` where both exist.
- `external_consensus_ev = mean(available EV-type external sources only)` -- Wheelo, ESPN, Betfair.
  Rank-only or editorial sources are never averaged into this.
- Where only Wheelo has data for a player, the result is explicitly labelled `"Wheelo benchmark"`,
  never implied to be a broader consensus.
- Agreement categories use fixed thresholds (±15% relative, or ±2.0 votes absolute, whichever is
  looser) chosen before any results were examined -- see `src/external/aggregate.py`.

## 5. Known limitations

- Betfair's round-heading text appears in non-monotonic order within its editorial prose (round
  mentions inside article bodies, not only as section headers) -- round-level attribution was judged
  unreliable and Betfair is therefore season-total only, excluded from the "Leader After Round" page.
- ESPN's round-column-to-official-round mapping (`OR`, `R1`-`R24`) is inferred from ESPN's own
  labelling convention, not independently cross-verified match-by-match the way Wheelo's was (ESPN's
  table doesn't expose enough per-row context, e.g. an opponent, to do that check).
- ESPN and Betfair are both partial-coverage snapshots (a subset of players/matches), not full-season
  sources -- `n_external_sources` and the `external_source_label` column make this visible per player
  rather than hiding it behind an averaged number.

## 6. Refresh

`python scripts/refresh_external_benchmarks.py` -- ingests the local Wheelo CSV, parses the latest
cached remote-source HTML snapshots in `data/external/raw/`, resolves identities, validates, and
writes `data/external/processed/*.csv` plus `source_status.json`. Does not fetch new remote HTML
itself (a separate, manual step, to avoid re-fetching on every run) and does not run on Streamlit
page load -- pages read the processed files, falling back gracefully if a file is missing.
