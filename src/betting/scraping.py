"""
Neds / PointsBet Brownlow-market ingestion -- REAL browser automation.

A prior pass attempted a plain HTTP GET and correctly found both bookmakers
serve JS-rendered app shells with zero market data in the raw HTML. This
version uses real Playwright browser automation (a genuine headless Chromium
instance, no login/CAPTCHA-bypass/stealth) to load each page as an ordinary
user's browser would, and observes the page's own ordinary public network
requests (Playwright's `page.on("response")`) to capture the real market
data those pages fetch client-side. This is NOT reverse-engineering a
private/authenticated API: it is the same JSON the bookmaker's own official
web client requests and renders, observed rather than guessed at.

Confirmed real endpoints (see data/betting/raw/*.status.json for the
machine-readable, timestamped record generated fresh each run):
- Neds: `GET https://api.neds.com.au/v2/sport/EventCard?id=<event_id>` --
  returns {"markets": {...}, "entrants": {...}, "prices": {...}} keyed by
  UUID. `prices` keys are "<entrant_id>:<market_id>:" -> {"odds": {
  "numerator": N, "denominator": D}} (fractional odds; decimal = 1 + N/D).
  Market lines (e.g. a player-votes O/U line) live on the market object's
  "handicap" field, not the entrant.
- PointsBet: `GET https://api.au.pointsbet.com/api/mes/v3/events/<event_id>`
  -- returns {"fixedOddsMarkets": [...]}, each market a dict with "name"
  (the real market name) and "outcomes": [{"name", "price", "points"}, ...].
  For Over/Under markets the line lives on the MARKET's own "points" field
  (each outcome's own "points" is 0 and unhelpful); the outcome "name" is
  "Over <line>" / "Under <line>".

Both were independently verified by loading the real pages and inspecting
the captured JSON directly (not assumed from documentation or a summariser
tool) -- this project has a documented near-miss where a WebFetch summary
once returned a plausible-looking but fabricated table for two sources; the
lesson applied here is to trust only a direct, locally-parsed JSON response
body, never a natural-language summary of one.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "betting" / "raw"
CONFIG_DIR = ROOT / "config"

SOURCES = {
    "neds": "https://www.neds.com.au/sports/australian-rules/brownlow/afl-brownlow-2026/6f3764b6-46d4-4dfe-8124-1804ea7af74f",
    "pointsbet_1": "https://pointsbet.com.au/sports/aussie-rules/AFL-Futures/2442638",
    "pointsbet_2": "https://pointsbet.com.au/sports/aussie-rules/AFL-Futures/2889213",
}

NEDS_EVENT_ID = "6f3764b6-46d4-4dfe-8124-1804ea7af74f"
POINTSBET_EVENT_IDS = {"pointsbet_1": "2442638", "pointsbet_2": "2889213"}

PARSER_VERSION = "2.0.0-real-browser-network-capture"


@dataclass
class SourceStatus:
    source: str
    url: str
    retrieved_at: str
    parser_version: str
    http_status: int | None
    bytes_downloaded: int
    contains_market_data: bool
    status: str  # "OK" | "NO_MARKET_DATA_IN_RESPONSE" | "FETCH_FAILED"
    n_markets_found: int
    n_selections_found: int
    detail: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_neds() -> tuple[dict | None, SourceStatus]:
    from playwright.sync_api import sync_playwright

    url = SOURCES["neds"]
    captured: dict = {}
    html_len = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def on_response(resp):
                if "/v2/sport/EventCard" in resp.url and NEDS_EVENT_ID in resp.url:
                    try:
                        body = resp.json()
                        if isinstance(body, dict) and "markets" in body:
                            captured["body"] = body
                    except Exception:
                        pass

            page.on("response", on_response)
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
            except Exception:
                page.wait_for_timeout(6000)
            page.wait_for_timeout(2000)
            html_len = len(page.content())
            browser.close()
    except Exception as exc:
        return None, SourceStatus(
            source="neds", url=url, retrieved_at=_now(), parser_version=PARSER_VERSION,
            http_status=None, bytes_downloaded=0, contains_market_data=False,
            status="FETCH_FAILED", n_markets_found=0, n_selections_found=0,
            detail=f"Playwright exception: {exc}",
        )

    body = captured.get("body")
    if not body:
        return None, SourceStatus(
            source="neds", url=url, retrieved_at=_now(), parser_version=PARSER_VERSION,
            http_status=200, bytes_downloaded=html_len, contains_market_data=False,
            status="NO_MARKET_DATA_IN_RESPONSE", n_markets_found=0, n_selections_found=0,
            detail="page loaded but the EventCard network response was not observed/parseable",
        )
    n_markets = len(body.get("markets", {}))
    n_selections = len(body.get("entrants", {}))
    status = SourceStatus(
        source="neds", url=url, retrieved_at=_now(), parser_version=PARSER_VERSION,
        http_status=200, bytes_downloaded=len(json.dumps(body)), contains_market_data=True,
        status="OK", n_markets_found=n_markets, n_selections_found=n_selections,
        detail="real market data captured from the page's own EventCard network response",
    )
    return body, status


def _fetch_pointsbet(source: str) -> tuple[dict | None, SourceStatus]:
    from playwright.sync_api import sync_playwright

    url = SOURCES[source]
    event_id = POINTSBET_EVENT_IDS[source]
    captured: dict = {}
    html_len = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def on_response(resp):
                if f"/api/mes/v3/events/{event_id}" in resp.url:
                    try:
                        body = resp.json()
                        if isinstance(body, dict) and "fixedOddsMarkets" in body:
                            captured["body"] = body
                    except Exception:
                        pass

            page.on("response", on_response)
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
            except Exception:
                page.wait_for_timeout(6000)
            page.wait_for_timeout(2000)
            html_len = len(page.content())
            browser.close()
    except Exception as exc:
        return None, SourceStatus(
            source=source, url=url, retrieved_at=_now(), parser_version=PARSER_VERSION,
            http_status=None, bytes_downloaded=0, contains_market_data=False,
            status="FETCH_FAILED", n_markets_found=0, n_selections_found=0,
            detail=f"Playwright exception: {exc}",
        )

    body = captured.get("body")
    if not body:
        return None, SourceStatus(
            source=source, url=url, retrieved_at=_now(), parser_version=PARSER_VERSION,
            http_status=200, bytes_downloaded=html_len, contains_market_data=False,
            status="NO_MARKET_DATA_IN_RESPONSE", n_markets_found=0, n_selections_found=0,
            detail="page loaded but the events API network response was not observed/parseable",
        )
    markets = body.get("fixedOddsMarkets", [])
    n_selections = sum(len(m.get("outcomes", [])) for m in markets)
    status = SourceStatus(
        source=source, url=url, retrieved_at=_now(), parser_version=PARSER_VERSION,
        http_status=200, bytes_downloaded=len(json.dumps(body)), contains_market_data=True,
        status="OK", n_markets_found=len(markets), n_selections_found=n_selections,
        detail="real market data captured from the page's own events-API network response",
    )
    return body, status


def refresh_raw_snapshots() -> list[SourceStatus]:
    """Steps 1-3: load each bookmaker page with a real headless browser,
    capture its own genuine network response containing market data, and
    save both the raw JSON and a machine-readable status record. A
    per-source failure (browser crash, page structure change, no response
    observed) is recorded, never fatal to the run and never papered over
    with invented data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []

    body, status = _fetch_neds()
    if body is not None:
        (RAW_DIR / "neds.json").write_text(json.dumps(body))
    (RAW_DIR / "neds.status.json").write_text(json.dumps(asdict(status), indent=2))
    statuses.append(status)

    for source in ("pointsbet_1", "pointsbet_2"):
        body, status = _fetch_pointsbet(source)
        if body is not None:
            (RAW_DIR / f"{source}.json").write_text(json.dumps(body))
        (RAW_DIR / f"{source}.status.json").write_text(json.dumps(asdict(status), indent=2))
        statuses.append(status)

    return statuses


# --------------------------------------------------------------------------
# Market-name -> canonical market_type mapping. Anything not recognised
# below is tagged "UNMODELLED" (matching src.betting.pricing.UNMODELLED)
# rather than guessed. Exotic multi-leg markets (Exacta/Quinella/Trifecta/
# First4/Boxed variants) are deliberately NOT parsed into individual legs
# this pass -- a correct parser for those would need careful, separately-
# verified leg-extraction logic, and shipping a rushed one risks silently
# wrong legs, which is worse than an honest UNMODELLED tag. They are still
# captured as raw rows (market_name preserved) so their existence is visible
# in the audit trail, just not priced.
# --------------------------------------------------------------------------
_ODDS_ODD = re.compile(r"^\s*(Over|Under)\s+([\d.]+)\s*$", re.IGNORECASE)


def _decimal_from_fraction(numerator: float, denominator: float) -> float | None:
    if denominator in (None, 0):
        return None
    return 1.0 + float(numerator) / float(denominator)


def _classify_market(market_name: str) -> tuple[str, dict]:
    """Returns (market_type, extra) where extra carries market-level
    parameters (n, position, threshold) parsed from the name. market_type is
    "UNMODELLED" for anything not confidently recognised."""
    name = market_name.strip()

    if re.fullmatch(r"(AFL )?Brownlow( Medal)?( 2026)?( Winner)?", name, re.IGNORECASE) or name == "AFL Brownlow 2026":
        return "WINNER", {}
    if re.match(r"^Brownlow Medal Winner", name, re.IGNORECASE):
        return "WINNER", {}

    m = re.match(r"^Top\s*(\d+)\b", name, re.IGNORECASE)
    if m:
        return "TOP_N", {"n": int(m.group(1))}

    if re.search(r"head\s*to\s*head|H(ead)?\s*2\s*H(ead)?|H2H", name, re.IGNORECASE):
        return "PLAYER_H2H", {}

    if "player to come fourth" in name.lower():
        return "EXACT_POSITION", {"position": 4}

    m = re.search(r"to poll\s*(\d+)\s*or more", name, re.IGNORECASE)
    if m:
        return "X_PLUS_VOTES", {"threshold": int(m.group(1))}

    if re.fullmatch(r"to poll a vote", name, re.IGNORECASE):
        return "TO_POLL_A_VOTE", {}

    if "team" in name.lower() and "vote" in name.lower():
        return "TEAM_VOTES_OU", {}

    if name.upper().startswith("O/U ") and "team" not in name.lower():
        return "PLAYER_VOTES_OU", {}

    return "UNMODELLED", {}


def _extract_ou_subject(market_name: str, market_type: str) -> tuple[str | None, str | None]:
    """For PLAYER_VOTES_OU / TEAM_VOTES_OU markets whose subject (the player
    or team the line applies to) is embedded in the market name rather than
    the entrant/outcome name (true for every Neds O/U market, and for
    PointsBet's team-total markets), extract it. Returns (player_name,
    team_name_string) -- exactly one is non-None."""
    if market_type == "PLAYER_VOTES_OU":
        m = re.match(r"^O/U\s+(.*)$", market_name.strip(), re.IGNORECASE)
        return (m.group(1).strip() if m else None), None
    if market_type == "TEAM_VOTES_OU":
        # e.g. "O/U Adelaide Crows Team Total Votes" (Neds) or
        # "Adelaide Crows Team Votes O/U (Brownlow Medal 2026)" (PointsBet).
        cleaned = re.sub(r"^O/U\s+", "", market_name.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*Team\s*(Total\s*)?Votes\s*O/?U.*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(.*?\)\s*$", "", cleaned).strip()
        return None, cleaned or None
    return None, None


def _load_team_mapping() -> dict[str, str]:
    mapping = {}
    if (CONFIG_DIR / "team_mapping.csv").exists():
        tm = pd.read_csv(CONFIG_DIR / "team_mapping.csv")
        for _, r in tm.iterrows():
            mapping[str(r["source_name"]).strip().lower()] = r["canonical_team_id"]
    return mapping


_TEAM_MAP = None


def _team_from_string(s: str) -> str | None:
    global _TEAM_MAP
    if _TEAM_MAP is None:
        _TEAM_MAP = _load_team_mapping()
    if not s:
        return None
    key = s.strip().lower()
    if key in _TEAM_MAP:
        return _TEAM_MAP[key]
    # Try longest-match containment (team-total-votes market names embed the
    # team name inside a longer string, e.g. "Adelaide Crows Team Votes O/U").
    best = None
    for src, tid in _TEAM_MAP.items():
        if src in key and (best is None or len(src) > len(best[0])):
            best = (src, tid)
    return best[1] if best else None


_NEDS_ENTRANT_NAME = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


def _split_neds_entrant(name: str) -> tuple[str, str | None]:
    """Neds entrant names are 'Player Name (Team Name)' for player-level
    markets. Team-only / Over/Under entrants have no parenthetical."""
    m = _NEDS_ENTRANT_NAME.match(name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return name.strip(), None


def parse_neds(body: dict) -> pd.DataFrame:
    rows = []
    markets = body.get("markets", {})
    entrants = body.get("entrants", {})
    prices = body.get("prices", {})

    for market in markets.values():
        if not market.get("visible", True):
            continue
        market_name = market["name"]
        market_type, extra = _classify_market(market_name)
        line = market.get("handicap")
        entrant_ids = market.get("entrant_ids", [])

        for eid in entrant_ids:
            entrant = entrants.get(eid)
            if entrant is None or not entrant.get("visible", True):
                continue
            price_key = next((k for k in prices if k.startswith(eid + ":")), None)
            odds_obj = prices.get(price_key, {}).get("odds") if price_key else None
            odds = _decimal_from_fraction(odds_obj["numerator"], odds_obj["denominator"]) if odds_obj else None

            raw_name = entrant["name"]
            player_name, team = _split_neds_entrant(raw_name)
            side = None
            if player_name.lower() in ("over", "under"):
                side = player_name.lower()
                player_name = None  # this row's "selection" is the O/U side, not a player
                subj_player, subj_team = _extract_ou_subject(market_name, market_type)
                player_name, team = subj_player, subj_team or team

            rows.append({
                "source": "neds", "market_type": market_type, "market_name": market_name,
                "selection": raw_name, "player_name": player_name, "team": team,
                "odds": odds, "line": line, "side": side,
                "n": extra.get("n"), "position": extra.get("position"), "threshold": extra.get("threshold"),
                "selection_id": f"neds:{market['id']}:{eid}",
                "settlement_comments": market.get("comments"),
            })
    return pd.DataFrame(rows)


def parse_pointsbet(body: dict, source: str) -> pd.DataFrame:
    rows = []
    for market in body.get("fixedOddsMarkets", []):
        if not market.get("isOpenForBetting", True):
            continue
        market_name = market.get("name") or ""
        market_type, extra = _classify_market(market_name)
        line = market.get("points")

        for outcome in market.get("outcomes", []):
            raw_name = outcome.get("name", "")
            odds = outcome.get("price")
            side = None
            player_name = raw_name
            team = _team_from_string(market_name) if market_type == "TEAM_VOTES_OU" else None

            ou = _ODDS_ODD.match(raw_name)
            if ou:
                side = ou.group(1).lower()
                if line is None:
                    try:
                        line = float(ou.group(2))
                    except ValueError:
                        pass
                player_name = None
                subj_player, subj_team = _extract_ou_subject(market_name, market_type)
                player_name, team = subj_player, subj_team or team

            rows.append({
                "source": source, "market_type": market_type, "market_name": market_name,
                "selection": raw_name, "player_name": player_name, "team": team,
                "odds": odds, "line": line, "side": side,
                "n": extra.get("n"), "position": extra.get("position"), "threshold": extra.get("threshold"),
                "selection_id": f"{source}:{market.get('key')}:{outcome.get('key')}",
                "settlement_comments": None,
            })
    return pd.DataFrame(rows)


def parse_all_from_snapshots() -> pd.DataFrame:
    """Reads whatever raw JSON snapshots are currently on disk (written by
    refresh_raw_snapshots()) and returns the combined normalised market-row
    table. Sources with no snapshot / no market data contribute zero rows,
    never fabricated ones."""
    columns = ["source", "market_type", "market_name", "selection", "player_name", "team",
               "odds", "line", "side", "n", "position", "threshold", "selection_id",
               "settlement_comments"]
    frames = []
    neds_path = RAW_DIR / "neds.json"
    if neds_path.exists():
        try:
            frames.append(parse_neds(json.loads(neds_path.read_text())))
        except Exception:
            pass
    for source in ("pointsbet_1", "pointsbet_2"):
        p = RAW_DIR / f"{source}.json"
        if p.exists():
            try:
                frames.append(parse_pointsbet(json.loads(p.read_text()), source))
            except Exception:
                pass
    if frames:
        out = pd.concat(frames, ignore_index=True)
        for c in columns:
            if c not in out.columns:
                out[c] = pd.Series(dtype="object")
        return out[columns]
    return pd.DataFrame(columns=columns)


def load_cached_statuses() -> list[dict]:
    out = []
    for source in SOURCES:
        path = RAW_DIR / f"{source}.status.json"
        if path.exists():
            out.append(json.loads(path.read_text()))
    return out
