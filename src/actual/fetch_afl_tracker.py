"""
2026 Brownlow ACTUAL votes -- extraction from the AFL's official live tracker.

Page: https://www.afl.com.au/brownlow-medal/live-tracker?Season=85&Round=1367
(#leaderboard / #round-by-round tabs).

Same approach as src/betting/scraping.py: a real headless Chromium (Playwright,
no login / stealth / CAPTCHA bypass) loads the public page exactly as a
browser would, and we observe the page's own ordinary network requests. The
tracker turned out to be driven by structured JSON, which we prefer over
visual scraping. Confirmed endpoints (observed 2026-09-23, page v5.53.20):

  ACTUAL Brownlow votes (the LARGE number in each round cell):
    GET https://api.afl.com.au/cfs/afl/bfawards/season/CD_S2026014
        -> {"seasonId","status":"CONCLUDED","matchVotes":[{"matchId",
            "roundNumber","votes":[{"player":{playerId,givenName,surname,...},
            "team":{teamId,teamAbbr,teamName},"votes":3|2|1,"eligible"}]}]}
    GET https://api.afl.com.au/cfs/afl/bfawards/leaderboard/season/CD_S2026014
        -> {"leaderboard":[{"player","team","eligible","winner","leader",
            "roundByRoundVotes":[{matchId,roundNumber,votes}],
            "roundByRoundTotalVotes":[...],"totalVotes"}]}
    Both require the header `x-media-mis-token`, whose value the page itself
    obtains from GET https://api.afl.com.au/cfs/afl/WMCTok (a public,
    per-page client token -- not a user credential; not persisted here).

  AFL PREDICTOR votes (the SMALL number underneath -- NOT actual votes):
    GET https://aflapi.afl.com.au/afl/v2/compseasons/85/award/brownlow
        ?page=0&pageSize=20&players=<ids of the 15 rows just rendered>
        -> {"players":[{"id","providerId","firstName","surname","eligible",
            "teamId","rounds":{"<round>":[{"providerId":matchId,"points"}
            | {"played":false,"bye":true} | {"providerId","played":false}]},
            "totalVotes"}]}
    The page calls this once per leaderboard page of 15 rows, so clicking
    "Show next 15 results" until exhaustion is also what yields predictor
    + bye / did-not-play markers for every player.

  Fixture / identity support (no token needed):
    GET https://aflapi.afl.com.au/afl/v2/matches?competition=1&compSeasonId=85&pageSize=250
    GET https://aflapi.afl.com.au/afl/v2/teams?compSeasonId=85&pageSize=100

The DOM cross-check verified the semantics the task specified: in the
leaderboard table each round cell holds `.stats-table__cell-button` (large,
actual) and `.stats-table__cell-predicted` (small, predictor); a bye cell has
class `--bye` with text "B", a did-not-play cell has class `--not-played`.
E.g. Nick Daicos Round 1 renders large "3" / small "1", matching bfawards=3
and predictor=1 -- the two feeds genuinely differ per round, so the split is
not cosmetic.

Run:  python -m src.actual.fetch_afl_tracker
Writes data/actual/raw/*.json (provenance snapshots) only. Building the CSVs
is src/actual/build_actual_votes.py, which works purely from these snapshots
so it is re-runnable offline and unit-testable.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "actual" / "raw"

SEASON_CD = "CD_S2026014"
COMP_SEASON_ID = 85
TRACKER_URL = (
    f"https://www.afl.com.au/brownlow-medal/live-tracker?Season={COMP_SEASON_ID}&Round=1367#leaderboard"
)
FETCHER_VERSION = "1.0.0-afl-tracker-network-capture"

LOAD_MORE_SELECTOR = "button.stats-table-load-more-button"
ROW_SELECTOR = "tr.stats-table__body-row"
MAX_LOAD_MORE_CLICKS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify(url: str) -> str | None:
    """Which provenance bucket a response URL belongs to (None = ignore)."""
    if f"/cfs/afl/bfawards/leaderboard/season/{SEASON_CD}" in url:
        return "bfawards_leaderboard"
    if f"/cfs/afl/bfawards/season/{SEASON_CD}" in url:
        return "bfawards_season"
    if f"/afl/v2/compseasons/{COMP_SEASON_ID}/award/brownlow" in url:
        return "predictor_page"
    if f"/afl/v2/matches?competition=1&compSeasonId={COMP_SEASON_ID}&pageSize=250" in url:
        return "matches"
    if f"/afl/v2/teams?compSeasonId={COMP_SEASON_ID}" in url:
        return "teams"
    return None


DOM_SNAPSHOT_JS = """
() => {
  const table = document.querySelector('table');
  const headers = Array.from(document.querySelectorAll('th.stats-table__header-cell'))
    .map(th => ({cls: th.className, text: th.innerText.trim()}));
  const rows = Array.from(document.querySelectorAll('tr.stats-table__body-row')).map(tr => {
    const fav = tr.querySelector('[data-favourite-provider-id]');
    const nameParts = Array.from(tr.querySelectorAll('.stats-table__row-brownlow-player-name p')).map(p => p.innerText.trim());
    const badge = tr.querySelector('.stats-table__player-cell-brownlow-badge');
    const cells = Array.from(tr.querySelectorAll('td.stats-table__cell')).map(td => {
      const inner = td.querySelector('.stats-table__cell-inner-wrapper');
      const btn = td.querySelector('.stats-table__cell-button');
      const pred = td.querySelector('.stats-table__cell-predicted');
      return {
        actual: btn ? btn.innerText.trim() : (inner ? inner.innerText.trim() : ''),
        title: btn ? btn.getAttribute('title') : null,
        bye: !!(inner && inner.className.includes('--bye')),
        not_played: !!(inner && inner.className.includes('--not-played')),
        predicted: pred ? pred.innerText.trim() : '',
      };
    });
    const tot = tr.querySelector('td.stats-table__total-votes-cell');
    const totDivs = tot ? Array.from(tot.querySelectorAll('div')).map(d => d.innerText.trim()) : [];
    return {
      row_class: tr.className,
      afl_player_id: fav ? fav.getAttribute('data-favourite-provider-id') : null,
      given_name: nameParts[0] || '',
      surname: nameParts.slice(1).join(' '),
      badge_alt: badge ? badge.getAttribute('alt') : null,
      header_text: tr.querySelector('th') ? tr.querySelector('th').innerText.trim() : '',
      cells: cells,
      total_actual: totDivs[0] || '',
      total_predicted: totDivs[1] || '',
    };
  });
  return {headers, rows};
}
"""


async def _run() -> dict:
    from playwright.async_api import async_playwright

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list] = {
        "bfawards_leaderboard": [], "bfawards_season": [], "predictor_page": [],
        "matches": [], "teams": [],
    }
    seen_urls: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1500, "height": 2200})
        page = await ctx.new_page()

        async def on_response(resp):
            kind = _classify(resp.url)
            if kind is None:
                return
            try:
                body = await resp.json()
            except Exception as exc:  # noqa: BLE001
                seen_urls.append({"url": resp.url, "status": resp.status, "kind": kind, "error": str(exc)})
                return
            seen_urls.append({"url": resp.url, "status": resp.status, "kind": kind, "retrieved_at": _now()})
            buckets[kind].append({"url": resp.url, "status": resp.status, "retrieved_at": _now(), "body": body})

        page.on("response", on_response)
        # Ad/analytics beacons never let the network go idle; wait on the table instead.
        await page.goto(TRACKER_URL, wait_until="domcontentloaded", timeout=90_000)
        await page.wait_for_selector(ROW_SELECTOR, timeout=60_000)
        await page.wait_for_timeout(3_000)

        # ---- Leaderboard pagination: click "Show next 15 results" until gone ----
        pagination_log: list[dict] = []
        n_rows = await page.locator(ROW_SELECTOR).count()
        pagination_log.append({"click": 0, "rows_visible": n_rows})
        for click in range(1, MAX_LOAD_MORE_CLICKS + 1):
            btn = page.locator(LOAD_MORE_SELECTOR)
            if await btn.count() == 0 or not await btn.first.is_visible():
                break
            label = (await btn.first.inner_text()).strip()
            await btn.first.scroll_into_view_if_needed()
            await btn.first.click()
            try:
                await page.wait_for_function(
                    "(n) => document.querySelectorAll('tr.stats-table__body-row').length > n",
                    arg=n_rows, timeout=20_000,
                )
            except Exception:  # noqa: BLE001
                pagination_log.append({"click": click, "label": label, "rows_visible": n_rows,
                                       "note": "row count did not increase within 20s; stopping"})
                break
            await page.wait_for_timeout(1_200)  # let the per-page predictor XHR land
            n_rows = await page.locator(ROW_SELECTOR).count()
            pagination_log.append({"click": click, "label": label, "rows_visible": n_rows})
        button_still_present = await page.locator(LOAD_MORE_SELECTOR).count() > 0
        await page.wait_for_timeout(2_000)

        dom = await page.evaluate(DOM_SNAPSHOT_JS)
        await browser.close()

    status = {
        "fetcher_version": FETCHER_VERSION,
        "tracker_url": TRACKER_URL,
        "retrieved_at": _now(),
        "season_cd": SEASON_CD,
        "comp_season_id": COMP_SEASON_ID,
        "pagination": pagination_log,
        "load_more_button_present_at_end": button_still_present,
        "dom_rows_after_pagination": len(dom["rows"]),
        "responses_captured": {k: len(v) for k, v in buckets.items()},
        "response_log": seen_urls,
    }

    def _write(name: str, obj) -> None:
        (RAW_DIR / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))

    def _latest(kind: str):
        return buckets[kind][-1] if buckets[kind] else None

    _write("afl_bfawards_season_CD_S2026014.json", _latest("bfawards_season"))
    _write("afl_bfawards_leaderboard_CD_S2026014.json", _latest("bfawards_leaderboard"))
    _write("aflapi_award_brownlow_predictor_pages.json", buckets["predictor_page"])
    # Trim fixture/teams to the fields the builder needs (provenance, not a full mirror).
    m = _latest("matches")
    if m:
        m_trim = dict(m)
        m_trim["body"] = {
            "pageInfo": m["body"].get("pageInfo"),
            "matches": [
                {k: x.get(k) for k in ("id", "providerId", "utcStartTime", "status", "round", "home", "away", "venue")}
                for x in m["body"].get("matches", [])
            ],
        }
        _write("aflapi_matches_compseason85.json", m_trim)
    t = _latest("teams")
    if t:
        t_trim = dict(t)
        t_trim["body"] = {"teams": [
            {k: x.get(k) for k in ("id", "providerId", "abbreviation", "name", "nickname")}
            for x in t["body"].get("teams", [])
        ]}
        _write("aflapi_teams_compseason85.json", t_trim)
    _write("dom_leaderboard_snapshot.json", dom)
    _write("fetch_status.json", status)
    return status


def main() -> int:
    status = asyncio.run(_run())
    print(json.dumps({k: v for k, v in status.items() if k != "response_log"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
