#!/usr/bin/env python3
"""
Repeatable external-benchmark refresh. Run this manually (it does NOT run on
every Streamlit page load -- the dashboard reads the processed files this
writes, falling back gracefully if a source is missing/stale).

Steps: (1) ingest local Wheelo CSV, (2) parse cached remote-source HTML
snapshots for ESPN/Betfair (fetching a fresh one is a separate, manual step --
see the module docstrings in src/external/*_source.py for the exact URLs;
this script deliberately does not perform network fetches itself, so a
single command remains safe to run repeatedly without hitting rate limits),
(3) identities are normalised inside each source module, (4) validate,
(5) write processed benchmark files, (6) write a source-status/provenance
report.

Usage: python scripts/refresh_external_benchmarks.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.external.aggregate import build_external_overview
from src.external.wheelo_source import identity_status_summary, load_wheelo_match_level, load_wheelo_season
from src.external.espn_source import load_espn_snapshot
from src.external.betfair_source import load_betfair_season

PROCESSED = ROOT / "data" / "external" / "processed"
RAW = ROOT / "data" / "external" / "raw"


def _latest(pattern: str) -> Path | None:
    hits = sorted(RAW.glob(pattern))
    return hits[-1] if hits else None


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    status = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "sources": {}}

    # 1. Wheelo (local CSV, always available -- the primary source).
    try:
        match_level = load_wheelo_match_level()
        season = load_wheelo_season()
        match_level.to_csv(PROCESSED / "wheelo_match_level.csv", index=False)
        season.to_csv(PROCESSED / "wheelo_season.csv", index=False)
        status["sources"]["wheelo"] = {
            "status": "ok", "url": None,
            "path": str(ROOT / "data" / "external" / "wheelo-brownlow-predictions.csv"),
            "n_rows": len(match_level), "identity_summary": identity_status_summary(),
        }
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: refresh must never crash the app
        status["sources"]["wheelo"] = {"status": "failed", "error": str(exc)}

    # 2. ESPN (parsed from the latest cached raw HTML snapshot, if any).
    espn_raw = _latest("espn_*.html")
    if espn_raw is None:
        status["sources"]["espn"] = {"status": "unavailable", "reason": "no cached snapshot found"}
    else:
        try:
            espn = load_espn_snapshot(espn_raw)
            espn.to_csv(PROCESSED / "espn_season.csv", index=False)
            status["sources"]["espn"] = {
                "status": "ok" if not espn.empty else "parsed_empty",
                "url": "https://www.espn.com.au/afl/story/_/page/POINTSBET20242/afl-2026-brownlow-medal-predictor-tracker-leaderboard-odds-every-vote",
                "raw_snapshot": str(espn_raw), "parser_version": "espn_source.py:v1",
                "n_rows": len(espn),
                "note": "partial coverage -- ESPN publishes its own top-N leaderboard, not the full competition",
            }
        except Exception as exc:  # noqa: BLE001
            status["sources"]["espn"] = {"status": "failed", "error": str(exc), "raw_snapshot": str(espn_raw)}

    # 3. Betfair (parsed from the latest cached raw HTML snapshot, if any).
    betfair_raw = _latest("betfair_*.html")
    if betfair_raw is None:
        status["sources"]["betfair"] = {"status": "unavailable", "reason": "no cached snapshot found"}
    else:
        try:
            betfair = load_betfair_season(betfair_raw)
            betfair.to_csv(PROCESSED / "betfair_season.csv", index=False)
            status["sources"]["betfair"] = {
                "status": "ok" if not betfair.empty else "parsed_empty",
                "url": "https://www.betfair.com.au/hub/sports/afl/brownlow-medal-predictor/",
                "raw_snapshot": str(betfair_raw), "parser_version": "betfair_source.py:v1",
                "n_rows": len(betfair),
                "note": "partial coverage -- only 22 of 207 2026 matches had a recap article in the "
                        "2026-09-18 snapshot; round-level attribution not reliable, season-total only",
            }
        except Exception as exc:  # noqa: BLE001
            status["sources"]["betfair"] = {"status": "failed", "error": str(exc), "raw_snapshot": str(betfair_raw)}

    # AFL.com.au: documented as permanently unavailable via static fetch for this snapshot round.
    afl_raw = _latest("afl_com_au_*.html")
    status["sources"]["afl_com_au"] = {
        "status": "unavailable",
        "reason": "page is a JavaScript-rendered navigation shell; no prediction data present in "
                  "static HTML (confirmed by direct fetch, not merely a summarisation-tool guess)",
        "raw_snapshot": str(afl_raw) if afl_raw else None,
        "url": "https://www.afl.com.au/brownlow-medal/predictor",
    }
    wheelo_site_raw = _latest("wheelo_site_*.html")
    status["sources"]["wheelo_live_site"] = {
        "status": "unavailable",
        "reason": "single-page app; static HTML contains only a player-name dropdown list, no "
                  "prediction values (data is loaded via a JS/API call not captured by a static fetch). "
                  "The local CSV (see 'wheelo' above) is used as the actual Wheelo source instead.",
        "raw_snapshot": str(wheelo_site_raw) if wheelo_site_raw else None,
        "url": "https://www.wheeloratings.com/afl_brownlow.html",
    }

    # 5. Overview / consensus (only meaningful once the above are written).
    overview = build_external_overview()
    overview.to_csv(PROCESSED / "external_overview.csv", index=False)

    # 6. Status/provenance report.
    (PROCESSED / "source_status.json").write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()
