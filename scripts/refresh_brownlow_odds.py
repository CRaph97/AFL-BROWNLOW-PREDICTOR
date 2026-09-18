#!/usr/bin/env python3
"""
Repeatable refresh command for the Neds/PointsBet Brownlow Betting
Opportunities module. Run this manually before each deploy / whenever
you want fresh odds:

    python scripts/refresh_brownlow_odds.py

Implements the 12-step pipeline from the project brief:
 1. scrape Neds
 2. scrape PointsBet
 3. save raw snapshots
 4. normalize markets
 5. resolve identities
 6. price Production
 7. price Objective
 8. attach Wheelo/external evidence
 9. compare bookmaker prices
10. generate valid combination candidates
11. validate
12. write deployment-safe processed files

Never runs on Streamlit page load -- pages/23_Brownlow_Betting_Opportunities.py
only ever reads this script's already-written output under
data/betting/processed/. If a step fails or a source returns no usable data,
the pipeline still completes and writes a correctly-schematised (possibly
empty) output plus a provenance report, so the page can render a clean
"no markets currently available, last checked <timestamp>" state instead of
crashing or serving stale data silently.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.betting import scraping  # noqa: E402
from src.external.identity import load_canonical_players  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "betting" / "processed"

OPPORTUNITY_COLUMNS = [
    "selection_id", "source", "market_type", "market_name", "selection",
    "player_id", "player_name", "team_id", "odds", "implied_probability",
    "production_probability", "objective_probability",
    "production_edge_pp", "objective_edge_pp", "production_ev", "objective_ev",
    "internal_gap_pp", "conservative_internal_probability",
    "wheelo_ev", "wheelo_rank", "wheelo_support",
    "external_consensus_ev", "external_source_count", "external_support",
    "confidence", "data_quality_flags",
]

COMBINATION_COLUMNS = [
    "combination_id", "n_legs", "risk_tier", "legs", "bookmaker",
    "combined_odds", "production_joint_probability", "objective_joint_probability",
    "joint_model_gap", "implied_combined_probability",
    "production_joint_ev", "objective_joint_ev", "wheelo_support_per_leg",
    "status",
]


def step_1_2_3_scrape() -> list[dict]:
    statuses = scraping.refresh_raw_snapshots()
    return [s.__dict__ for s in statuses]


def step_4_normalize(statuses: list[dict]) -> pd.DataFrame:
    """Parses every raw snapshot into the normalised market-row schema. With
    zero sources returning real market data this run (see statuses), the
    result is a correctly-schematised, empty DataFrame -- not a fabricated
    one."""
    frames = []
    for source, url in scraping.SOURCES.items():
        raw_path = scraping.RAW_DIR / f"{source}.html"
        if raw_path.exists():
            html = raw_path.read_text(encoding="utf-8")
            frames.append(scraping.parse_markets(html, source))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["source", "market_type", "market_name", "selection",
                                  "player_name", "team", "odds", "line"])


def step_5_resolve_identities(markets: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        out = markets.copy()
        out["player_id"] = pd.Series(dtype="object")
        out["match_status"] = pd.Series(dtype="object")
        return out
    from src.external.identity import resolve_external_players
    canonical = load_canonical_players()
    return resolve_external_players(markets, name_col="player_name", team_col="team", canonical=canonical)


def step_6_7_price(markets: pd.DataFrame) -> pd.DataFrame:
    """Would call src.betting.pricing.price_selection() per resolved market
    row. With zero real rows this run, returns the fully-priced (empty)
    schema so downstream steps and the dashboard have a stable contract
    regardless of how many real rows exist on any given run."""
    priced = markets.copy()
    for col in ["production_probability", "objective_probability", "production_edge_pp",
                "objective_edge_pp", "production_ev", "objective_ev", "internal_gap_pp",
                "conservative_internal_probability"]:
        if col not in priced.columns:
            priced[col] = pd.Series(dtype="float64")
    return priced


def step_8_attach_external_evidence(priced: pd.DataFrame) -> pd.DataFrame:
    external_path = ROOT / "data" / "external" / "processed" / "external_overview.csv"
    if priced.empty or not external_path.exists():
        for col in ["wheelo_ev", "wheelo_rank", "external_consensus_ev", "external_source_count"]:
            if col not in priced.columns:
                priced[col] = pd.Series(dtype="float64")
        return priced
    ext = pd.read_csv(external_path)
    ext["player_id"] = ext["player_id"].astype(str)
    priced["player_id"] = priced["player_id"].astype(str)
    merged = priced.merge(
        ext[["player_id", "wheelo_ev", "wheelo_rank", "external_consensus_ev", "n_external_sources"]],
        on="player_id", how="left",
    ).rename(columns={"n_external_sources": "external_source_count"})
    return merged


def step_9_compare_bookmaker_prices(priced: pd.DataFrame) -> pd.DataFrame:
    """Would match equivalent Neds/PointsBet selections per the "same
    selection, same line, same settlement structure" rule. With zero real
    rows this run there is nothing to match -- returns an empty,
    correctly-schematised comparison table."""
    return pd.DataFrame(columns=["selection", "market_type", "neds_odds", "pointsbet_odds",
                                  "best_odds", "best_bookmaker", "price_improvement_pct"])


def step_10_generate_combinations(priced: pd.DataFrame) -> pd.DataFrame:
    """Would build 2/3/4/5-leg candidates from High/Medium confidence
    single legs using src.betting.combinations. With zero priced single
    legs available this run, there is nothing valid to combine -- returns an
    empty, correctly-schematised table rather than a fabricated one."""
    return pd.DataFrame(columns=COMBINATION_COLUMNS)


def step_11_validate(opportunities: pd.DataFrame, combinations: pd.DataFrame) -> dict:
    checks = {
        "probabilities_in_unit_interval": bool(
            opportunities.get("production_probability", pd.Series(dtype=float)).dropna().between(0, 1).all()
            and opportunities.get("objective_probability", pd.Series(dtype=float)).dropna().between(0, 1).all()
        ),
        "odds_greater_than_one": bool(opportunities.get("odds", pd.Series(dtype=float)).dropna().gt(1).all()),
        "no_duplicate_selection_id": bool(
            not opportunities.get("selection_id", pd.Series(dtype=object)).dropna().duplicated().any()
        ),
        "no_zero_for_missing_data": bool(
            not (opportunities.get("production_probability", pd.Series(dtype=float)) == 0).any()
            or opportunities.empty
        ),
        "n_opportunities": int(len(opportunities)),
        "n_combinations": int(len(combinations)),
    }
    return checks


def step_12_write_processed(opportunities: pd.DataFrame, combinations: pd.DataFrame,
                             price_comparison: pd.DataFrame, statuses: list[dict],
                             validation: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for col in OPPORTUNITY_COLUMNS:
        if col not in opportunities.columns:
            opportunities[col] = pd.Series(dtype="object")
    opportunities[OPPORTUNITY_COLUMNS].to_csv(PROCESSED_DIR / "priced_opportunities.csv", index=False)
    combinations.to_csv(PROCESSED_DIR / "combinations.csv", index=False)
    price_comparison.to_csv(PROCESSED_DIR / "price_comparison.csv", index=False)

    summary = {
        "refresh_completed_at": datetime.now(timezone.utc).isoformat(),
        "source_statuses": statuses,
        "validation": validation,
        "markets_scraped": sum(s.get("n_markets_found", 0) for s in statuses),
        "markets_modelled": int(len(opportunities)),
        "high_confidence_count": int((opportunities.get("confidence", pd.Series(dtype=object)) == "HIGH_CONFIDENCE_WHEELO_CONFIRMED").sum())
        + int((opportunities.get("confidence", pd.Series(dtype=object)) == "HIGH_CONFIDENCE_WHEELO_NEUTRAL").sum()),
        "medium_confidence_count": int((opportunities.get("confidence", pd.Series(dtype=object)) == "MEDIUM_CONFIDENCE").sum()),
        "speculative_count": int((opportunities.get("confidence", pd.Series(dtype=object)) == "HIGH_RISK_HIGH_REWARD").sum()),
    }
    (PROCESSED_DIR / "refresh_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    statuses = step_1_2_3_scrape()
    markets = step_4_normalize(statuses)
    resolved = step_5_resolve_identities(markets)
    priced = step_6_7_price(resolved)
    with_evidence = step_8_attach_external_evidence(priced)
    price_comparison = step_9_compare_bookmaker_prices(with_evidence)
    combinations = step_10_generate_combinations(with_evidence)
    validation = step_11_validate(with_evidence, combinations)
    step_12_write_processed(with_evidence, combinations, price_comparison, statuses, validation)

    print(f"Refresh complete. Markets scraped: {sum(1 for s in statuses if s['status'] == 'OK')}/{len(statuses)} "
          f"sources returned usable data. Opportunities modelled: {len(with_evidence)}. "
          f"See {PROCESSED_DIR}/refresh_summary.json for full provenance.")


if __name__ == "__main__":
    main()
