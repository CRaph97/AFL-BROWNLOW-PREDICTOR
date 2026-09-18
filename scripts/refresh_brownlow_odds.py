#!/usr/bin/env python3
"""
Repeatable refresh command for the Neds/PointsBet Brownlow Betting
Opportunities module. Run this manually before each deploy / whenever
you want fresh odds:

    python scripts/refresh_brownlow_odds.py

Implements the 12-step pipeline from the project brief:
 1. scrape Neds (real Playwright browser automation)
 2. scrape PointsBet (real Playwright browser automation)
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.betting import scraping  # noqa: E402
from src.betting import pricing  # noqa: E402
from src.betting import combinations as combos  # noqa: E402
from src.betting.market_data import (  # noqa: E402
    load_production_simulations, load_objective_simulations, load_team_lookup,
)
from src.betting.classification import classify_confidence, classify_wheelo_support  # noqa: E402
from src.external.identity import load_canonical_players, resolve_external_players  # noqa: E402

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
    """Parses every raw snapshot into the normalised market-row schema."""
    return scraping.parse_all_from_snapshots()


def _resolve_team_id(team_string: str | None, canonical_teams: set[str]) -> str | None:
    if not team_string:
        return None
    tid = scraping._team_from_string(team_string)
    return tid if tid in canonical_teams else None


def step_5_resolve_identities(markets: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        out = markets.copy()
        out["player_id"] = pd.Series(dtype="object")
        out["match_status"] = pd.Series(dtype="object")
        out["team_id"] = pd.Series(dtype="object")
        return out

    canonical = load_canonical_players()
    canonical_teams = set(canonical["team_id"].unique())

    with_player = markets[markets["player_name"].notna()].copy()
    without_player = markets[markets["player_name"].isna()].copy()

    if not with_player.empty:
        # resolve_external_players() joins on an already-canonical team_id
        # (confirmed against src/external/wheelo_source.py's established
        # usage) -- bookmaker team strings ("Collingwood", "Western
        # Bulldogs") must be mapped through config/team_mapping.csv FIRST,
        # exactly like every other identity-resolution call in this project.
        # Passing the raw display string here silently failed to match on
        # team for every single row (case/format mismatch), which is what
        # caused an initial, wrong "IDENTITY_AMBIGUOUS" flag on obviously
        # resolvable players like Nick Daicos -- caught during validation,
        # not shipped.
        with_player["team_for_resolution"] = with_player["team"].apply(
            lambda t: scraping._team_from_string(t) if isinstance(t, str) else None
        )
        resolved = resolve_external_players(
            with_player, name_col="player_name", team_col="team_for_resolution", canonical=canonical
        )
        resolved = resolved.drop(columns=["team_for_resolution"], errors="ignore")
    else:
        resolved = with_player.copy()
        resolved["player_id"] = pd.Series(dtype="object")
        resolved["match_status"] = pd.Series(dtype="object")

    # Team-level rows (TEAM_VOTES_OU): resolve the team string to a canonical
    # team_id; player_id stays NA (this row prices against every player on
    # the team jointly, not one player).
    without_player["team_id_resolved"] = without_player["team"].apply(
        lambda t: _resolve_team_id(t, canonical_teams)
    )
    without_player["player_id"] = pd.NA
    without_player["match_status"] = without_player["team_id_resolved"].apply(
        lambda t: "resolved" if t else "unresolved"
    )

    combined = pd.concat([resolved, without_player], ignore_index=True, sort=False)
    if "team_id" not in combined.columns:
        combined["team_id"] = pd.NA
    # For player-resolved rows, team_id already came back from
    # resolve_external_players (the canonical player's team). For team-level
    # rows, fill from team_id_resolved.
    if "team_id_resolved" in combined.columns:
        combined["team_id"] = combined["team_id"].where(
            combined["team_id"].notna(), combined["team_id_resolved"]
        )
        combined = combined.drop(columns=["team_id_resolved"])
    return combined


def _flags_for_row(row: pd.Series, price_status: str | None) -> list[str]:
    flags = []
    if row.get("market_type") == "UNMODELLED":
        flags.append("UNMODELLED")
    if row.get("match_status") == "ambiguous":
        flags.append("IDENTITY_AMBIGUOUS")
    if row.get("match_status") == "unresolved" and row.get("market_type") != "UNMODELLED":
        flags.append("IDENTITY_AMBIGUOUS")
    if price_status and "not in simulation set" in str(price_status):
        flags.append("INSUFFICIENT_SIMULATION_SUPPORT")
    odds = row.get("odds")
    if odds is not None and not pd.isna(odds) and odds <= 1.0:
        flags.append("PRICE_SUSPECT")
    return flags


def _price_single_player_row(row: pd.Series, prod_sims, obj_sims) -> tuple[float | None, float | None, str | None]:
    mt = row["market_type"]
    pid = row.get("player_id")
    if mt == "UNMODELLED" or pd.isna(pid):
        return None, None, "UNMODELLED" if mt == "UNMODELLED" else "unresolved identity"
    pid = str(pid)
    kwargs = {}
    if mt == "WINNER":
        kwargs = {"player_id": pid}
    elif mt == "TOP_N":
        kwargs = {"player_id": pid, "n": int(row["n"])}
    elif mt == "EXACT_POSITION":
        kwargs = {"player_id": pid, "position": int(row["position"])}
    elif mt == "PLAYER_VOTES_OU":
        kwargs = {"player_id": pid, "line": float(row["line"]), "side": row["side"]}
    elif mt == "X_PLUS_VOTES":
        kwargs = {"player_id": pid, "threshold": int(row["threshold"])}
    elif mt == "TO_POLL_A_VOTE":
        kwargs = {"player_id": pid}
    else:
        return None, None, None  # handled elsewhere (TEAM_VOTES_OU, PLAYER_H2H)

    prod_result = pricing.price_selection(prod_sims, mt, **kwargs)
    obj_result = pricing.price_selection(obj_sims, mt, **kwargs)
    return prod_result.probability, obj_result.probability, prod_result.status


def _price_team_row(row: pd.Series, prod_sims, obj_sims, team_lookup: pd.Series) -> tuple[float | None, float | None, str | None]:
    team_id = row.get("team_id")
    if pd.isna(team_id) or row["market_type"] == "UNMODELLED":
        return None, None, "unresolved team" if row["market_type"] != "UNMODELLED" else "UNMODELLED"
    team_players = team_lookup[team_lookup == team_id].index.tolist()
    if not team_players:
        return None, None, "no players resolved for team"
    kwargs = {"team_players": team_players, "line": float(row["line"]), "side": row["side"]}
    prod_result = pricing.price_selection(prod_sims, "TEAM_VOTES_OU", **kwargs)
    obj_result = pricing.price_selection(obj_sims, "TEAM_VOTES_OU", **kwargs)
    return prod_result.probability, obj_result.probability, prod_result.status


def _price_h2h_group(group: pd.DataFrame, prod_sims, obj_sims) -> pd.DataFrame:
    """A PLAYER_H2H market's two rows share (source, market_name). Resolve
    each row's own probability as "this row's player wins the head-to-head",
    from price_player_h2h's {a_wins, b_wins} -- never guessed, never
    assuming symmetry."""
    out = group.copy()
    if len(group) != 2 or group["player_id"].isna().any():
        out["production_probability"] = None
        out["objective_probability"] = None
        return out
    ids = group["player_id"].astype(str).tolist()
    prod_h2h = pricing.price_player_h2h(prod_sims, ids[0], ids[1])
    obj_h2h = pricing.price_player_h2h(obj_sims, ids[0], ids[1])
    out.iloc[0, out.columns.get_loc("production_probability")] = prod_h2h.get("a_wins")
    out.iloc[1, out.columns.get_loc("production_probability")] = prod_h2h.get("b_wins")
    out.iloc[0, out.columns.get_loc("objective_probability")] = obj_h2h.get("a_wins")
    out.iloc[1, out.columns.get_loc("objective_probability")] = obj_h2h.get("b_wins")
    return out


def step_6_7_price(markets: pd.DataFrame) -> pd.DataFrame:
    priced = markets.copy()
    for col in ["production_probability", "objective_probability"]:
        priced[col] = pd.Series([None] * len(priced), dtype="object")
    if priced.empty:
        for col in ["production_edge_pp", "objective_edge_pp", "production_ev", "objective_ev",
                    "internal_gap_pp", "conservative_internal_probability", "implied_probability"]:
            priced[col] = pd.Series(dtype="float64")
        return priced

    prod_sims = load_production_simulations()
    obj_sims = load_objective_simulations()
    team_lookup = load_team_lookup()

    price_status = [None] * len(priced)
    h2h_mask = priced["market_type"] == "PLAYER_H2H"
    team_mask = priced["market_type"] == "TEAM_VOTES_OU"
    single_mask = ~h2h_mask & ~team_mask

    for idx in priced.index[single_mask]:
        p, o, status = _price_single_player_row(priced.loc[idx], prod_sims, obj_sims)
        priced.at[idx, "production_probability"] = p
        priced.at[idx, "objective_probability"] = o
        price_status[priced.index.get_loc(idx)] = status

    for idx in priced.index[team_mask]:
        p, o, status = _price_team_row(priced.loc[idx], prod_sims, obj_sims, team_lookup)
        priced.at[idx, "production_probability"] = p
        priced.at[idx, "objective_probability"] = o
        price_status[priced.index.get_loc(idx)] = status

    if h2h_mask.any():
        h2h_rows = priced[h2h_mask]
        priced_h2h_parts = []
        for (_, _), group in h2h_rows.groupby(["source", "market_name"]):
            priced_h2h_parts.append(_price_h2h_group(group, prod_sims, obj_sims))
        if priced_h2h_parts:
            h2h_priced = pd.concat(priced_h2h_parts)
            priced.loc[h2h_priced.index, "production_probability"] = h2h_priced["production_probability"]
            priced.loc[h2h_priced.index, "objective_probability"] = h2h_priced["objective_probability"]

    priced["price_status"] = price_status

    priced["implied_probability"] = 1.0 / priced["odds"].astype(float)
    priced["production_probability"] = pd.to_numeric(priced["production_probability"], errors="coerce")
    priced["objective_probability"] = pd.to_numeric(priced["objective_probability"], errors="coerce")
    priced["production_edge_pp"] = (priced["production_probability"] - priced["implied_probability"]) * 100.0
    priced["objective_edge_pp"] = (priced["objective_probability"] - priced["implied_probability"]) * 100.0
    priced["production_ev"] = priced["production_probability"] * priced["odds"].astype(float) - 1.0
    priced["objective_ev"] = priced["objective_probability"] * priced["odds"].astype(float) - 1.0
    priced["internal_gap_pp"] = (priced["production_probability"] - priced["objective_probability"]).abs() * 100.0
    priced["conservative_internal_probability"] = priced[["production_probability", "objective_probability"]].min(axis=1)

    priced["data_quality_flags"] = [
        ",".join(_flags_for_row(priced.loc[i], price_status[priced.index.get_loc(i)])) or None
        for i in priced.index
    ]
    return priced


def step_8_attach_external_evidence(priced: pd.DataFrame) -> pd.DataFrame:
    external_path = ROOT / "data" / "external" / "processed" / "external_overview.csv"
    evidence_cols = ["wheelo_ev", "wheelo_rank", "external_consensus_ev", "external_source_count"]
    if priced.empty or not external_path.exists():
        for col in evidence_cols:
            if col not in priced.columns:
                priced[col] = pd.Series(dtype="float64")
        return priced
    # IMPORTANT: do not pre-create empty placeholder evidence columns before
    # the merge below -- a prior version of this function did, and because
    # `ext` also carries a real `wheelo_ev` column, pandas' merge `suffixes`
    # logic treated the empty pre-existing column as the "left" one (kept
    # unsuffixed) and silently renamed the REAL incoming data to
    # "wheelo_ev_ext" instead, shadowed and never used -- a genuine bug
    # caught only by checking the non-null count after merging, not by
    # eyeballing that the column existed. Any pre-creation must happen AFTER
    # the merge, only to backfill columns still missing (e.g. because `ext`
    # itself lacks one), never before it.
    ext = pd.read_csv(external_path)
    ext["player_id"] = ext["player_id"].astype(str)
    priced = priced.copy()
    priced["player_id_str"] = pd.Series(
        pd.array(pd.to_numeric(priced["player_id"], errors="coerce"), dtype="Int64"),
        index=priced.index,
    ).astype(str).replace("<NA>", pd.NA)
    merged = priced.merge(
        ext[["player_id", "wheelo_ev", "wheelo_rank", "external_consensus_ev", "n_external_sources"]],
        left_on="player_id_str", right_on="player_id", how="left", suffixes=("", "_ext"),
    ).rename(columns={"n_external_sources": "external_source_count"})
    if "player_id_ext" in merged.columns:
        merged = merged.drop(columns=["player_id_ext"])
    merged = merged.drop(columns=["player_id_str"], errors="ignore")
    for col in evidence_cols:
        if col not in merged.columns:
            merged[col] = pd.Series(dtype="float64")
    return merged


def _classify_row(row: pd.Series) -> tuple[str, str]:
    flags = row.get("data_quality_flags")
    settlement_flag = None
    if isinstance(flags, str) and any(
        f in flags for f in ("IDENTITY_AMBIGUOUS", "SETTLEMENT_REVIEW_REQUIRED", "PRICE_SUSPECT", "UNMODELLED",
                              "INSUFFICIENT_SIMULATION_SUPPORT")
    ):
        settlement_flag = flags

    prod_ev_positive = row.get("production_ev")
    wheelo_ev = row.get("wheelo_ev")
    direction_positive = bool(prod_ev_positive is not None and not pd.isna(prod_ev_positive) and prod_ev_positive > 0)
    wheelo_support = classify_wheelo_support(
        direction_positive,
        wheelo_ev if pd.notna(wheelo_ev) else None,
        row.get("production_ev") if pd.notna(row.get("production_ev")) else None,
        row.get("objective_ev") if pd.notna(row.get("objective_ev")) else None,
    )
    result = classify_confidence(
        row.get("production_edge_pp") if pd.notna(row.get("production_edge_pp")) else None,
        row.get("objective_edge_pp") if pd.notna(row.get("objective_edge_pp")) else None,
        row.get("internal_gap_pp") if pd.notna(row.get("internal_gap_pp")) else None,
        wheelo_support,
        settlement_flag,
    )
    return result.confidence, wheelo_support


def step_8b_classify(priced: pd.DataFrame) -> pd.DataFrame:
    if priced.empty:
        priced["confidence"] = pd.Series(dtype="object")
        priced["wheelo_support"] = pd.Series(dtype="object")
        return priced
    results = priced.apply(_classify_row, axis=1, result_type="expand")
    priced["confidence"] = results[0]
    priced["wheelo_support"] = results[1]
    priced["external_support"] = "INSUFFICIENT_DATA"  # ESPN/Betfair are context-only; no quantitative gate here
    return priced


def step_9_compare_bookmaker_prices(priced: pd.DataFrame) -> pd.DataFrame:
    """Matches equivalent Neds/PointsBet selections: same market_type, same
    resolved player_id (or team_id for team markets), same line, same side.
    Never merges non-equivalent markets."""
    cols = ["selection", "market_type", "neds_odds", "pointsbet_odds", "best_odds",
            "best_bookmaker", "price_improvement_pct"]
    if priced.empty:
        return pd.DataFrame(columns=cols)

    key_cols = ["market_type", "player_id", "team_id", "line", "side", "n", "position", "threshold"]
    priced = priced.copy()
    priced["_source_group"] = priced["source"].apply(lambda s: "pointsbet" if s.startswith("pointsbet") else s)
    priced["_key"] = priced[key_cols].apply(
        lambda r: "|".join("" if pd.isna(v) else str(v) for v in r), axis=1
    )

    rows = []
    for key, group in priced.groupby("_key"):
        sources_present = set(group["_source_group"])
        if not ({"neds", "pointsbet"} <= sources_present):
            continue
        neds_row = group[group["_source_group"] == "neds"].iloc[0]
        pb_row = group[group["_source_group"] == "pointsbet"].iloc[0]
        if pd.isna(neds_row["odds"]) or pd.isna(pb_row["odds"]):
            continue
        best_odds = max(neds_row["odds"], pb_row["odds"])
        best_book = "Neds" if neds_row["odds"] >= pb_row["odds"] else "PointsBet"
        worst = min(neds_row["odds"], pb_row["odds"])
        improvement = (best_odds - worst) / worst * 100.0 if worst else None
        rows.append({
            "selection": neds_row["selection"], "market_type": neds_row["market_type"],
            "neds_odds": neds_row["odds"], "pointsbet_odds": pb_row["odds"],
            "best_odds": best_odds, "best_bookmaker": best_book,
            "price_improvement_pct": improvement,
        })
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _risk_tier(n_legs: int) -> str:
    return {2: "Lower variance", 3: "Balanced", 4: "Higher return", 5: "Speculative"}.get(n_legs, "Speculative")


def step_10_generate_combinations(priced: pd.DataFrame) -> pd.DataFrame:
    if priced.empty:
        return pd.DataFrame(columns=COMBINATION_COLUMNS)

    candidates = priced[
        priced["confidence"].isin(["HIGH_CONFIDENCE_WHEELO_CONFIRMED", "HIGH_CONFIDENCE_WHEELO_NEUTRAL", "MEDIUM_CONFIDENCE"])
        & priced["market_type"].isin(["WINNER", "TOP_N", "X_PLUS_VOTES", "TEAM_VOTES_OU"])
    ]
    candidates = candidates[candidates["player_id"].notna() | (candidates["market_type"] == "TEAM_VOTES_OU")]
    candidates = candidates.drop_duplicates(subset=["player_id", "team_id", "market_type", "n", "threshold", "line", "side"])
    if candidates.empty:
        return pd.DataFrame(columns=COMBINATION_COLUMNS)

    prod_sims = load_production_simulations()
    obj_sims = load_objective_simulations()
    team_lookup = load_team_lookup()

    legs_pool = []
    for _, row in candidates.head(12).iterrows():
        if row["market_type"] == "TEAM_VOTES_OU":
            team_players = team_lookup[team_lookup == row["team_id"]].index.tolist()
            if not team_players:
                continue
            leg = combos.build_leg(
                f"{row['team_id']} {row['side']} {row['line']} team votes", None, "team_votes_ou",
                team_players=team_players, line=float(row["line"]), side=row["side"],
                confidence=row["confidence"], wheelo_support=row.get("wheelo_support", ""),
            )
        else:
            pid = str(row["player_id"])
            if row["market_type"] == "WINNER":
                leg = combos.build_leg(f"{row['player_name']} to win", pid, "winner",
                                        confidence=row["confidence"], wheelo_support=row.get("wheelo_support", ""))
            elif row["market_type"] == "TOP_N":
                leg = combos.build_leg(f"{row['player_name']} top {int(row['n'])}", pid, "top_n", n=int(row["n"]),
                                        confidence=row["confidence"], wheelo_support=row.get("wheelo_support", ""))
            else:
                leg = combos.build_leg(f"{row['player_name']} {int(row['threshold'])}+ votes", pid, "x_plus_votes",
                                        threshold=int(row["threshold"]),
                                        confidence=row["confidence"], wheelo_support=row.get("wheelo_support", ""))
        legs_pool.append((leg, row["odds"], row["source"]))

    import itertools
    rows = []
    combo_id = 0
    for n_legs in (2, 3, 4, 5):
        n_found = 0
        for combo in itertools.combinations(legs_pool, n_legs):
            if n_found >= 3:
                break
            leg_id_sets = [set(l[0].player_ids) for l in combo]
            if any(a & b for i, a in enumerate(leg_id_sets) for b in leg_id_sets[i + 1:]):
                continue  # overlapping players across legs -- not a valid distinct-leg combination
            legs = [c[0] for c in combo]
            result = combos.price_combination(legs, prod_sims, obj_sims)
            if result.rejected_reason:
                continue
            combo_id += 1
            n_found += 1
            rows.append({
                "combination_id": f"combo_{combo_id}", "n_legs": n_legs, "risk_tier": _risk_tier(n_legs),
                "legs": " + ".join(l.label for l in legs), "bookmaker": "candidate — verify price with bookmaker",
                "combined_odds": None,
                "production_joint_probability": result.production_joint_probability,
                "objective_joint_probability": result.objective_joint_probability,
                "joint_model_gap": result.joint_model_gap,
                "implied_combined_probability": None,
                "production_joint_ev": None, "objective_joint_ev": None,
                "wheelo_support_per_leg": "; ".join(l.wheelo_support or "n/a" for l in legs),
                "status": "candidate — verify price with bookmaker" + (" [CORRELATED LEGS]" if result.correlated_legs_flag else ""),
            })
    return pd.DataFrame(rows, columns=COMBINATION_COLUMNS) if rows else pd.DataFrame(columns=COMBINATION_COLUMNS)


def step_11_validate(opportunities: pd.DataFrame, combinations_df: pd.DataFrame) -> dict:
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
            not ((opportunities.get("production_probability", pd.Series(dtype=float)).notna())
                 & (opportunities.get("production_probability", pd.Series(dtype=float)) == 0)
                 & opportunities.get("player_id", pd.Series(dtype=object)).isna()).any()
        ) if not opportunities.empty else True,
        "flagged_rows_excluded_from_headline": bool(
            opportunities[opportunities["data_quality_flags"].notna()]["confidence"].isin(
                ["HIGH_CONFIDENCE_WHEELO_CONFIRMED", "HIGH_CONFIDENCE_WHEELO_NEUTRAL", "MEDIUM_CONFIDENCE"]
            ).sum() == 0
        ) if not opportunities.empty and "data_quality_flags" in opportunities.columns else True,
        "n_opportunities": int(len(opportunities)),
        "n_combinations": int(len(combinations_df)),
    }
    return checks


def step_12_write_processed(opportunities: pd.DataFrame, combinations_df: pd.DataFrame,
                             price_comparison: pd.DataFrame, statuses: list[dict],
                             validation: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for col in OPPORTUNITY_COLUMNS:
        if col not in opportunities.columns:
            opportunities[col] = pd.Series(dtype="object")
    opportunities[OPPORTUNITY_COLUMNS].to_csv(PROCESSED_DIR / "priced_opportunities.csv", index=False)
    combinations_df.to_csv(PROCESSED_DIR / "combinations.csv", index=False)
    price_comparison.to_csv(PROCESSED_DIR / "price_comparison.csv", index=False)

    summary = {
        "refresh_completed_at": datetime.now(timezone.utc).isoformat(),
        "source_statuses": statuses,
        "validation": validation,
        "markets_scraped": sum(s.get("n_markets_found", 0) for s in statuses),
        "markets_modelled": int((opportunities["market_type"] != "UNMODELLED").sum()) if not opportunities.empty else 0,
        "high_confidence_count": int((opportunities.get("confidence", pd.Series(dtype=object)) == "HIGH_CONFIDENCE_WHEELO_CONFIRMED").sum())
        + int((opportunities.get("confidence", pd.Series(dtype=object)) == "HIGH_CONFIDENCE_WHEELO_NEUTRAL").sum()),
        "medium_confidence_count": int((opportunities.get("confidence", pd.Series(dtype=object)) == "MEDIUM_CONFIDENCE").sum()),
        "speculative_count": int((opportunities.get("confidence", pd.Series(dtype=object)) == "HIGH_RISK_HIGH_REWARD").sum()),
    }
    (PROCESSED_DIR / "refresh_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def main() -> None:
    statuses = step_1_2_3_scrape()
    markets = step_4_normalize(statuses)
    resolved = step_5_resolve_identities(markets)
    priced = step_6_7_price(resolved)
    with_evidence = step_8_attach_external_evidence(priced)
    classified = step_8b_classify(with_evidence)
    price_comparison = step_9_compare_bookmaker_prices(classified)
    combinations_df = step_10_generate_combinations(classified)
    validation = step_11_validate(classified, combinations_df)
    step_12_write_processed(classified, combinations_df, price_comparison, statuses, validation)

    print(f"Refresh complete. Markets scraped: {sum(1 for s in statuses if s['status'] == 'OK')}/{len(statuses)} "
          f"sources returned usable data. Opportunities modelled: {len(classified)}. "
          f"See {PROCESSED_DIR}/refresh_summary.json for full provenance.")


if __name__ == "__main__":
    main()
