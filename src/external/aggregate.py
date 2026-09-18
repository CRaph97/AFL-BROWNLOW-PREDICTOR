"""
Cross-source aggregation: our_midpoint, external_consensus_ev, agreement
categories. Pure computation over already-resolved per-source tables --
never mutates Production or Objective, never fabricates a value for a source
that doesn't have one (NaN propagates through every mean/comparison here).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

# Agreement-category thresholds, fixed BEFORE looking at results (documented
# here, not tuned against what "looks right" per the project brief). "Agree"
# means two EV-type values are within 15% of their own average, or within an
# absolute 2.0 votes for low-EV players where a percentage threshold is too
# strict to be meaningful (e.g. 0.2 vs 0.3 votes is a 40% relative gap that
# is clearly not a meaningful disagreement in absolute terms).
AGREEMENT_REL_THRESHOLD = 0.15
AGREEMENT_ABS_THRESHOLD = 2.0


def _agree(a: float, b: float) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    if abs(a - b) <= AGREEMENT_ABS_THRESHOLD:
        return True
    denom = (abs(a) + abs(b)) / 2 or 1.0
    return abs(a - b) / denom <= AGREEMENT_REL_THRESHOLD


def build_external_overview() -> pd.DataFrame:
    """One row per canonical player_id, joining Production, Objective, and
    every external EV-type source, plus the consensus/agreement columns
    specified in the project brief. Rank-only or editorial sources are
    intentionally NOT included here (they cannot be averaged into an EV
    consensus without fabricating a number)."""
    from src.external.wheelo_source import load_wheelo_season
    from src.external.espn_source import load_espn_snapshot
    from src.external.betfair_source import load_betfair_season
    from src.external.identity import load_canonical_players

    prod = pd.read_csv(REPORTS / "2026_leaderboard.csv")[["player_id", "rank", "FINAL_ENSEMBLE"]].copy()
    prod["player_id"] = prod["player_id"].astype(str)
    prod = prod.rename(columns={"rank": "production_rank", "FINAL_ENSEMBLE": "production_ev"})

    obj = pd.read_csv(REPORTS / "2026_objective_leaderboard.csv")[["player_id", "rank", "objective_ev"]].copy()
    obj["player_id"] = obj["player_id"].astype(str)
    obj = obj[~obj["player_id"].str.startswith("NOID")]
    obj = obj.rename(columns={"rank": "objective_rank"})

    wheelo = load_wheelo_season()[["player_id", "wheelo_ev", "wheelo_rank"]]
    espn = load_espn_snapshot()
    espn_season = espn[espn["match_status"] == "resolved"][["player_id", "season_votes"]].rename(
        columns={"season_votes": "espn_ev"}
    ) if not espn.empty else pd.DataFrame(columns=["player_id", "espn_ev"])
    espn_season["espn_rank"] = espn_season["espn_ev"].rank(ascending=False, method="min") if not espn_season.empty else None

    betfair = load_betfair_season()
    betfair = betfair[["player_id", "betfair_ev", "betfair_rank"]] if not betfair.empty else pd.DataFrame(
        columns=["player_id", "betfair_ev", "betfair_rank"]
    )

    base = load_canonical_players()[["player_id", "player_name", "team_id"]]
    df = base.merge(prod, on="player_id", how="left") \
        .merge(obj, on="player_id", how="left") \
        .merge(wheelo, on="player_id", how="left") \
        .merge(espn_season, on="player_id", how="left") \
        .merge(betfair, on="player_id", how="left")

    df["our_midpoint"] = df[["production_ev", "objective_ev"]].mean(axis=1, skipna=True)
    df["our_internal_gap"] = (df["production_ev"] - df["objective_ev"]).abs()

    ext_cols = ["wheelo_ev", "espn_ev", "betfair_ev"]
    df["n_external_sources"] = df[ext_cols].notna().sum(axis=1)
    df["external_consensus_ev"] = df[ext_cols].mean(axis=1, skipna=True)
    df["external_min"] = df[ext_cols].min(axis=1, skipna=True)
    df["external_max"] = df[ext_cols].max(axis=1, skipna=True)
    df["external_range"] = df["external_max"] - df["external_min"]

    def _label(row):
        present = [c for c in ext_cols if pd.notna(row[c])]
        if not present:
            return "insufficient external data"
        if present == ["wheelo_ev"]:
            return "Wheelo benchmark"
        names = {"wheelo_ev": "Wheelo", "espn_ev": "ESPN", "betfair_ev": "Betfair"}
        return " + ".join(names[c] for c in present) + " benchmark"

    df["external_source_label"] = df.apply(_label, axis=1)
    df["our_external_gap"] = df["our_midpoint"] - df["external_consensus_ev"]

    # Rank on the external consensus (only meaningful where it exists).
    df["external_consensus_rank"] = df["external_consensus_ev"].rank(ascending=False, method="min")

    def _category(row):
        has_prod, has_obj = pd.notna(row["production_ev"]), pd.notna(row["objective_ev"])
        has_ext = row["n_external_sources"] > 0
        if not has_ext:
            return "INSUFFICIENT EXTERNAL DATA"
        prod_ext = has_prod and _agree(row["production_ev"], row["external_consensus_ev"])
        obj_ext = has_obj and _agree(row["objective_ev"], row["external_consensus_ev"])
        internal_agree = has_prod and has_obj and _agree(row["production_ev"], row["objective_ev"])
        if internal_agree and (prod_ext or obj_ext):
            return "STRONG CONVERGENCE"
        if internal_agree and not (prod_ext or obj_ext):
            return "OUR MODELS AGREE / EXTERNAL DIFFERS"
        if prod_ext and not obj_ext:
            return "PRODUCTION + EXTERNAL AGREE"
        if obj_ext and not prod_ext:
            return "OBJECTIVE + EXTERNAL AGREE"
        return "HIGH DISAGREEMENT"

    df["agreement_category"] = df.apply(_category, axis=1)
    return df.sort_values("production_rank", na_position="last").reset_index(drop=True)
