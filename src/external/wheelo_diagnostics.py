"""
Wheelo-specific disagreement diagnostics -- findings only, never used to
alter Production or Objective."""
from __future__ import annotations

import pandas as pd

from src.external.aggregate import _agree
from src.external.wheelo_source import load_wheelo_match_level, load_wheelo_season


def season_gap_table() -> pd.DataFrame:
    from src.external.aggregate import build_external_overview

    df = build_external_overview()
    df = df[df["wheelo_ev"].notna()].copy()
    df["prod_vs_wheelo_gap"] = df["production_ev"] - df["wheelo_ev"]
    df["obj_vs_wheelo_gap"] = df["objective_ev"] - df["wheelo_ev"]
    df["midpoint_vs_wheelo_gap"] = df["our_midpoint"] - df["wheelo_ev"]
    return df[[
        "player_id", "player_name", "team_id", "production_ev", "objective_ev",
        "our_midpoint", "wheelo_ev", "prod_vs_wheelo_gap", "obj_vs_wheelo_gap", "midpoint_vs_wheelo_gap",
    ]]


def match_level_gap_table() -> pd.DataFrame:
    """Largest individual-match gaps between Wheelo's per-match EV and
    Production's/Objective's per-match EV, plus the three-way convergence
    diagnostics the brief specifies."""
    wheelo = load_wheelo_match_level()
    wheelo = wheelo[wheelo["match_status"] == "resolved"][["round", "player_id", "wheelo_ev"]]

    prod = pd.read_csv("reports/2026_predicted_votes.csv")[["round", "player_id", "expected_votes"]].rename(
        columns={"expected_votes": "production_ev"}
    )
    prod["player_id"] = prod["player_id"].astype(str)

    obj = pd.read_csv("reports/2026_objective_votes.csv")[["round", "player_id", "expected_votes"]].rename(
        columns={"expected_votes": "objective_ev"}
    )
    obj["player_id"] = obj["player_id"].astype(str)
    obj = obj[~obj["player_id"].str.startswith("NOID")]

    merged = wheelo.merge(prod, on=["round", "player_id"], how="left").merge(
        obj, on=["round", "player_id"], how="left"
    )
    merged["prod_gap"] = (merged["production_ev"] - merged["wheelo_ev"]).abs()
    merged["obj_gap"] = (merged["objective_ev"] - merged["wheelo_ev"]).abs()

    def _diagnosis(row):
        p_ok, o_ok = pd.notna(row["production_ev"]), pd.notna(row["objective_ev"])
        prod_wheelo = p_ok and _agree(row["production_ev"], row["wheelo_ev"])
        obj_wheelo = o_ok and _agree(row["objective_ev"], row["wheelo_ev"])
        prod_obj = p_ok and o_ok and _agree(row["production_ev"], row["objective_ev"])
        if prod_wheelo and obj_wheelo:
            return "all-three convergence"
        if prod_obj and not (prod_wheelo or obj_wheelo):
            return "Production+Objective agree, Wheelo differs"
        if obj_wheelo and not prod_wheelo:
            return "Wheelo+Objective agree, Production differs"
        if prod_wheelo and not obj_wheelo:
            return "Wheelo+Production agree, Objective differs"
        return "no clear two-way agreement"

    merged["diagnosis"] = merged.apply(_diagnosis, axis=1)
    return merged
