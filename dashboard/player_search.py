"""
Shared helpers for pages/28_Player_Search.py. Read-only over existing,
already-computed outputs -- nothing here fits a model, runs a simulation, or
recomputes a probability/EV. It only assembles values that other pages
(Finishing Order, To Poll a Vote, Model Agreement, Team Player Rankings)
already compute independently, into one per-player view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed
from dashboard import finishing_order as fo


def _normalise_player_id(x) -> str:
    """Same contract as dashboard.betting_opportunities._normalise_player_id
    / dashboard.finishing_order._normalise_player_id -- duplicated as a tiny
    pure function (not imported) to avoid a cross-module dependency for a
    two-line helper, kept behaviourally identical."""
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x)


@st.cache_data
def canonical_player_table() -> pd.DataFrame:
    """The full canonical 2026 player set (not only bookmaker-market
    players) -- the same dual-model comparison every other cross-model page
    already uses, so player identity here is guaranteed consistent with
    Model Agreement / H2H / Team Rankings."""
    return d.load_dual_model_comparison()


def season_p_any(player_id, sim_totals: np.ndarray, sim_players: pd.DataFrame) -> float | None:
    """P(season total votes >= 1) from real simulation draws -- the exact
    same computation src/betting/pricing.py::price_x_plus_votes(threshold=1)
    already uses for the "To Poll a Vote" market, re-applied here to the full
    player index rather than only bookmaker-listed players."""
    pid = _normalise_player_id(player_id)
    idx = sim_players.copy()
    idx["_pid"] = idx["player_id"].apply(_normalise_player_id)
    match = idx[idx["_pid"] == pid]
    if match.empty:
        return None
    col = match.index[0]
    return float((sim_totals[:, col] >= 1).mean())


@st.cache_data
def season_snapshot(player_id) -> dict:
    """Production / Objective / Wheelo season snapshot for one player:
    season EV, overall (global) rank, team rank, and season P(any vote)
    where reliably derivable. Wheelo NEVER gets a fabricated P(any)/P2/P1 --
    only EV + rank, since it has no persisted per-simulation draws (unlike
    Production/Objective) and no per-match P2/P1 (only a P3-equivalent)."""
    pid = _normalise_player_id(player_id)
    comparison = canonical_player_table()
    comparison = comparison.assign(_pid=comparison["player_id"].apply(_normalise_player_id))
    row = comparison[comparison["_pid"] == pid]
    if row.empty:
        return {}
    row = row.iloc[0]

    team_id = row["team_id"]
    team_rankings = d.dual_model_team_rankings(team_id) if pd.notna(team_id) else pd.DataFrame()
    team_row = team_rankings[team_rankings["player_id"].apply(_normalise_player_id) == pid] if not team_rankings.empty else pd.DataFrame()

    prod_totals, prod_players = fo.load_production_sim()
    obj_totals, obj_players = fo.load_objective_sim()
    wheelo = fo.load_wheelo()
    wheelo = wheelo.assign(_pid=wheelo["player_id"].apply(_normalise_player_id))
    wheelo_row = wheelo[wheelo["_pid"] == pid]
    wheelo_team = wheelo[wheelo["team_id"] == team_id] if pd.notna(team_id) and not wheelo.empty else pd.DataFrame()
    wheelo_team_rank = None
    if not wheelo_row.empty and not wheelo_team.empty and pd.notna(wheelo_row.iloc[0]["wheelo_ev"]):
        wheelo_team_rank = int(
            (wheelo_team["wheelo_ev"] > wheelo_row.iloc[0]["wheelo_ev"]).sum() + 1
        )

    return {
        "player_name": row["player_name"],
        "team_id": team_id,
        "production": {
            "ev": row["production_ev"] if row["in_production"] else None,
            "rank": int(row["production_rank"]) if row["in_production"] and pd.notna(row["production_rank"]) else None,
            "team_rank": int(team_row.iloc[0]["production_team_rank"]) if not team_row.empty and pd.notna(team_row.iloc[0].get("production_team_rank")) else None,
            "p_any": season_p_any(pid, prod_totals, prod_players),
        },
        "objective": {
            "ev": row["objective_ev"] if row["in_objective"] else None,
            "rank": int(row["objective_rank"]) if row["in_objective"] and pd.notna(row["objective_rank"]) else None,
            "team_rank": int(team_row.iloc[0]["objective_team_rank"]) if not team_row.empty and pd.notna(team_row.iloc[0].get("objective_team_rank")) else None,
            "p_any": season_p_any(pid, obj_totals, obj_players),
        },
        "wheelo": {
            "ev": float(wheelo_row.iloc[0]["wheelo_ev"]) if not wheelo_row.empty and pd.notna(wheelo_row.iloc[0]["wheelo_ev"]) else None,
            "rank": int(wheelo_row.iloc[0]["wheelo_rank"]) if not wheelo_row.empty and pd.notna(wheelo_row.iloc[0]["wheelo_rank"]) else None,
            "team_rank": wheelo_team_rank,
        },
        "model_gap": row["absolute_difference"] if pd.notna(row.get("absolute_difference")) else None,
    }


@st.cache_data
def season_trajectory(player_id) -> pd.DataFrame:
    """Round-by-round cumulative EV/Votes for Production, Objective, and
    Wheelo, from each source's own existing match-level file -- no new
    blending, no re-derivation. Returns one row per round with cumulative
    columns; a model missing that round (or entirely) contributes NaN, never
    a fabricated 0."""
    pid = _normalise_player_id(player_id)

    prod = d.load_match_probabilities()
    prod = prod[prod["player_id"].apply(_normalise_player_id) == pid][["round", "expected_votes"]]
    prod = prod.rename(columns={"expected_votes": "production_ev"}).groupby("round", as_index=False).sum()

    obj = d.load_objective_votes()
    obj = obj[obj["player_id"].apply(_normalise_player_id) == pid][["round", "expected_votes"]]
    obj = obj.rename(columns={"expected_votes": "objective_ev"}).groupby("round", as_index=False).sum()

    wheelo_ml = ed.load_wheelo_match_level()
    if not wheelo_ml.empty:
        wml = wheelo_ml[(wheelo_ml["player_id"].apply(_normalise_player_id) == pid) & (wheelo_ml["match_status"] == "resolved")]
        wml = wml[["round", "wheelo_ev"]].rename(columns={"wheelo_ev": "wheelo_votes"}).groupby("round", as_index=False).sum()
    else:
        wml = pd.DataFrame(columns=["round", "wheelo_votes"])

    merged = prod.merge(obj, on="round", how="outer").merge(wml, on="round", how="outer").sort_values("round")
    merged["production_cumulative"] = merged["production_ev"].cumsum()
    merged["objective_cumulative"] = merged["objective_ev"].cumsum()
    merged["wheelo_cumulative"] = merged["wheelo_votes"].cumsum()
    return merged.reset_index(drop=True)


def player_bookmaker_markets(player_id, player_name: str, opportunities_display: pd.DataFrame) -> pd.DataFrame:
    """This player's own resolved bookmaker markets from the already-priced,
    already-classified betting pipeline output (dashboard.betting_opportunities
    .load_opportunities() + .prepare_display()) -- no new pricing or
    classification, just a filter by identity."""
    if opportunities_display.empty:
        return opportunities_display
    pid = _normalise_player_id(player_id)
    by_id = opportunities_display["player_id"].apply(
        lambda x: _normalise_player_id(x) if pd.notna(x) else None
    ) == pid
    by_name = opportunities_display["player_name"] == player_name
    return opportunities_display[by_id | by_name]
