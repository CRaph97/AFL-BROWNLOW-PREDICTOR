"""
Read-only cross-model finishing-position analysis for the "Finishing Order"
page. Every probability here is computed directly from the already-generated
Production/Objective Monte Carlo draw arrays (data/processed/mc_totals_2026.npy,
mc_totals_objective_2026.npy) via src/models/order_scenarios.py's existing,
tested rank/order machinery -- this module never re-derives a probability from
a static EV rank, and never multiplies independent marginal probabilities.
Wheelo publishes point predictions only (no persisted per-simulation draw
array), so it is surfaced here as EV + season rank only -- never a fabricated
Top-N or exact-order probability.

Minimum-sample rule for exact-order probabilities: an order supported by
fewer than MIN_SUPPORTING_DRAWS simulations is reported as "very rare" rather
than a falsely-precise tiny percentage -- below ~10 supporting draws, the
relative sampling error of a proportion estimate (its width scales with
1/sqrt(count)) is too large for the number to mean much, e.g. 3/100000 could
easily have "really" been 1 or 8 under a re-run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.models import order_scenarios as os_

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEPLOYMENT = ROOT / "data" / "deployment"
REPORTS = ROOT / "reports"
EXTERNAL_PROCESSED = ROOT / "data" / "external" / "processed"

MIN_SUPPORTING_DRAWS = 10
MAX_N = 20


def _load_sim_array(local_filename: str, deployment_filename: str) -> np.ndarray:
    """Local-with-deployment-fallback for a Monte Carlo draw array, mirroring
    this project's established pattern (dashboard/data.py's `_resolve_path()`,
    dashboard/betting_data.py's Markets-repo fallback). `data/processed/` is
    gitignored (large local research artefacts, never meant to ship), so a
    fresh Streamlit Cloud clone never has these files -- without a fallback,
    every Finishing Order calculation raises FileNotFoundError there, which
    is exactly the live bug this fixes. The deployment copy is a lossless
    `np.savez_compressed` of the exact same int16 array (verified byte-for-
    byte identical after round-tripping) -- never a re-derived, rounded, or
    approximated value."""
    local_path = PROCESSED / local_filename
    if local_path.exists():
        return np.load(local_path)
    deployment_path = DEPLOYMENT / deployment_filename
    with np.load(deployment_path) as npz:
        return npz["totals"]


def _normalise_player_id(x) -> str:
    """Shared with dashboard/betting_opportunities.py's helper -- duplicated
    here as a tiny pure function (not imported) to avoid a betting-module
    dependency in this page; kept identical in behaviour: a numeric value
    (int/float/numeric-string) normalises to its plain integer string, a
    non-numeric string (e.g. a NOID2026_* synthetic id) passes through."""
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x)


@st.cache_data
def load_production_sim() -> tuple[np.ndarray, pd.DataFrame]:
    totals = _load_sim_array("mc_totals_2026.npy", "mc_totals_2026.npz")
    players = pd.read_csv(REPORTS / "2026_mc_player_index.csv")
    return totals, players


@st.cache_data
def load_objective_sim() -> tuple[np.ndarray, pd.DataFrame]:
    totals = _load_sim_array("mc_totals_objective_2026.npy", "mc_totals_objective_2026.npz")
    players = pd.read_csv(REPORTS / "2026_objective_mc_player_index.csv")
    return totals, players


@st.cache_data
def load_wheelo() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_PROCESSED / "external_overview.csv")
    return df[["player_id", "player_name", "team_id", "wheelo_ev", "wheelo_rank"]].copy()


@st.cache_data
def topn_table(n: int) -> pd.DataFrame:
    """One row per player present in EITHER model's simulation, with real
    P(final rank <= n) for Production/Objective (from actual simulated ranks,
    never inferred from a static leaderboard rank), plus Wheelo EV/rank
    (season-level point prediction only -- never a fabricated Top-N
    probability for Wheelo, per the brief)."""
    prod_totals, prod_players = load_production_sim()
    obj_totals, obj_players = load_objective_sim()
    wheelo = load_wheelo()

    prod = os_.contender_probabilities(prod_totals, prod_players, thresholds=(n,))
    prod = prod.rename(columns={f"prob_top{n}" if n > 1 else "prob_winner": "production_topn"})
    prod["player_id"] = prod["player_id"].apply(_normalise_player_id)
    prod["production_rank"] = prod["production_topn"].rank(ascending=False, method="min").astype(int)

    obj = os_.contender_probabilities(obj_totals, obj_players, thresholds=(n,))
    obj = obj.rename(columns={f"prob_top{n}" if n > 1 else "prob_winner": "objective_topn"})
    obj["player_id"] = obj["player_id"].apply(_normalise_player_id)
    obj["objective_rank"] = obj["objective_topn"].rank(ascending=False, method="min").astype(int)

    wheelo = wheelo.copy()
    wheelo["player_id"] = wheelo["player_id"].apply(_normalise_player_id)

    obj = obj.rename(columns={"mean_votes": "objective_mean_votes"})

    merged = prod[["player_id", "player_name", "team_id", "production_topn", "production_rank", "mean_votes"]].merge(
        obj[["player_id", "objective_topn", "objective_rank", "objective_mean_votes"]], on="player_id", how="outer",
    ).merge(
        wheelo[["player_id", "wheelo_ev", "wheelo_rank"]], on="player_id", how="left",
    )
    merged["player_name"] = merged["player_name"].fillna(
        merged["player_id"].map(dict(zip(obj["player_id"], obj["player_name"])))
    )
    merged["production_topn"] = merged["production_topn"].fillna(0.0)
    merged["objective_topn"] = merged["objective_topn"].fillna(0.0)
    merged["model_gap_pp"] = (merged["production_topn"] - merged["objective_topn"]).abs() * 100

    # Combined evidence for default sorting -- NOT a sort by either model's
    # own Top-N probability alone. Sorting by a single Top-N probability
    # directly repeats a bug already found and fixed once in this project
    # (order_scenarios.contender_probabilities's own docstring): for a small
    # N, most players tie at exactly 0.0 probability, and a stable sort then
    # orders that entire tied block by arbitrary row order, burying real
    # contenders. Summing both models' probabilities has far fewer exact
    # ties and keeps genuinely-in-contention players near the top by
    # construction.
    merged["combined_evidence"] = merged["production_topn"] + merged["objective_topn"]
    merged["agreement_label"] = merged["model_gap_pp"].apply(_agreement_label)
    return merged.sort_values("combined_evidence", ascending=False).reset_index(drop=True)


def _agreement_label(gap_pp: float) -> str:
    """Threshold documented here, not tuned against results: <=7.5pp mirrors
    the same internal-agreement gap already used as the High-Confidence
    threshold in src/betting/classification.py, for consistency across the
    app rather than a new, unrelated number."""
    if pd.isna(gap_pp):
        return "Insufficient data"
    if gap_pp <= 7.5:
        return "Models agree"
    if gap_pp <= 25:
        return "Models diverge"
    return "Models strongly diverge"


def exact_order_probability(player_ids_in_order: list[str]) -> dict:
    """Real joint-simulation-based probabilities for one specific ordered
    list of players (positions 1..k), for both models independently. Never
    a product of marginal probabilities -- computed by directly comparing
    each simulation's own actual finishing order (or actual rank set) to the
    requested one.

    Deliberately does NOT call order_scenarios.ranks_from_totals() for the
    "all k players in top k" probability. That function breaks ties in
    simulated vote totals via a double-argsort, while order_scenarios's own
    order_matix() (used here for the exact-order match) breaks ties via a
    single stable argsort -- two different, independently-reasonable
    conventions that normally never get compared to each other (each of
    order_scenarios.py's existing callers only uses one or the other), but
    combining them here produced a real, verified-impossible result on real
    data: an exact order's probability (31.03%) exceeding the "all 3 players
    finish in the top 3, any order" probability (28.68%) for the same
    players -- exact order is a subset of that broader event, so it can
    never be more likely. Fixed by deriving BOTH quantities from the exact
    same single stable sort, which makes the subset relationship hold by
    construction (verified: zero rows where exact-order-match is True but
    all-in-top-k is False, across all 100,000 draws)."""
    k = len(player_ids_in_order)
    result = {}
    for label, (totals, players) in (
        ("production", load_production_sim()),
        ("objective", load_objective_sim()),
    ):
        players = players.copy()
        players["player_id"] = players["player_id"].apply(_normalise_player_id)
        ids_norm = [_normalise_player_id(p) for p in player_ids_in_order]
        id_to_col = {pid: i for i, pid in enumerate(players["player_id"])}
        if not all(pid in id_to_col for pid in ids_norm):
            result[label] = {"exact_order_prob": None, "exact_order_supporting_draws": 0,
                              "n_sims": totals.shape[0], "all_in_topk_prob": None,
                              "unresolved": [p for p in ids_norm if p not in id_to_col]}
            continue
        cols = [id_to_col[pid] for pid in ids_norm]
        n_sims = totals.shape[0]

        # Both quantities derived from the SAME single stable sort (see
        # docstring above) so "all k in top k" can never be smaller than one
        # specific exact order's probability.
        stable_order = os_.order_matrix(totals, k)  # (n_sims, k) column indices, positions 1..k
        stable_rank_topk = np.argsort(np.argsort(-totals, axis=1, kind="stable"), axis=1) + 1

        col_arr = np.array(cols)
        exact_match = (stable_order == col_arr[None, :]).all(axis=1)
        support = int(exact_match.sum())

        all_in_topk = (stable_rank_topk[:, cols] <= k).all(axis=1).mean()

        result[label] = {
            "exact_order_prob": support / n_sims,
            "exact_order_supporting_draws": support,
            "n_sims": n_sims,
            "all_in_topk_prob": float(all_in_topk),
            "unresolved": [],
        }
    return result
