"""
Exact-order finishing scenarios, computed generically from a (n_sims x
n_players) raw Monte Carlo totals array -- works identically for the
production ensemble's persisted draws (data/processed/mc_totals_2026.npy,
100,000 sims) and the standalone Objective Stats Model's draws
(data/processed/mc_totals_objective_2026.npy, 20,000 sims, see
run_2026_objective_montecarlo.py). This module does not fit, retrain, or
reweight anything -- it only ranks and counts existing simulation draws.

Tie-handling: ties in a simulated player's final integer vote total are
broken by the same argsort-of-argsort convention already used by both
run_2026_montecarlo.py's and run_2026_objective_montecarlo.py's summarise()
(numpy's default sort applied twice) -- not a new rule invented here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEPTHS = (5, 7, 10)
TOP_K_SCENARIOS = 10
POSITION_COLS = ["first", "second", "third", "fourth", "fifth",
                  "sixth", "seventh", "eighth", "ninth", "tenth"]


def ranks_from_totals(totals: np.ndarray) -> np.ndarray:
    """(n_sims, n_players) integer ranks, 1 = best. Ties broken by the
    argsort-of-argsort convention shared with both montecarlo scripts."""
    return (-totals).argsort(axis=1).argsort(axis=1) + 1


def order_matrix(totals: np.ndarray, depth: int) -> np.ndarray:
    """(n_sims, depth) matrix of player-column-indices in finishing order
    (index 0 = winner) for each simulated season."""
    return np.argsort(-totals, axis=1, kind="stable")[:, :depth]


def top_exact_orders(totals: np.ndarray, players: pd.DataFrame, depth: int,
                      top_k: int = TOP_K_SCENARIOS) -> pd.DataFrame:
    """Top-`top_k` most frequent exact finishing orders at the given depth,
    with each one's probability and the running cumulative probability."""
    n_sims = totals.shape[0]
    orders = order_matrix(totals, depth)
    names = players["player_name"].to_numpy()

    # Encode each simulated order as a tuple of player_ids for exact counting.
    ids = players["player_id"].to_numpy()
    order_ids = ids[orders]  # (n_sims, depth) of player_id
    tuples = [tuple(row) for row in order_ids]
    counts = pd.Series(tuples).value_counts()

    rows = []
    cum = 0.0
    id_to_name = dict(zip(players["player_id"], players["player_name"]))
    for rank, (tup, cnt) in enumerate(counts.head(top_k).items(), start=1):
        prob = cnt / n_sims
        cum += prob
        row = {"scenario_rank": rank, "probability": prob, "cumulative_probability": cum}
        for i, col in enumerate(POSITION_COLS):
            row[col] = id_to_name[tup[i]] if i < depth else None
        rows.append(row)
    return pd.DataFrame(rows)


def most_common_by_position(totals: np.ndarray, players: pd.DataFrame, depth: int) -> pd.DataFrame:
    orders = order_matrix(totals, depth)
    names = players["player_name"].to_numpy()
    rows = []
    for pos in range(depth):
        col = orders[:, pos]
        mode_idx = np.bincount(col, minlength=len(names)).argmax()
        share = (col == mode_idx).mean()
        rows.append({"position": pos + 1, "most_common_player": names[mode_idx], "probability": share})
    return pd.DataFrame(rows)


def contender_probabilities(totals: np.ndarray, players: pd.DataFrame,
                             thresholds=(1, 2, 3, 5, 7, 10)) -> pd.DataFrame:
    """Per-player Winner/TopN probabilities and mean simulated votes.

    Sorted by mean_votes (expected votes), NOT prob_winner: with a dominant
    favourite in the draws, prob_winner is ~0.0 for nearly every real
    contender (only the outright title favourites have any mass there), so
    sorting by it leaves hundreds of players tied at exactly 0.0 -- a stable
    sort then orders that entire tied block by whatever row order `players`
    happened to arrive in (e.g. team-alphabetical from the index CSV), not by
    actual contention. That previously caused genuine top-3-EV players (e.g.
    Marcus Bontempelli, near-zero prob_winner but ~93% prob_top10) to be
    pushed below a `.head(N)` cutoff in favour of fringe players who simply
    sat earlier in the arbitrary tie order. mean_votes has no such tie
    plateau and is already the ranking metric used everywhere else in this
    project (the leaderboard, the simulation summary), so sorting by it here
    keeps this table's ordering consistent with them.
    """
    ranks = ranks_from_totals(totals)
    out = players.copy()
    for t in thresholds:
        label = "prob_winner" if t == 1 else f"prob_top{t}"
        out[label] = (ranks <= t).mean(axis=0)
    out["mean_votes"] = totals.mean(axis=0)
    return out.sort_values("mean_votes", ascending=False).reset_index(drop=True)


def build_order_scenarios_csv(model_totals: dict[str, tuple[np.ndarray, pd.DataFrame]],
                               depths=DEPTHS, top_k: int = TOP_K_SCENARIOS) -> pd.DataFrame:
    """model_totals: {model_label: (totals_array, players_df)}. Returns the
    long-format table matching reports/2026_order_scenarios.csv's schema."""
    frames = []
    for model_label, (totals, players) in model_totals.items():
        for depth in depths:
            top = top_exact_orders(totals, players, depth, top_k=top_k)
            top.insert(0, "order_depth", depth)
            top.insert(0, "model", model_label)
            frames.append(top.drop(columns=["cumulative_probability"]))
    cols = ["model", "order_depth", "scenario_rank"] + POSITION_COLS + ["probability"]
    return pd.concat(frames, ignore_index=True)[cols]
