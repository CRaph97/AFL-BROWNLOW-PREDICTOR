"""
Lightweight season Monte Carlo for the standalone 2026 Objective Stats Model.

Mirrors run_2026_montecarlo.py's Gumbel-max sampling of the production
ensemble, but simpler: the Objective model has ONE utility per player per
match (`objective_score`, already the exact same quantity its own p3/p2/p1/p0
columns were derived from via the shared Plackett-Luce marginalization in
src/models/plackett_luce.py), not a 3-scenario mixture -- so each simulated
match just adds Gumbel noise to objective_score and takes the top 3 (Yellott's
theorem: this reproduces the model's own coherent p3/p2/p1 exactly, in the
simulation limit). Ties in a simulated player's final integer vote total are
broken by the same argsort-of-argsort convention already used in
run_2026_montecarlo.py's summarise() -- not a new rule.

Deliberately modest N_SIMS (this is an auxiliary, exploratory model, not the
production forecast): fast enough to regenerate on demand, still stable
enough for the season summary stats, the order-scenario page, and the
betting-comparison page's objective-side market probabilities.

This module NEVER touches the production model or its files, and never
retrains or reweights the Objective model itself -- it only samples from its
already-computed, frozen match-level scores.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

N_SIMS = 20_000
RNG_SEED = 20260917
THRESHOLDS = [10, 15, 20, 25, 30, 35]

# Must match objective_stats_model.py's compute_objective_scores() UTILITY_TEMPERATURE
# exactly: the marginalisation that produced the CSV's own p3/p2/p1/p0 columns was run
# on objective_score/15.0 (a display-vs-utility scale split -- see that function's
# docstring), not on raw objective_score. Verified empirically: this script's simulated
# mean season votes must match reports/2026_objective_leaderboard.csv's objective_ev to
# within Monte Carlo noise (see tests/test_objective_montecarlo.py) -- using the wrong
# temperature (e.g. 1.0, i.e. raw objective_score) silently over-concentrates the
# simulated distribution and was caught exactly this way during development.
UTILITY_TEMPERATURE = 15.0


def simulate_season(scores: pd.DataFrame, n_sims: int = N_SIMS, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    matches = scores.groupby("match_id")
    n_matches = scores["match_id"].nunique()

    all_players = scores[["player_id", "player_name", "team_id"]].drop_duplicates().reset_index(drop=True)
    player_index = {pid: i for i, pid in enumerate(all_players["player_id"])}
    n_players = len(all_players)

    totals = np.zeros((n_sims, n_players), dtype=np.int16)

    for match_id, g in matches:
        u = g["objective_score"].to_numpy(dtype=float) / UTILITY_TEMPERATURE
        idx = np.array([player_index[p] for p in g["player_id"]])

        gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, len(u)))
        scores_sim = u[None, :] + gumbel
        top3_local = np.argpartition(-scores_sim, kth=2, axis=1)[:, :3]
        row_scores = np.take_along_axis(scores_sim, top3_local, axis=1)
        order = np.argsort(-row_scores, axis=1)
        top3_sorted = np.take_along_axis(top3_local, order, axis=1)

        winners_3 = idx[top3_sorted[:, 0]]
        winners_2 = idx[top3_sorted[:, 1]]
        winners_1 = idx[top3_sorted[:, 2]]

        totals[np.arange(n_sims), winners_3] += 3
        totals[np.arange(n_sims), winners_2] += 2
        totals[np.arange(n_sims), winners_1] += 1

    return totals, all_players, n_matches


def summarise(totals: np.ndarray, players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ranks = (-totals).argsort(axis=1).argsort(axis=1) + 1

    for j, row in players.iterrows():
        col = totals[:, j]
        rank_col = ranks[:, j]
        rows.append({
            "player_id": row["player_id"], "player_name": row["player_name"], "team_id": row["team_id"],
            "sim_mean_votes": col.mean(), "sim_median_votes": np.median(col),
            "sim_mode_votes": np.bincount(col).argmax(),
            "sim_p10": np.percentile(col, 10), "sim_p90": np.percentile(col, 90),
            "sim_p2_5": np.percentile(col, 2.5), "sim_p97_5": np.percentile(col, 97.5),
            **{f"prob_ge_{t}": (col >= t).mean() for t in THRESHOLDS},
            "mean_rank": rank_col.mean(), "median_rank": np.median(rank_col),
            "prob_rank_1": (rank_col == 1).mean(), "prob_top3_rank": (rank_col <= 3).mean(),
            "prob_top10_rank": (rank_col <= 10).mean(),
        })
    return pd.DataFrame(rows).sort_values("sim_mean_votes", ascending=False)


def run():
    scores = pd.read_csv(REPORTS_DIR / "2026_objective_match_scores.csv")
    n_matches_expected = scores["match_id"].nunique()
    print(f"Running {N_SIMS:,} lightweight Monte Carlo sims over {n_matches_expected} matches "
          f"(single-scenario Gumbel-max on the Objective model's own utility)...")

    totals, players, n_matches = simulate_season(scores, N_SIMS)

    summary = summarise(totals, players)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "2026_objective_simulation_summary.csv", index=False)

    total_votes_per_sim = totals.sum(axis=1)
    expected_total = 6 * n_matches
    max_dev = np.abs(total_votes_per_sim - expected_total).max()
    print(f"QUALITY CHECK: expected total votes/season = {expected_total}, "
          f"observed range = [{total_votes_per_sim.min()}, {total_votes_per_sim.max()}], "
          f"max deviation = {max_dev} (must be 0)")
    assert max_dev == 0, "Objective Monte Carlo simulation violated the 6-votes-per-match constraint."

    np.save(PROCESSED_DIR / "mc_totals_objective_2026.npy", totals)
    players.to_csv(REPORTS_DIR / "2026_objective_mc_player_index.csv", index=False)
    print(f"Wrote reports/2026_objective_simulation_summary.csv, "
          f"reports/2026_objective_mc_player_index.csv, "
          f"data/processed/mc_totals_objective_2026.npy ({len(players)} players)")
    return summary


if __name__ == "__main__":
    run()
