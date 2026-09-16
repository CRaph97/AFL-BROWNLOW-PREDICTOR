"""
Phase 5, section 8: season Monte Carlo simulation from the final ensemble.

The final ensemble is a LINEAR OPINION POOL over Scenario A/B/C's probabilities
(see build_2026_ensemble.py's module docstring for why utility-space blending
was wrong and replaced). A probability mixture of several Plackett-Luce models
is not itself a single Plackett-Luce process, so it cannot be sampled by
Gumbel-max on one blended utility. Instead it is sampled as a genuine
GENERATIVE MIXTURE, which exactly reproduces the mixture's marginals in the
simulation limit: for each (match, simulation), first draw WHICH scenario's
model applies according to the ensemble weights (0.45 / 0.20 / 0.35 for
A / B / C), then draw that match's full 3-2-1 ranking via the Gumbel-max trick
(Yellott's theorem) applied to THAT scenario's own fitted utility. Every
simulated match still allocates exactly one 3, one 2 and one 1 vote by
construction, regardless of which scenario was drawn.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

N_SIMS = 100_000
RNG_SEED = 20260917
THRESHOLDS = [10, 15, 20, 25, 30, 35]
SCENARIO_WEIGHTS = np.array([0.45, 0.20, 0.35])  # A, B, C -- must match build_2026_ensemble.ENSEMBLE_WEIGHTS


def simulate_season(preds: pd.DataFrame, n_sims: int = N_SIMS, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    matches = preds.groupby("match_id")
    n_matches = preds["match_id"].nunique()

    all_players = preds[["player_id", "player_name", "team_id"]].drop_duplicates().reset_index(drop=True)
    player_index = {pid: i for i, pid in enumerate(all_players["player_id"])}
    n_players = len(all_players)

    totals = np.zeros((n_sims, n_players), dtype=np.int16)

    for i, (match_id, g) in enumerate(matches):
        u_stack = g[["utility_raw_A", "utility_raw_B", "utility_raw_C"]].to_numpy(dtype=float).T  # (3, n_p)
        idx = np.array([player_index[p] for p in g["player_id"]])
        n_p = len(idx)

        scenario_choice = rng.choice(3, size=n_sims, p=SCENARIO_WEIGHTS)
        u_sim = u_stack[scenario_choice]  # (n_sims, n_p) -- each sim's row is its drawn scenario's utility

        gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, n_p))
        scores = u_sim + gumbel
        top3_local = np.argpartition(-scores, kth=2, axis=1)[:, :3]
        row_scores = np.take_along_axis(scores, top3_local, axis=1)
        order = np.argsort(-row_scores, axis=1)
        top3_sorted = np.take_along_axis(top3_local, order, axis=1)

        winners_3 = idx[top3_sorted[:, 0]]
        winners_2 = idx[top3_sorted[:, 1]]
        winners_1 = idx[top3_sorted[:, 2]]

        totals[np.arange(n_sims), winners_3] += 3
        totals[np.arange(n_sims), winners_2] += 2
        totals[np.arange(n_sims), winners_1] += 1

        if (i + 1) % 25 == 0 or (i + 1) == n_matches:
            print(f"  simulated {i+1}/{n_matches} matches")

    return totals, all_players


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
    common_u = pd.read_parquet(PROCESSED_DIR / "common_scenario_utilities_2026.parquet")
    all_scenarios = pd.read_parquet(PROCESSED_DIR / "all_2026_scenarios_and_ensemble.parquet")
    names = all_scenarios[all_scenarios["scenario"] == "FINAL_ENSEMBLE"][
        ["match_id", "player_id", "player_name", "team_id"]].drop_duplicates()
    ensemble = common_u.merge(names, on=["match_id", "player_id"], how="left")

    n_matches = ensemble["match_id"].nunique()
    print(f"Running {N_SIMS:,} Monte Carlo simulations over {n_matches} 2026 matches "
          f"(generative mixture: A/B/C drawn per-match-per-sim with weights {SCENARIO_WEIGHTS.tolist()})...")
    totals, players = simulate_season(ensemble, N_SIMS)

    summary = summarise(totals, players)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "2026_simulation_summary.csv", index=False)
    print(f"Wrote reports/2026_simulation_summary.csv ({len(summary)} players)")

    total_votes_per_sim = totals.sum(axis=1)
    expected_total = 6 * n_matches
    max_dev = np.abs(total_votes_per_sim - expected_total).max()
    print(f"QUALITY CHECK: expected total votes/season = {expected_total}, "
          f"observed range = [{total_votes_per_sim.min()}, {total_votes_per_sim.max()}], "
          f"max deviation = {max_dev} (must be 0)")
    assert max_dev == 0, "Monte Carlo simulation violated the 6-votes-per-match constraint somewhere."

    np.save(PROCESSED_DIR / "mc_totals_2026.npy", totals)
    players.to_csv(REPORTS_DIR / "2026_mc_player_index.csv", index=False)
    print("Saved raw simulation totals array -> data/processed/mc_totals_2026.npy")
    return summary


if __name__ == "__main__":
    run()
