"""
Regression tests for the contender-probability player-mapping bug.

Root cause (found during a Phase-5 QA pass): contender_probabilities() sorted
its output by prob_winner, which is ~0.0 for nearly every real contender (only
1-3 players ever have positive mass on outright winning in a 100k/20k-draw
simulation). A stable sort left hundreds of players tied at exactly 0.0,
ordered by whatever arbitrary row order they arrived in from the player-index
CSV -- so pages/14_Order_Scenarios.py's `.head(15)` after that sort could
(and did) bury a genuine top-3-EV player (Marcus Bontempelli) behind fringe
players who simply sat earlier in that arbitrary tie order. The Monte Carlo
array <-> player identity mapping itself was never wrong -- these tests
assert both things: the mapping is exact, and the sort/selection is sane.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.order_scenarios import contender_probabilities, order_matrix

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DEPLOYMENT = ROOT / "data" / "deployment"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "mc_totals_2026.npy").exists(),
    reason="local-only Monte Carlo artefacts not present in this environment",
)


def _load_production():
    totals = np.load(PROCESSED / "mc_totals_2026.npy")
    players = pd.read_csv(REPORTS / "2026_mc_player_index.csv")
    return totals, players


def _load_objective():
    totals = np.load(PROCESSED / "mc_totals_objective_2026.npy")
    players = pd.read_csv(REPORTS / "2026_objective_mc_player_index.csv")
    return totals, players


def test_production_mc_mean_matches_simulation_summary():
    """mean(mc_totals[:, i]) must match reports/2026_simulation_summary.csv's
    sim_mean_votes for that same player_id -- the two are supposed to be the
    exact same number (the simulation summary IS a summary of these draws),
    so any gap at all indicates a column<->identity mapping bug."""
    totals, players = _load_production()
    sim = pd.read_csv(REPORTS / "2026_simulation_summary.csv")
    players = players.assign(mc_mean=totals.mean(axis=0))
    merged = players.merge(sim[["player_id", "sim_mean_votes"]], on="player_id", how="inner")
    assert len(merged) > 500  # sanity: the join actually matched most players
    assert (merged["mc_mean"] - merged["sim_mean_votes"]).abs().max() < 1e-6


def test_objective_mc_mean_matches_simulation_summary():
    totals, players = _load_objective()
    sim = pd.read_csv(REPORTS / "2026_objective_simulation_summary.csv")
    players = players.assign(mc_mean=totals.mean(axis=0))
    merged = players.merge(sim[["player_id", "sim_mean_votes"]], on="player_id", how="inner")
    assert len(merged) > 700
    assert (merged["mc_mean"] - merged["sim_mean_votes"]).abs().max() < 1e-6


def test_bontempelli_present_and_ranked_in_production_contenders():
    totals, players = _load_production()
    out = contender_probabilities(totals, players)
    row = out[out["player_name"] == "Marcus Bontempelli"]
    assert len(row) == 1
    # He's a real top-3-EV player -- must land inside a top-15 view, and his
    # mean_votes must be a plausible Brownlow contender total, not a stray
    # near-zero value from a mismapped column.
    rank_position = out.index[out["player_name"] == "Marcus Bontempelli"][0]
    assert rank_position < 15
    assert row["mean_votes"].iloc[0] > 15


def test_contender_table_sorted_by_mean_votes_not_prob_winner():
    """Guards the actual bug: the output must be monotonically non-increasing
    in mean_votes. (It is not required to be monotonic in prob_winner, since
    prob_winner is ~0 for most real contenders and isn't the sort key.)"""
    totals, players = _load_production()
    out = contender_probabilities(totals, players)
    assert out["mean_votes"].is_monotonic_decreasing


def test_no_systematic_mapping_mismatch_across_full_player_set():
    """Every player_id's simulated mean must match its leaderboard-adjacent
    simulation-summary mean, checked across the FULL player set, not just a
    named few -- a real column-shift bug would fail this for most players,
    not just one."""
    totals, players = _load_production()
    sim = pd.read_csv(REPORTS / "2026_simulation_summary.csv")
    players = players.assign(mc_mean=totals.mean(axis=0))
    merged = players.merge(sim[["player_id", "sim_mean_votes"]], on="player_id", how="inner")
    mismatches = merged[(merged["mc_mean"] - merged["sim_mean_votes"]).abs() > 1e-6]
    assert mismatches.empty, f"{len(mismatches)} players have a mismatched MC column mapping"


def test_deployment_csvs_match_fresh_computation():
    """The bundled deployment CSVs must be byte-equivalent (post-fix) to what
    contender_probabilities() computes right now from the real arrays -- if
    someone regenerates the arrays without refreshing the deployment snapshot,
    this will catch the drift."""
    for csv_name, (totals, players) in [
        ("contender_probabilities_production.csv", _load_production()),
        ("contender_probabilities_objective.csv", _load_objective()),
    ]:
        fresh = contender_probabilities(totals, players)
        bundled = pd.read_csv(DEPLOYMENT / csv_name)
        pd.testing.assert_frame_equal(
            fresh.reset_index(drop=True), bundled.reset_index(drop=True), check_exact=False, rtol=1e-6,
        )


def test_order_scenario_player_names_align_with_simulation_columns():
    """order_matrix() indexes into `players` by column position -- confirm
    that for a small subset, the column position it returns as position-0
    (the simulated winner) genuinely corresponds to that player's real
    highest-total simulations, i.e. names line up with columns, not just with
    row-count."""
    totals, players = _load_production()
    orders = order_matrix(totals, depth=1)
    winner_col_per_sim = orders[:, 0]
    # Nick Daicos wins the outright simulated total in the vast majority of
    # sims (he has ~99% prob_winner) -- his player_id's column index must be
    # the overwhelming mode of the simulated winner column.
    daicos_col = players.index[players["player_name"] == "Nick Daicos"][0]
    mode_col = np.bincount(winner_col_per_sim, minlength=totals.shape[1]).argmax()
    assert mode_col == daicos_col
