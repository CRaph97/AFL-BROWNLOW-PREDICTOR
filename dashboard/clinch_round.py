"""
Clinch Round -- when the Brownlow becomes mathematically decided, per
Production and Objective's own real season simulations, plus Wheelo's
deterministic expected trajectory.

METHODOLOGY (read before touching this file):

This module does NOT run a new simulation. It REPLAYS the exact, already-
validated Monte Carlo computation in src/models/run_2026_montecarlo.py and
src/models/run_2026_objective_montecarlo.py -- same seed (20260917), same
per-match Gumbel-max draws, same `groupby("match_id")` iteration order (which
sorts LEXICOGRAPHICALLY on the match_id string, e.g. "...R10..." before
"...R2..." -- the same string-sort defect this project has already found and
fixed twice elsewhere for round labels). The replay is required only to keep
information the original scripts discarded (each match's REAL round tag) --
it produces bit-identical per-simulation vote totals to the existing
data/processed/mc_totals_2026.npy / mc_totals_objective_2026.npy arrays,
verified in tests/test_clinch_round_page.py by checking the replay's
own final cumulative distribution against reports/2026_simulation_summary.csv
/ reports/2026_objective_simulation_summary.csv (mean/percentiles, which are
sensitive to any drift in the underlying per-sim totals).

Because vote increments commute, "cumulative votes through real round R" can
be computed by binning each match's increment into `round_bucket[real_round]`
during the ORIGINAL (RNG-order-preserving) match loop, then summing buckets
0..R afterwards -- this requires zero new randomness and is mathematically
exact regardless of the original loop's (lexicographic) order.

MATHEMATICAL CLINCH: at round R, the current leader (by cumulative votes) is
clinched iff leader_votes > max over every other player of
(their cumulative votes + 3 * their team's real remaining matches after R).
Strict inequality: a rival who could exactly TIE is not counted as blocking a
clinch (a tie is not a clinch). This is monotonic once true (remaining-game
ceilings can only shrink and the leader's votes can only grow), so "first
round satisfying this" is well-defined per simulation.

PROJECTED WINNER POINT (a real methodological judgement call, stated plainly
here and in the page): for the real leaderboard favourite (highest season EV
in reports/2026_leaderboard.csv / 2026_objective_leaderboard.csv), this is
P(that player is the simulation's EVENTUAL winner | they are CURRENTLY the
leader at round R) -- i.e. "when they're leading at this point, how often do
they go on to actually win". This is a genuine conditional probability
computed directly from the replay, not an approximation -- but it is one
defensible formalisation of "projected winner point" among several possible
ones, and is deliberately different from (and never conflated with) the
mathematical clinch test above.

WHEELO: deterministic only. Wheelo has no persisted season simulation, so its
"expected-trajectory clinch" is a single trajectory (cumulative real Wheelo
EV vs. a 3-per-remaining-game ceiling for the nearest rival), never a
probability, never plotted on the same axis as Production/Objective's
genuine simulation-based probabilities.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import external_data as ed

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PROD_SEED = 20260917
OBJ_SEED = 20260917
OBJ_UTILITY_TEMPERATURE = 15.0  # must match src/models/run_2026_objective_montecarlo.py exactly

_MATCH_ID_ROUND_RE = re.compile(r"_R(\d+)_")


def _round_from_match_id(match_id: str) -> int:
    m = _MATCH_ID_ROUND_RE.search(match_id)
    if not m:
        raise ValueError(f"could not extract round from match_id: {match_id!r}")
    return int(m.group(1))


@st.cache_data
def team_remaining_matches() -> dict:
    """{(team_id, round): count of that team's REAL remaining matches after
    `round`}, from the real 2026 fixture (dashboard.data.match_meta_table()),
    accounting for byes -- never an assumed constant games-remaining count."""
    meta = d.match_meta_table().copy()
    meta["round"] = meta["round"].astype(int)
    max_round = int(meta["round"].max())

    team_rounds: dict[str, list[int]] = {}
    for _, row in meta.iterrows():
        for team in (row["team_a"], row["team_b"]):
            team_rounds.setdefault(team, []).append(row["round"])

    remaining = {}
    for team, rounds_played in team_rounds.items():
        rounds_played = sorted(rounds_played)
        for r in range(0, max_round + 1):
            remaining[(team, r)] = sum(1 for rp in rounds_played if rp > r)
    return remaining


def _player_remaining_vector(players: pd.DataFrame, remaining_lookup: dict, round_: int) -> np.ndarray:
    return np.array([remaining_lookup.get((t, round_), 0) for t in players["team_id"]], dtype=np.int32)


def _replay(
    match_scores_source: pd.DataFrame,
    utility_cols: list[str] | None,
    single_utility_col: str | None,
    seed: int,
    n_sims: int,
    scenario_weights: np.ndarray | None,
    utility_temperature: float,
) -> dict:
    """Shared replay engine for both Production (3-scenario mixture) and
    Objective (single utility) -- mirrors simulate_season() in the two real
    run_*montecarlo.py scripts exactly, bucketing each match's increment by
    its REAL round instead of only accumulating into one final total.

    MEMORY: matches must be visited in `groupby("match_id")`'s lexicographic
    order for RNG-draw reproducibility (see module docstring), but that order
    scatters rounds arbitrarily (round 2's matches can be drawn after round
    19's) -- so a naive per-round (n_sims x n_players) bucket must keep EVERY
    round's bucket alive simultaneously (measured: ~4.3GB peak RSS for
    Production, enough to be killed by Streamlit Cloud's memory ceiling with
    no Python traceback, just a generic "Oh no"). Fixed with a two-pass split:
    pass 1 preserves the exact RNG call sequence/order but stores only each
    match's compact (n_sims,) winner-index triple (int16, no player
    dimension) instead of a dense per-round player-width array; pass 2 (pure
    integer addition, no RNG, so order-independent) folds those triples into
    ONE reused (n_sims x n_players) array in real round order, computing the
    same per-round summary stats as before. Mathematically identical --
    addition is commutative/associative and no dtype narrower than the
    original was introduced for anything retained -- verified bit-identical
    against the pre-existing mc_totals*.npy arrays in
    tests/test_clinch_round_page.py::TestBitIdenticalReplay."""
    rng = np.random.default_rng(seed)
    matches = match_scores_source.groupby("match_id")  # same default (sorted/lexicographic) order as the real scripts

    all_players = match_scores_source[["player_id", "player_name", "team_id"]].drop_duplicates().reset_index(drop=True)
    player_index = {pid: i for i, pid in enumerate(all_players["player_id"])}
    n_players = len(all_players)

    max_round = max(_round_from_match_id(mid) for mid in match_scores_source["match_id"].unique())

    # Pass 1: RNG-order-preserving. Only (n_sims,) index triples per match are
    # kept -- no (n_sims x n_players) allocation here at all.
    by_round: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {r: [] for r in range(max_round + 1)}
    for match_id, g in matches:
        real_round = _round_from_match_id(match_id)
        idx = np.array([player_index[p] for p in g["player_id"]])
        n_p = len(idx)

        if scenario_weights is not None:
            u_stack = g[utility_cols].to_numpy(dtype=float).T  # (3, n_p)
            scenario_choice = rng.choice(3, size=n_sims, p=scenario_weights)
            u_sim = u_stack[scenario_choice]
            gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, n_p))
            scores_sim = u_sim + gumbel
        else:
            u = g[single_utility_col].to_numpy(dtype=float) / utility_temperature
            gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, n_p))
            scores_sim = u[None, :] + gumbel

        top3_local = np.argpartition(-scores_sim, kth=2, axis=1)[:, :3]
        row_scores = np.take_along_axis(scores_sim, top3_local, axis=1)
        order = np.argsort(-row_scores, axis=1)
        top3_sorted = np.take_along_axis(top3_local, order, axis=1)

        w3, w2, w1 = idx[top3_sorted[:, 0]], idx[top3_sorted[:, 1]], idx[top3_sorted[:, 2]]
        by_round[real_round].append((w3.astype(np.int16), w2.astype(np.int16), w1.astype(np.int16)))

    remaining_lookup = team_remaining_matches()

    cumulative = np.zeros((n_sims, n_players), dtype=np.int32)
    leader_idx_by_round = np.zeros((max_round + 1, n_sims), dtype=np.int32)
    leader_votes_by_round = np.zeros((max_round + 1, n_sims), dtype=np.int32)
    clinched_by_round = np.zeros((max_round + 1, n_sims), dtype=bool)
    # Mean cumulative per player per round -- small (n_rounds x n_players),
    # used for the Round Explorer's illustrative trajectory, never for the
    # per-simulation clinch/probability statistics above.
    mean_cumulative_by_round = np.zeros((max_round + 1, n_players), dtype=np.float64)

    # Pass 2: pure arithmetic, real round order, one (n_sims x n_players)
    # array reused throughout instead of one per round.
    for r in range(max_round + 1):
        sim_arange = np.arange(n_sims)
        for w3, w2, w1 in by_round.pop(r):
            cumulative[sim_arange, w3] += 3
            cumulative[sim_arange, w2] += 2
            cumulative[sim_arange, w1] += 1

        remaining_vec = _player_remaining_vector(all_players, remaining_lookup, r)
        leader_idx = cumulative.argmax(axis=1)
        leader_votes = cumulative[np.arange(n_sims), leader_idx]

        ceiling = cumulative + 3 * remaining_vec[None, :]
        ceiling_masked = ceiling.copy()
        ceiling_masked[np.arange(n_sims), leader_idx] = -1  # exclude leader's own ceiling from the rival max
        max_rival_ceiling = ceiling_masked.max(axis=1)

        leader_idx_by_round[r] = leader_idx
        leader_votes_by_round[r] = leader_votes
        clinched_by_round[r] = leader_votes > max_rival_ceiling  # strict: an exact tie does NOT clinch
        mean_cumulative_by_round[r] = cumulative.mean(axis=0)

    final_winner_idx = leader_idx_by_round[max_round]

    # First round each simulation is clinched (mathematically monotonic --
    # see module docstring -- so this is well-defined); NaN if never clinched
    # (a genuine season-long unresolved tie at the final round).
    clinch_round_per_sim = np.full(n_sims, np.nan)
    any_clinched = clinched_by_round.any(axis=0)
    first_clinch = clinched_by_round.argmax(axis=0)  # first True index; meaningless where any_clinched is False
    clinch_round_per_sim[any_clinched] = first_clinch[any_clinched]

    return {
        "players": all_players,
        "max_round": max_round,
        "leader_idx_by_round": leader_idx_by_round,
        "leader_votes_by_round": leader_votes_by_round,
        "clinched_by_round": clinched_by_round,
        "clinch_round_per_sim": clinch_round_per_sim,
        "final_winner_idx": final_winner_idx,
        "mean_cumulative_by_round": mean_cumulative_by_round,
        "final_cumulative": cumulative,  # small (n_sims x n_players) int32; kept only for the
                                          # bit-identical-replay correctness proof against mc_totals*.npy
        "n_sims": n_sims,
    }


# cache_resource (not cache_data): the returned dict carries a ~250MB
# final_cumulative array; cache_data deep-copies its return value on EVERY
# cache hit (not just the first computation) to guard against mutation --
# measured at ~0.4s per access, the dominant cost of every rerun/slider move.
# cache_resource returns the same shared object with no copy, which is safe
# here since nothing downstream ever mutates these arrays in place.
@st.cache_resource
def replay_production() -> dict:
    # common_scenario_utilities_2026.parquet is small (~270KB) and genuinely
    # required (it IS the real per-match utility input the replay draws
    # from) -- data/processed/ is gitignored, so a fresh Streamlit Cloud
    # clone never has it; falls back to the bundled deployment copy via this
    # project's established local-with-deployment-fallback pattern.
    common_u = pd.read_parquet(d._resolve_path(
        PROCESSED / "common_scenario_utilities_2026.parquet",
        "common_scenario_utilities_2026.parquet",
    ))
    # Player name/team lookup: previously read the full 4.3MB
    # all_2026_scenarios_and_ensemble.parquet (also gitignored) just to pull
    # out player_id -> (player_name, team_id) -- reports/2026_mc_player_index.csv
    # (already git-tracked, ~18KB) contains the exact same 577 (player_id,
    # player_name, team_id) triples (verified: zero mismatches against the
    # parquet's own FINAL_ENSEMBLE rows), so this avoids a second deployment
    # dependency entirely rather than bundling another large file.
    names = pd.read_csv(REPORTS / "2026_mc_player_index.csv")
    names["player_id"] = names["player_id"].astype(str)
    common_u = common_u.copy()
    common_u["player_id"] = common_u["player_id"].astype(str)
    ensemble = common_u.merge(names, on="player_id", how="left")
    return _replay(
        ensemble,
        utility_cols=["utility_raw_A", "utility_raw_B", "utility_raw_C"],
        single_utility_col=None,
        seed=PROD_SEED,
        n_sims=100_000,
        scenario_weights=np.array([0.45, 0.20, 0.35]),
        utility_temperature=1.0,
    )


# See replay_production() for why cache_resource, not cache_data.
@st.cache_resource
def replay_objective() -> dict:
    scores = pd.read_csv(REPORTS / "2026_objective_match_scores.csv")
    return _replay(
        scores,
        utility_cols=None,
        single_utility_col="objective_score",
        seed=OBJ_SEED,
        n_sims=20_000,
        scenario_weights=None,
        utility_temperature=OBJ_UTILITY_TEMPERATURE,
    )


def conditional_win_prob_by_round(replay: dict, target_player_id) -> pd.Series:
    """P(target player is the EVENTUAL winner | they are the CURRENT leader
    at round R) for every round -- the "projected winner point" metric, see
    module docstring for why this specific definition was chosen. NaN for a
    round where the target player is never the current leader in any
    simulation (undefined, not zero)."""
    players = replay["players"]
    match = players.index[players["player_id"] == target_player_id]
    if len(match) == 0:
        return pd.Series(dtype=float)
    target_idx = match[0]
    is_final_winner = replay["final_winner_idx"] == target_idx

    out = {}
    for r in range(replay["max_round"] + 1):
        is_leader = replay["leader_idx_by_round"][r] == target_idx
        n_leader = is_leader.sum()
        out[r] = float((is_leader & is_final_winner).sum() / n_leader) if n_leader > 0 else np.nan
    return pd.Series(out)


def conditional_win_prob_at_round(replay: dict, target_player_id, round_: int) -> float:
    """Single-round value of conditional_win_prob_by_round(), computing only
    the requested round instead of the full per-round Series -- the Round
    Explorer only ever needs the currently-selected round, so looping over
    every round just to discard 24 of the 25 results was pure waste. Same
    formula, same result as conditional_win_prob_by_round(replay, pid)[round_]."""
    players = replay["players"]
    match = players.index[players["player_id"] == target_player_id]
    if len(match) == 0:
        return float("nan")
    target_idx = match[0]
    is_leader = replay["leader_idx_by_round"][round_] == target_idx
    n_leader = is_leader.sum()
    if n_leader == 0:
        return float("nan")
    is_final_winner = replay["final_winner_idx"] == target_idx
    return float((is_leader & is_final_winner).sum() / n_leader)


def earliest_round_reaching(prob_by_round: pd.Series, threshold: float) -> int | None:
    hits = prob_by_round[prob_by_round >= threshold]
    return int(hits.index.min()) if not hits.empty else None


@st.cache_data
def wheelo_expected_trajectory() -> dict:
    """Deterministic (no simulation, no probability): cumulative real Wheelo
    match EV by round for every player, each rival's remaining-games ceiling,
    and the round the leader's cumulative EV first exceeds every rival's
    ceiling. Uses only match_status == "resolved" rows, matching this
    project's established Wheelo-reliability convention elsewhere."""
    wheelo = ed.load_wheelo_match_level()
    wheelo = wheelo[wheelo["match_status"] == "resolved"].copy()
    wheelo["round"] = wheelo["round"].astype(int)
    max_round = int(wheelo["round"].max())

    remaining_lookup = team_remaining_matches()
    team_by_player = wheelo.drop_duplicates("player_id").set_index("player_id")["team_id"].to_dict()

    pivot = wheelo.pivot_table(index="player_id", columns="round", values="wheelo_ev", aggfunc="sum", fill_value=0.0)
    for r in range(max_round + 1):
        if r not in pivot.columns:
            pivot[r] = 0.0
    pivot = pivot[sorted(pivot.columns)]
    cumulative = pivot.cumsum(axis=1)

    # Vectorised (numpy, not per-cell .loc) -- a Python-level double loop over
    # ~577 players x ~577 rivals x 25 rounds was measured too slow to be usable
    # interactively; this does the identical arithmetic via matrix ops.
    player_ids = cumulative.index.to_numpy()
    remaining_matrix = np.array(
        [[remaining_lookup.get((team_by_player.get(pid), r), 0) for r in cumulative.columns] for pid in player_ids],
        dtype=np.float64,
    )  # (n_players, n_rounds)
    cum_matrix = cumulative.to_numpy()  # (n_players, n_rounds)
    ceiling_matrix = cum_matrix + 3 * remaining_matrix

    clinch_round_by_player = {}
    for i, pid in enumerate(player_ids):
        ceilings_excl_self = ceiling_matrix.copy()
        ceilings_excl_self[i, :] = -np.inf
        max_rival_ceiling = ceilings_excl_self.max(axis=0)  # (n_rounds,)
        clinched_mask = cum_matrix[i] > max_rival_ceiling
        clinch_round_by_player[pid] = int(np.argmax(clinched_mask)) if clinched_mask.any() else None

    return {
        "cumulative_by_round": cumulative,
        "clinch_round_by_player": clinch_round_by_player,
        "max_round": max_round,
    }
