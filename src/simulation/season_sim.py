"""
2027 season simulation framework (model-agnostic, reusable).

Input: a match-level prediction frame with coherent within-match P3/P2/P1
(any candidate model's output). Each simulated season draws, for every
match independently, one ordered 3-2-1 allocation from the Plackett-Luce
sequential process implied by the utilities (so every completed match
awards exactly 3+2+1 = 6 votes to three distinct players), then sums to
season totals. From the draws: winner / top-N probabilities, exact
finishing-order queries, H2H, X+ votes, To Poll a Vote, team leader, and a
clinch-round analysis (earliest round after which the eventual leader's lead
exceeds the maximum votes still available to the runner-up).

No 2027 forecast is produced here: the framework is validated against
historical / frozen outputs in tests (tests/test_2027_rd.py) and by
`validate_against_frozen()` which compares a re-simulation of the frozen 2026
Production probabilities to the frozen simulation summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20270101


def utilities_from_probs(p3: np.ndarray) -> np.ndarray:
    """Plackett-Luce utilities consistent with the marginal P3 (u = log p3)."""
    return np.log(np.clip(p3, 1e-12, 1.0))


def simulate_matches(preds: pd.DataFrame, n_sims: int, seed: int = SEED, round_col: str = "round") -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Returns (totals [n_sims x n_players] int16, players frame, votes_by_round [n_sims x n_rounds x n_players] int8 if round_col present else None)."""
    rng = np.random.default_rng(seed)
    players = preds[["player_id", "player_name", "team_id"]].drop_duplicates("player_id").reset_index(drop=True)
    col = {p: i for i, p in enumerate(players["player_id"])}
    totals = np.zeros((n_sims, len(players)), dtype=np.int16)
    has_round = round_col in preds.columns
    rounds = sorted(preds[round_col].unique()) if has_round else [0]
    ridx = {r: i for i, r in enumerate(rounds)}
    by_round = np.zeros((n_sims, len(rounds), len(players)), dtype=np.int16) if has_round else None
    for mid, g in preds.groupby("match_id", sort=False):
        u = utilities_from_probs(g["p3"].to_numpy())
        cols = np.array([col[p] for p in g["player_id"]])
        # Gumbel-max trick: sorting u + Gumbel noise gives an exact Plackett-Luce ordering draw
        gum = rng.gumbel(size=(n_sims, len(u)))
        order = np.argsort(-(u[None, :] + gum), axis=1)[:, :3]
        picks = cols[order]  # n_sims x 3
        rows = np.arange(n_sims)
        for k, v in enumerate((3, 2, 1)):
            totals[rows, picks[:, k]] += v
            if has_round:
                by_round[rows, ridx[g[round_col].iloc[0]], picks[:, k]] += v
    return totals, players, by_round


def summarise(totals: np.ndarray, players: pd.DataFrame) -> pd.DataFrame:
    ranks = np.argsort(np.argsort(-totals, axis=1, kind="stable"), axis=1) + 1
    out = players.copy()
    out["sim_mean"] = totals.mean(axis=0); out["sim_median"] = np.median(totals, axis=0)
    out["sim_p2_5"] = np.percentile(totals, 2.5, axis=0); out["sim_p97_5"] = np.percentile(totals, 97.5, axis=0)
    out["p_winner"] = (ranks == 1).mean(axis=0)
    for k in (3, 5, 10, 20):
        out[f"p_top{k}"] = (ranks <= k).mean(axis=0)
    out["p_poll_a_vote"] = (totals >= 1).mean(axis=0)
    for x in (10, 15, 20, 25, 30):
        out[f"p_{x}_plus"] = (totals >= x).mean(axis=0)
    return out.sort_values("sim_mean", ascending=False).reset_index(drop=True)


def h2h(totals: np.ndarray, players: pd.DataFrame, a: str, b: str) -> dict:
    col = {p: i for i, p in enumerate(players["player_id"])}
    ta, tb = totals[:, col[a]], totals[:, col[b]]
    return {"p_a_wins": float((ta > tb).mean()), "p_b_wins": float((tb > ta).mean()), "p_tie": float((ta == tb).mean())}


def exact_order(totals: np.ndarray, players: pd.DataFrame, ordered_ids: list[str]) -> dict:
    col = {p: i for i, p in enumerate(players["player_id"])}
    k = len(ordered_ids)
    order = np.argsort(-totals, axis=1, kind="stable")[:, :k]
    target = np.array([col[p] for p in ordered_ids])
    exact = (order == target[None, :]).all(axis=1).mean()
    ranks = np.argsort(np.argsort(-totals, axis=1, kind="stable"), axis=1) + 1
    setp = (ranks[:, target] <= k).all(axis=1).mean()
    return {"p_exact_order": float(exact), "p_all_in_top_k": float(setp), "n_sims": int(totals.shape[0])}


def team_leader(totals: np.ndarray, players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, g in players.groupby("team_id"):
        idx = g.index.to_numpy()
        sub = totals[:, idx]
        mx = sub.max(axis=1, keepdims=True)
        is_leader = (sub == mx)
        share = is_leader / is_leader.sum(axis=1, keepdims=True)  # ties share
        p = share.mean(axis=0)
        for i, pid in zip(idx, g["player_id"]):
            rows.append({"team_id": team, "player_id": pid, "player_name": players.loc[i, "player_name"], "p_team_leader": float(p[list(idx).index(i)])})
    return pd.DataFrame(rows).sort_values(["team_id", "p_team_leader"], ascending=[True, False])


def clinch_round(by_round: np.ndarray, players: pd.DataFrame, votes_per_round_max: int = 3) -> pd.DataFrame:
    """For each simulation, the earliest round after which the eventual
    winner's lead over every other player exceeds the votes still available
    to them (3 per remaining round). Returns per-player P(clinch by round r)."""
    n_sims, n_rounds, n_players = by_round.shape
    cum = np.cumsum(by_round, axis=1).astype(float)
    final = cum[:, -1, :]
    winner = np.argmax(final, axis=1)
    rows = []
    clinch = np.full(n_sims, n_rounds, dtype=int)
    for r in range(n_rounds):
        remaining = (n_rounds - 1 - r) * votes_per_round_max
        cur = cum[:, r, :]
        lead = cur[np.arange(n_sims), winner][:, None] - cur
        lead[np.arange(n_sims), winner] = np.inf
        clinched = (lead > remaining).all(axis=1)
        clinch = np.where((clinch == n_rounds) & clinched, r, clinch)
    for w in np.unique(winner):
        m = winner == w
        rows.append({"player_id": players.loc[w, "player_id"], "player_name": players.loc[w, "player_name"], "p_winner": float(m.mean()),
                     "median_clinch_round_index": float(np.median(clinch[m])), "p_clinch_before_final_round": float((clinch[m] < n_rounds - 1).mean())})
    return pd.DataFrame(rows).sort_values("p_winner", ascending=False)


def validate_against_frozen(n_sims: int = 20000) -> dict:
    """Re-simulate the frozen 2026 Production match probabilities and compare
    to the frozen simulation summary (reports/2026_simulation_summary.csv)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    mp = pd.read_csv(root / "reports" / "2026_match_probabilities.csv"); mp["player_id"] = mp["player_id"].astype(str)
    totals, players, _ = simulate_matches(mp, n_sims)
    s = summarise(totals, players)
    ref = pd.read_csv(root / "reports" / "2026_simulation_summary.csv"); ref["player_id"] = ref["player_id"].astype(str)
    j = s.merge(ref[["player_id", "sim_mean_votes", "prob_rank_1", "prob_top10_rank"]], on="player_id")
    return {"n_players": int(len(j)), "max_abs_diff_mean": float((j["sim_mean"] - j["sim_mean_votes"]).abs().max()),
            "corr_p_winner": float(np.corrcoef(j["p_winner"], j["prob_rank_1"])[0, 1]),
            "max_abs_diff_p_top10": float((j["p_top10"] - j["prob_top10_rank"]).abs().max()),
            "daicos_p_winner": float(j.loc[j["player_id"] == "12943", "p_winner"].iloc[0])}
