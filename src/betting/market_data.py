"""
Canonical, read-only access to the Production and Objective Monte Carlo
simulation draws that back every market-pricing function in
src/betting/pricing.py.

STRICT RULE: this module never fits, retrains, reweights, or otherwise
mutates either model. It only loads the already-persisted simulation arrays
(data/processed/mc_totals_2026.npy for Production, 100,000 draws;
data/processed/mc_totals_objective_2026.npy for Objective, 20,000 draws) and
their player-column indices (reports/2026_mc_player_index.csv,
reports/2026_objective_mc_player_index.csv). Shapes/conventions match
src/models/order_scenarios.py exactly -- reused here rather than
re-documented, since that module is the established source of truth for tie
handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class SimulationSet:
    """One model's raw (n_sims, n_players) simulation draws plus the
    player_id <-> column-index mapping needed to look a specific player up."""
    totals: np.ndarray  # (n_sims, n_players) int16
    player_ids: list[str]  # player_ids[i] is the identity of column i
    id_to_col: dict[str, int]
    n_sims: int

    def col(self, player_id: str) -> int | None:
        return self.id_to_col.get(str(player_id))

    def team_columns(self, team_id: str, team_lookup: pd.Series) -> list[int]:
        """Column indices for every player on `team_id`, per `team_lookup`
        (a player_id -> team_id Series, e.g. from the leaderboard)."""
        return [
            col for pid, col in self.id_to_col.items()
            if team_lookup.get(pid) == team_id
        ]


def _load(totals_path: Path, index_path: Path) -> SimulationSet:
    totals = np.load(totals_path)
    idx = pd.read_csv(index_path)
    idx["player_id"] = idx["player_id"].astype(str)
    if len(idx) != totals.shape[1]:
        raise ValueError(
            f"{index_path.name} has {len(idx)} rows but {totals_path.name} has "
            f"{totals.shape[1]} columns -- index/array out of sync, refusing to guess"
        )
    player_ids = idx["player_id"].tolist()
    id_to_col = {pid: i for i, pid in enumerate(player_ids)}
    return SimulationSet(totals=totals, player_ids=player_ids, id_to_col=id_to_col, n_sims=totals.shape[0])


def load_production_simulations() -> SimulationSet:
    return _load(PROCESSED / "mc_totals_2026.npy", REPORTS / "2026_mc_player_index.csv")


def load_objective_simulations() -> SimulationSet:
    return _load(PROCESSED / "mc_totals_objective_2026.npy", REPORTS / "2026_objective_mc_player_index.csv")


def load_team_lookup() -> pd.Series:
    """player_id (str) -> team_id, from the current Production leaderboard
    (the canonical, always-present source -- Objective's own team_id values
    are checked for agreement in tests, not re-derived here)."""
    lb = pd.read_csv(REPORTS / "2026_leaderboard.csv")
    lb["player_id"] = lb["player_id"].astype(str)
    return lb.set_index("player_id")["team_id"]


def load_player_names() -> pd.Series:
    lb = pd.read_csv(REPORTS / "2026_leaderboard.csv")
    lb["player_id"] = lb["player_id"].astype(str)
    return lb.set_index("player_id")["player_name"]
