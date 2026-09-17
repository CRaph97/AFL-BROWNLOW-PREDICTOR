"""
Objective-model-side market probabilities for the Betting Opportunities
Compare-Both view, mirroring (not importing -- separate repo boundary) the
same tie-handling conventions already used in AFL-BROWNLOW-MARKETS'
src/model/probability.py for the production model:

- WINNER / TOP_N: strict-greater-than-zero-ties get full credit; an exact
  tie for the target rank splits credit 1/(number tied).
- PLAYER_VOTES_OU: strict > / < the line (matches how a real O/U bet settles).
- PICK_YOUR_OWN_VOTES ("to poll N or more"): >= threshold.
- PLAYER_H2H: P(A strictly outpolls B); ties excluded from both sides
  (matches a real head-to-head bet pushing on a dead heat).
- TEAM_TOP_POLLER: same tie-shared-credit rule, restricted to one team's
  playing roster columns.

Only reads this repo's own Objective-model simulation draws
(data/processed/mc_totals_objective_2026.npy) plus config/team_mapping.csv
for team-name resolution. Never reads or writes anything in the separate
AFL-BROWNLOW-MARKETS repo -- that boundary is enforced entirely in
dashboard/betting_data.py, which is the only caller of this module.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


class ObjectiveMarketModel:
    """Loads once, resolves names/teams, exposes per-market probability
    lookups. Every lookup returns None (not a fabricated number) when a
    player/team/market shape can't be reliably resolved."""

    def __init__(self, totals: np.ndarray, players: pd.DataFrame, team_mapping: pd.DataFrame):
        self.totals = totals
        self.players = players.reset_index(drop=True)
        self._name_to_col = {_norm_name(n): i for i, n in enumerate(self.players["player_name"])}
        self._surname_initial_to_cols: dict[tuple[str, str], list[int]] = {}
        for i, n in enumerate(self.players["player_name"]):
            parts = _norm_name(n).split()
            if len(parts) >= 2:
                key = (parts[-1], parts[0][0])
                self._surname_initial_to_cols.setdefault(key, []).append(i)

        self.team_columns: dict[str, list[int]] = {}
        for tid, sub in self.players.groupby("team_id"):
            self.team_columns[tid] = sub.index.tolist()

        # source_name -> canonical_team_id, longest source_name first so a
        # full-nickname match ("Gold Coast SUNS") wins over a shorter alias
        # ("Gold Coast") when both are substrings of a market_name.
        self._team_aliases = sorted(
            team_mapping[["source_name", "canonical_team_id"]].itertuples(index=False, name=None),
            key=lambda t: -len(t[0]),
        )

    def resolve_player_col(self, name: str) -> int | None:
        key = _norm_name(name)
        if key in self._name_to_col:
            return self._name_to_col[key]
        parts = key.split()
        if len(parts) >= 2:
            cand = self._surname_initial_to_cols.get((parts[-1], parts[0][0]))
            if cand and len(cand) == 1:
                return cand[0]
        return None

    def resolve_team_id(self, market_name: str) -> str | None:
        if not isinstance(market_name, str):
            return None
        for source_name, team_id in self._team_aliases:
            if source_name.lower() in market_name.lower():
                return team_id
        return None

    def prob_winner(self, col: int) -> float:
        target = self.totals[:, [col]]
        strictly_greater = (self.totals > target).sum(axis=1)
        ties = (self.totals == target).sum(axis=1)
        credit = np.where(strictly_greater == 0, 1.0 / ties, 0.0)
        return float(credit.mean())

    def prob_top_n(self, col: int, n: float) -> float:
        target = self.totals[:, [col]]
        rank = (self.totals > target).sum(axis=1) + 1
        return float((rank <= n).mean())

    def prob_over(self, col: int, line: float) -> float:
        return float((self.totals[:, col] > line).mean())

    def prob_under(self, col: int, line: float) -> float:
        return float((self.totals[:, col] < line).mean())

    def prob_at_least(self, col: int, threshold: float) -> float:
        return float((self.totals[:, col] >= threshold).mean())

    def prob_h2h(self, col_a: int, col_b: int) -> float:
        return float((self.totals[:, col_a] > self.totals[:, col_b]).mean())

    def prob_top_of_team(self, col: int, team_id: str) -> float | None:
        cols = self.team_columns.get(team_id)
        if not cols or col not in cols:
            return None
        sub = self.totals[:, cols]
        pos = cols.index(col)
        target = sub[:, [pos]]
        strictly_greater = (sub > target).sum(axis=1)
        ties = (sub == target).sum(axis=1)
        credit = np.where(strictly_greater == 0, 1.0 / ties, 0.0)
        return float(credit.mean())


def load_objective_market_model(processed_dir: Path | None = None, reports_dir: Path | None = None,
                                 config_dir: Path | None = None) -> "ObjectiveMarketModel | None":
    processed_dir = processed_dir or (ROOT / "data" / "processed")
    reports_dir = reports_dir or (ROOT / "reports")
    config_dir = config_dir or (ROOT / "config")

    totals_path = processed_dir / "mc_totals_objective_2026.npy"
    index_path = reports_dir / "2026_objective_mc_player_index.csv"
    mapping_path = config_dir / "team_mapping.csv"
    if not (totals_path.exists() and index_path.exists() and mapping_path.exists()):
        return None

    totals = np.load(totals_path)
    players = pd.read_csv(index_path)
    team_mapping = pd.read_csv(mapping_path)
    return ObjectiveMarketModel(totals, players, team_mapping)


def objective_probability_for_row(model: "ObjectiveMarketModel", row: pd.Series) -> tuple[float | None, str]:
    """Returns (probability_or_None, reason_if_none). Only the priority market
    types get a real attempt; everything else is explicitly NOT AVAILABLE."""
    mt = row.get("market_type")

    if mt == "WINNER":
        col = model.resolve_player_col(row.get("player") or row.get("selection"))
        if col is None:
            return None, "player not resolved"
        return model.prob_winner(col), ""

    if mt == "TOP_N":
        col = model.resolve_player_col(row.get("player") or row.get("selection"))
        if col is None:
            return None, "player not resolved"
        line = row.get("line")
        if pd.isna(line):
            return None, "missing line"
        return model.prob_top_n(col, float(line)), ""

    if mt == "PLAYER_VOTES_OU":
        col = model.resolve_player_col(row.get("player"))
        if col is None:
            return None, "player not resolved"
        line = row.get("line")
        if pd.isna(line):
            return None, "missing line"
        sel = str(row.get("selection", "")).strip().lower()
        if sel == "over":
            return model.prob_over(col, float(line)), ""
        if sel == "under":
            return model.prob_under(col, float(line)), ""
        return None, f"unrecognised O/U selection '{row.get('selection')}'"

    if mt == "PICK_YOUR_OWN_VOTES":
        col = model.resolve_player_col(row.get("player") or row.get("selection"))
        if col is None:
            return None, "player not resolved"
        line = row.get("line")
        if pd.isna(line):
            return None, "missing line"
        return model.prob_at_least(col, float(line)), ""

    if mt == "PLAYER_H2H":
        col_a = model.resolve_player_col(row.get("player"))
        col_b = model.resolve_player_col(row.get("opponent_or_pair"))
        if col_a is None or col_b is None:
            return None, "one or both players not resolved"
        return model.prob_h2h(col_a, col_b), ""

    if mt == "TEAM_TOP_POLLER":
        col = model.resolve_player_col(row.get("player") or row.get("selection"))
        team_id = model.resolve_team_id(row.get("market_name"))
        if col is None or team_id is None:
            return None, "player or team not resolved"
        p = model.prob_top_of_team(col, team_id)
        return (p, "") if p is not None else (None, "player not on resolved team roster")

    if mt == "TEAM_VOTES_OU":
        return None, "excluded pending price-freshness verification (matches production's own exclusion)"

    return None, f"market type '{mt}' outside this pass's priority list"
