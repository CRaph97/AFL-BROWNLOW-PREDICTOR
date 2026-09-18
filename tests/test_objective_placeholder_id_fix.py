"""
Regression tests for a real Objective-model season-leaderboard bug: 5 real 2026
players (Charlie Cameron, Jack Ross, Billy Wilson, Jack Graham, Jack Williams)
never had their afltables id resolved for ANY of their matches, so CORE
assigned a fresh `NOID2026_<row-index>` placeholder per match-row (correct,
documented CORE behaviour -- see build_2026_extension.py::build_core_2026()).
The Objective season leaderboard groups by player_id to sum a player's match
EVs into one season total, so each of these players fragmented into one
near-zero row PER MATCH instead of a single correctly-summed season row (e.g.
Charlie Cameron: 23 separate rows of 0.0002-0.51 EV, no row anywhere showing
his real ~0.94 season total).

This was a real prediction-value bug (failure mode "duplicated/fragmented
underlying predictions"), not a harmless duplicate-export bug -- confirmed by
summing the pre-fix fragment rows and finding no single row matched that sum.

Fix: src.models.objective_stats_model._stabilise_placeholder_ids() replaces
every NOID2026_* id, in this pipeline's own in-memory copy only, with a stable
synthetic id derived from (team_id, normalised player_name) -- safe because
that name+team pairing already IS one agreed identity in the source data, just
missing a numeric id. The shared model_core_2026.parquet file (which Production
also reads) is never written to, so Production is unaffected.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models.objective_stats_model import _stabilise_placeholder_ids, load_2026_input_frame

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

PREVIOUSLY_FRAGMENTED_PLAYERS = [
    "Charlie Cameron",
    "Jack Ross",
    "Billy Wilson",
    "Jack Graham",
    "Jack Williams",
]


def test_stabilise_placeholder_ids_collapses_same_name_team_to_one_id():
    df = pd.DataFrame(
        {
            "player_id": ["NOID2026_5", "NOID2026_19", "NOID2026_101", "12345"],
            "player_name": ["Charlie Cameron", "Charlie Cameron", "Charlie Cameron", "Nick Daicos"],
            "team_id": ["brisbane_lions", "brisbane_lions", "brisbane_lions", "collingwood"],
        }
    )
    out = _stabilise_placeholder_ids(df)
    assert out["player_id"].nunique() == 2
    assert out.loc[out["player_name"] == "Charlie Cameron", "player_id"].nunique() == 1
    assert (out.loc[out["player_name"] == "Nick Daicos", "player_id"] == "12345").all()


def test_stabilise_placeholder_ids_keeps_different_players_distinct():
    """Two different real players who both happened to be unresolved must NOT
    be merged just because they're both placeholders -- only a shared
    (team_id, normalised name) may collapse."""
    df = pd.DataFrame(
        {
            "player_id": ["NOID2026_1", "NOID2026_2"],
            "player_name": ["Jack Ross", "Jack Graham"],
            "team_id": ["richmond", "west_coast"],
        }
    )
    out = _stabilise_placeholder_ids(df)
    assert out["player_id"].nunique() == 2


def test_load_2026_input_frame_produces_stable_ids_per_player():
    """End-to-end on real data: every one of the 5 known-affected players must
    have exactly one distinct player_id across all of their match rows."""
    df = load_2026_input_frame()
    for name in PREVIOUSLY_FRAGMENTED_PLAYERS:
        rows = df[df["player_name"] == name]
        assert len(rows) > 0, f"{name} has no rows at all -- test data problem"
        assert rows["player_id"].nunique() == 1, (
            f"{name} still has {rows['player_id'].nunique()} distinct ids across "
            f"{len(rows)} matches: {sorted(rows['player_id'].unique())}"
        )


@pytest.mark.skipif(
    not (REPORTS / "2026_objective_leaderboard.csv").exists(),
    reason="objective leaderboard not present in this environment",
)
class TestRebuiltObjectiveLeaderboard:
    def setup_method(self):
        self.lb = pd.read_csv(REPORTS / "2026_objective_leaderboard.csv")

    def test_no_duplicate_player_id_rows(self):
        assert not self.lb["player_id"].duplicated().any()

    def test_previously_fragmented_players_have_exactly_one_row(self):
        for name in PREVIOUSLY_FRAGMENTED_PLAYERS:
            rows = self.lb[self.lb["player_name"] == name]
            assert len(rows) == 1, f"expected exactly 1 row for {name}, found {len(rows)}"

    def test_charlie_cameron_ev_matches_real_summed_match_data(self):
        """Recompute Charlie Cameron's season EV independently from the raw
        match-level file and confirm it matches the leaderboard exactly --
        proof this is a real fix, not merely fewer visible rows."""
        match_scores = pd.read_csv(REPORTS / "2026_objective_match_scores.csv")
        cam_id = self.lb.loc[self.lb["player_name"] == "Charlie Cameron", "player_id"].iloc[0]
        expected = match_scores.loc[match_scores["player_id"] == cam_id, "expected_votes"].sum()
        actual = self.lb.loc[self.lb["player_name"] == "Charlie Cameron", "objective_ev"].iloc[0]
        assert actual == pytest.approx(expected, abs=1e-6)
        assert actual > 0.5, "sanity check: a real ~23-match player should have a non-trivial season EV"

    def test_production_leaderboard_has_no_placeholder_ids(self):
        """This fix touches only the Objective pipeline's in-memory copy of
        CORE/ADVANCED, and never writes to the shared model_core_2026.parquet
        file -- Production's own leaderboard build already excludes
        unresolved-id rows entirely (its player_id column is pure int64, never
        a NOID2026_* string), so it was never exposed to this bug and remains
        unaffected by this fix."""
        prod = pd.read_csv(REPORTS / "2026_leaderboard.csv")
        assert prod["player_id"].astype(str).str.startswith("NOID2026_").sum() == 0
        assert pd.api.types.is_integer_dtype(prod["player_id"])

    def test_season_total_votes_still_1242(self):
        import json

        qc_path = REPORTS / "2026_quality_checks.json"
        if qc_path.exists():
            qc = json.loads(qc_path.read_text())
            assert qc["actual_season_total_expected_votes"] == pytest.approx(1242.0, abs=1e-6)
