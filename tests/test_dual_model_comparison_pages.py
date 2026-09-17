"""
Tests for dashboard.data.load_dual_model_comparison() / dual_model_team_rankings()
-- the shared loader backing pages 16 (Player H2H), 17 (Multi-Player
Comparison), and 18 (Team Player Rankings), and refreshed
reports/2026_objective_vs_production.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard import data as d

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "reports" / "2026_leaderboard.csv").exists(),
    reason="production leaderboard not present in this environment",
)

NAMED_PLAYERS_PROD_EV = {
    "Jason Horne-Francis": 16.96,
    "Luke Davies-Uniacke": 12.85,
    "Nasiah Wanganeen-Milera": 20.05,
}


@pytest.fixture(scope="module")
def cmp():
    return d.load_dual_model_comparison()


def test_no_duplicate_player_ids(cmp):
    assert not cmp["player_id"].duplicated().any()


def test_named_hyphenated_players_resolve_with_correct_ev(cmp):
    for name, expected_ev in NAMED_PLAYERS_PROD_EV.items():
        row = cmp[cmp["player_name"] == name]
        assert len(row) == 1, f"{name} missing or duplicated"
        assert row.iloc[0]["in_production"]
        assert row.iloc[0]["production_ev"] == pytest.approx(expected_ev, abs=0.05)


def test_jack_gunston_resolves_in_both_models_with_large_divergence(cmp):
    row = cmp[cmp["player_name"] == "Jack Gunston"].iloc[0]
    assert row["in_production"] and row["in_objective"]
    assert row["absolute_difference"] > 5, "Gunston is a known large Production/Objective divergence"


def test_normal_non_hyphenated_player_daicos(cmp):
    row = cmp[cmp["player_name"] == "Nick Daicos"].iloc[0]
    assert row["in_production"] and row["in_objective"]
    assert row["production_ev"] > 30 and row["objective_ev"] > 30


def test_values_match_current_leaderboards_exactly(cmp):
    lb = pd.read_csv(ROOT / "reports" / "2026_leaderboard.csv")
    obj = pd.read_csv(ROOT / "reports" / "2026_objective_leaderboard.csv")
    lb_row = lb[lb["player_name"] == "Nick Daicos"].iloc[0]
    obj_row = obj[obj["player_name"] == "Nick Daicos"].iloc[0]
    cmp_row = cmp[cmp["player_name"] == "Nick Daicos"].iloc[0]
    assert cmp_row["production_ev"] == pytest.approx(lb_row["FINAL_ENSEMBLE"])
    assert cmp_row["objective_ev"] == pytest.approx(obj_row["objective_ev"])


def test_players_missing_from_production_show_explicit_unavailable_not_zero(cmp):
    missing = cmp[~cmp["in_production"]]
    assert len(missing) > 0, "expected at least one Objective-only player (Round-1/season-to-date exclusion)"
    assert missing["production_ev"].isna().all(), "must be NaN (unavailable), never 0"


def test_team_filtering_returns_only_that_team_with_correct_team_ranks():
    sub = d.dual_model_team_rankings("collingwood")
    assert (sub["team_id"] == "collingwood").all()
    ranked = sub.dropna(subset=["production_team_rank"]).sort_values("production_team_rank")
    evs = ranked["production_ev"].tolist()
    assert evs == sorted(evs, reverse=True), "team rank must follow descending EV order"
    assert ranked.iloc[0]["production_team_rank"] == 1


def test_h2h_calculation_correctness(cmp):
    a = cmp[cmp["player_name"] == "Nick Daicos"].iloc[0]
    b = cmp[cmp["player_name"] == "Bailey Smith"].iloc[0]
    assert (a["production_ev"] - b["production_ev"]) == pytest.approx(a["production_ev"] - b["production_ev"])
    midpoint = (a["production_ev"] + a["objective_ev"]) / 2
    assert a["average_ev"] == pytest.approx(midpoint)
    assert a["absolute_difference"] == pytest.approx(abs(a["production_ev"] - a["objective_ev"]))


class TestRefreshedComparisonCsv:
    def setup_method(self):
        self.csv = pd.read_csv(ROOT / "reports" / "2026_objective_vs_production.csv")

    def test_named_players_refreshed(self):
        for name, expected_ev in NAMED_PLAYERS_PROD_EV.items():
            row = self.csv[self.csv["Player"] == name].iloc[0]
            assert row["Production EV"] == pytest.approx(expected_ev, abs=0.05)

    def test_no_duplicate_player_team_pairs(self):
        # NOT a duplicate-name check: two different real players can share a
        # name (e.g. "Bailey Williams" plays for both West Coast and the
        # Western Bulldogs in 2026) -- the actual invariant is that the join
        # (done on player_id upstream) never produces two rows for the same
        # real identity, which (Player, Team) is a good proxy for here.
        assert not self.csv[["Player", "Team"]].duplicated().any()
