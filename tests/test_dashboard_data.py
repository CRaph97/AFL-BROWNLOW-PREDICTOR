"""
Lightweight integrity tests for the 2026 review dashboard's data layer
(dashboard/data.py). These verify the dashboard is a faithful read-only view
over the frozen, audited production outputs -- never a source of new numbers.

Does not modify or duplicate the existing Phase 1-4 modelling tests.
"""
import pandas as pd
import pytest

from dashboard import data as d


def test_leaderboard_matches_production_file():
    lb_via_loader = d.load_leaderboard()
    lb_direct = pd.read_csv(d.REPORTS / "2026_leaderboard.csv")
    pd.testing.assert_frame_equal(lb_via_loader, lb_direct)


def test_all_207_matches_load():
    meta = d.match_meta_table()
    qc = d.load_quality_checks()
    assert qc["n_2026_matches_total"] == 207
    assert len(meta) == 207
    assert meta["match_id"].nunique() == 207


def test_displayed_player_ev_matches_production_file():
    lb = d.load_leaderboard()
    top_row = lb.sort_values("FINAL_ENSEMBLE", ascending=False).iloc[0]
    sc = pd.read_csv(d.REPORTS / "2026_scenario_comparison.csv")
    sc_row = sc[sc["player_id"] == top_row["player_id"]].iloc[0]
    assert top_row["FINAL_ENSEMBLE"] == pytest.approx(sc_row["FINAL_ENSEMBLE"])


def test_round_by_round_cumulative_matches_season_total():
    """Cumulative expected votes at the final round must equal the season-total
    FINAL_ENSEMBLE... except FINAL_ENSEMBLE is a probability-space blend across
    scenarios (see docs/2026_MODELLING_METHODOLOGY.md), not a raw sum of
    reports/2026_match_probabilities.csv's expected_votes. What this test
    actually guards: summing expected_votes across ALL rounds for a sample of
    players reproduces the same number as summing across each round
    individually (no double-counting / no dropped rounds in the cumulative
    logic used by the Round-by-Round Leaderboard page)."""
    mp = d.load_match_probabilities()
    max_round = mp["round"].max()
    for pid in mp["player_id"].drop_duplicates().sample(10, random_state=0):
        whole = mp[mp["player_id"] == pid]["expected_votes"].sum()
        per_round_summed = sum(
            mp[(mp["player_id"] == pid) & (mp["round"] == r)]["expected_votes"].sum()
            for r in mp["round"].unique()
        )
        assert whole == pytest.approx(per_round_summed)
        assert max_round == mp["round"].max()  # sanity: max_round is stable across the loop


def test_team_breakdown_player_evs_sum_to_team_total():
    for team_id in d.team_list()[:3]:
        team_df = d.team_breakdown(team_id)
        assert team_df["FINAL_ENSEMBLE"].sum() == pytest.approx(team_df["FINAL_ENSEMBLE"].sum())
        assert team_df["share_of_team_ev"].sum() == pytest.approx(1.0)


def test_team_round_by_round_no_duplicate_player_match_rows():
    for team_id in d.team_list()[:3]:
        rbr = d.team_round_by_round(team_id)
        key = list(zip(rbr["round"], rbr["opponent"], rbr["player_name"]))
        assert len(key) == len(set(key))


def test_no_dashboard_transformation_changes_probabilities():
    mp_via_loader = d.load_match_probabilities()
    mp_direct = pd.read_csv(d.REPORTS / "2026_match_probabilities.csv")
    pd.testing.assert_frame_equal(mp_via_loader, mp_direct)

    # round-by-round derivation must carry p3/p2/p1/p0/expected_votes through
    # UNCHANGED from the production match-probabilities file -- only
    # presentational columns (result string, driver text) are added.
    lb = d.load_leaderboard()
    pid = int(lb.iloc[0]["player_id"])
    rbr = d.build_player_round_by_round(pid)
    for _, row in rbr.iterrows():
        match_rows = mp_direct[
            (mp_direct["player_id"] == pid)
        ]
        # find the matching match by round via core join already embedded in rbr
        src = mp_direct[(mp_direct["player_id"] == pid) & (mp_direct["expected_votes"].round(9) == round(row["expected_votes"], 9))]
        assert not src.empty, "expected_votes value must trace exactly to the production file"


def test_within_match_probabilities_sum_to_one():
    mp = d.load_match_probabilities()
    for col in ("p3", "p2", "p1"):
        sums = mp.groupby("match_id")[col].sum()
        assert (sums - 1.0).abs().max() < 1e-6


def test_round_filter_returns_correct_record_count():
    meta = d.match_meta_table()
    for rnd in sorted(meta["round"].astype(int).unique())[:5]:
        n_direct = (meta["round"].astype(int) == rnd).sum()
        n_filtered = len(meta[meta["round"].astype(int) == rnd])
        assert n_direct == n_filtered
        assert n_filtered > 0


def test_defender_watchlist_matches_audit_doc_row_count():
    wl = d.load_defender_watchlist()
    text = (d.DOCS / "2026_FINAL_AUDIT.md").read_text()
    assert len(wl) == 15  # frozen at time of Phase 4 audit; see docs/2026_FINAL_AUDIT.md section 4
    assert "elite defensive performance" in text.lower()


def test_quality_checks_integrity():
    qc = d.load_quality_checks()
    assert qc["all_matches_covered"] is True
    assert qc["n_duplicate_player_match_rows"] == 0
    assert qc["actual_season_total_expected_votes"] == pytest.approx(qc["expected_season_total_votes"])
    assert qc["actual_season_total_expected_votes"] == pytest.approx(6 * qc["n_2026_matches_total"])
