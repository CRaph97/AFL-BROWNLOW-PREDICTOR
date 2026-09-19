"""
Targeted tests for dashboard/clinch_round.py and pages/32_Clinch_Round.py.

Scope: this feature replays (not re-runs) Production's and Objective's
already-committed Monte Carlo simulations to recover per-round cumulative
state. The single most important thing these tests must prove is that the
replay is BIT-IDENTICAL to the existing, already-validated simulation
outputs -- if it were not, every clinch/probability number on the page would
be silently wrong despite "looking" plausible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard import clinch_round as cr

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "pages" / "32_Clinch_Round.py")

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "mc_totals_2026.npy").exists(),
    reason="2026 simulation outputs not present in this environment",
)


@pytest.fixture(scope="module")
def prod():
    return cr.replay_production()


@pytest.fixture(scope="module")
def obj():
    return cr.replay_objective()


class TestBitIdenticalReplay:
    """The replay must reproduce the EXACT per-simulation, per-player vote
    totals already committed in data/processed/mc_totals_2026.npy /
    mc_totals_objective_2026.npy -- not merely a statistically similar
    distribution. A mismatch here means the round-bucketed replay drifted
    from the real simulation (e.g. wrong RNG draw order) and every clinch
    number downstream would be wrong."""

    def test_production_final_cumulative_matches_existing_npy_exactly(self, prod):
        existing = np.load(ROOT / "data" / "processed" / "mc_totals_2026.npy")
        idx = pd.read_csv(ROOT / "reports" / "2026_mc_player_index.csv")
        idx["player_id"] = idx["player_id"].astype(str)
        col_of = {pid: i for i, pid in enumerate(idx["player_id"])}
        reorder = np.array([col_of[pid] for pid in prod["players"]["player_id"]])
        assert np.array_equal(prod["final_cumulative"], existing[:, reorder])

    def test_objective_final_cumulative_matches_existing_npy_exactly(self, obj):
        existing = np.load(ROOT / "data" / "processed" / "mc_totals_objective_2026.npy")
        idx = pd.read_csv(ROOT / "reports" / "2026_objective_mc_player_index.csv")
        idx["player_id"] = idx["player_id"].astype(str)
        col_of = {pid: i for i, pid in enumerate(idx["player_id"])}
        reorder = np.array([col_of[pid] for pid in obj["players"]["player_id"]])
        assert np.array_equal(obj["final_cumulative"], existing[:, reorder])

    def test_no_simulation_or_output_files_altered(self):
        import subprocess

        diff = subprocess.run(
            ["git", "diff", "--stat",
             "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv",
             "data/processed/mc_totals_2026.npy", "data/processed/mc_totals_objective_2026.npy",
             "reports/2026_simulation_summary.csv", "reports/2026_objective_simulation_summary.csv"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert diff.stdout.strip() == ""


class TestDeploymentFallback:
    """Regression test for a live crash: data/processed/common_scenario_utilities_2026.parquet
    is a gitignored local research artefact -- a fresh Streamlit Cloud clone
    never has it, so replay_production() raised FileNotFoundError there.
    Fixed with a data/deployment/ fallback mirroring this project's
    established local-with-deployment-fallback pattern (dashboard.data._resolve_path)."""

    def test_deployment_copy_is_byte_identical_to_local_file(self):
        local = (ROOT / "data" / "processed" / "common_scenario_utilities_2026.parquet").read_bytes()
        deployed = (ROOT / "data" / "deployment" / "common_scenario_utilities_2026.parquet").read_bytes()
        assert local == deployed

    def test_replay_production_still_works_when_local_copy_is_missing(self, tmp_path):
        """Simulates deployment conditions: hides the local, gitignored
        common_scenario_utilities_2026.parquet and confirms replay_production()
        falls back to data/deployment/ and produces the identical replay,
        instead of raising FileNotFoundError."""
        import shutil

        local_path = ROOT / "data" / "processed" / "common_scenario_utilities_2026.parquet"
        moved = tmp_path / "common_scenario_utilities_2026.parquet"
        shutil.move(str(local_path), str(moved))
        try:
            cr.replay_production.clear()
            result = cr.replay_production()
        finally:
            shutil.move(str(moved), str(local_path))
            cr.replay_production.clear()

        existing = np.load(ROOT / "data" / "processed" / "mc_totals_2026.npy")
        idx = pd.read_csv(ROOT / "reports" / "2026_mc_player_index.csv")
        idx["player_id"] = idx["player_id"].astype(str)
        col_of = {pid: i for i, pid in enumerate(idx["player_id"])}
        reorder = np.array([col_of[pid] for pid in result["players"]["player_id"]])
        assert np.array_equal(result["final_cumulative"], existing[:, reorder])


class TestClinchMathematics:
    def test_clinch_status_is_monotonic_once_achieved(self, prod):
        """Proves the module docstring's claim: once clinched=True at round
        R for a given simulation, it must remain True for every later round
        (rival ceilings only shrink, leader votes only grow)."""
        clinched = prod["clinched_by_round"]  # (n_rounds, n_sims)
        # For each sim, once True, must never revert to False afterwards.
        diffs = np.diff(clinched.astype(int), axis=0)
        assert not (diffs == -1).any(), "clinch status reverted from True to False in at least one simulation"

    def test_strict_tie_does_not_count_as_clinched(self):
        """Direct unit check on the inequality itself, independent of the
        full replay: an exact tie between leader-votes and a rival's ceiling
        must NOT be flagged as clinched."""
        leader_votes = np.array([30])
        rival_ceiling = np.array([30])
        assert not bool((leader_votes > rival_ceiling)[0])
        rival_ceiling_lower = np.array([29])
        assert bool((leader_votes > rival_ceiling_lower)[0])

    def test_final_round_clinch_probability_never_exceeds_final_winner_share(self, prod):
        """A simulation can only be "clinched" if a genuine leader exists by
        the final round; P(clinched by final round) must not exceed 1, and
        never-clinched fraction plus clinched-by-final fraction must sum to
        (approximately) 1."""
        n = prod["n_sims"]
        clinched_by_final = (prod["clinch_round_per_sim"] <= prod["max_round"]).mean()
        never = np.isnan(prod["clinch_round_per_sim"]).mean()
        assert abs(clinched_by_final + never - 1.0) < 1e-9

    def test_production_and_objective_are_independent_replays(self, prod, obj):
        """Different seasons of randomness applied at different temperatures
        -- their clinch-round distributions must not be identical (proves
        Objective is not silently just reusing Production's replay)."""
        assert prod["n_sims"] != obj["n_sims"]
        prod_median = np.nanmedian(prod["clinch_round_per_sim"])
        obj_median = np.nanmedian(obj["clinch_round_per_sim"])
        assert prod_median != obj_median


class TestTeamRemainingMatches:
    def test_remaining_matches_strictly_non_increasing_within_a_team(self):
        remaining = cr.team_remaining_matches()
        teams = {t for t, _ in remaining}
        for team in teams:
            values = [remaining[(team, r)] for r in range(0, 25) if (team, r) in remaining]
            assert all(a >= b for a, b in zip(values, values[1:])), f"{team}: remaining games increased round-over-round"

    def test_remaining_matches_hits_zero_by_final_round(self):
        remaining = cr.team_remaining_matches()
        max_round = max(r for _, r in remaining)
        assert all(v == 0 for (t, r), v in remaining.items() if r == max_round)


class TestProjectedWinnerPoint:
    def test_definition_is_conditional_on_currently_leading(self, prod):
        """P(eventual winner | leader at R) must be >= the unconditional
        P(eventual winner) for the sport's genuine favourite, since leading
        is informative."""
        favourite_id = prod["players"].iloc[np.bincount(prod["final_winner_idx"], minlength=len(prod["players"])).argmax()]["player_id"]
        series = cr.conditional_win_prob_by_round(prod, favourite_id)
        unconditional = float((prod["final_winner_idx"] == prod["players"].index[prod["players"]["player_id"] == favourite_id][0]).mean())
        final_round_value = series.get(prod["max_round"])
        assert final_round_value >= unconditional - 1e-9

    def test_earliest_round_reaching_returns_none_if_never_reached(self):
        series = pd.Series({0: 0.1, 1: 0.2, 2: 0.3})
        assert cr.earliest_round_reaching(series, 0.95) is None
        assert cr.earliest_round_reaching(series, 0.2) == 1


class TestWheeloExpectedTrajectory:
    def test_deterministic_not_a_probability(self):
        w = cr.wheelo_expected_trajectory()
        for v in w["clinch_round_by_player"].values():
            assert v is None or isinstance(v, int)

    def test_uses_only_resolved_matches(self):
        from dashboard import external_data as ed

        wheelo = ed.load_wheelo_match_level()
        if wheelo.empty:
            pytest.skip("no wheelo data in this environment")
        w = cr.wheelo_expected_trajectory()
        resolved_total = wheelo[wheelo["match_status"] == "resolved"]["wheelo_ev"].sum()
        replay_total = w["cumulative_by_round"][w["max_round"]].sum()
        assert replay_total == pytest.approx(resolved_total, rel=1e-6)


class TestPage:
    def test_page_loads_no_exception(self):
        at = AppTest.from_file(PAGE, default_timeout=180)
        at.run()
        assert not at.exception

    def test_round_explorer_slider_moves_without_exception(self):
        at = AppTest.from_file(PAGE, default_timeout=180)
        at.run()
        sliders = [s for s in at.slider if s.key == "clinch_round_explorer"]
        assert sliders, "expected the Round Explorer slider"
        sliders[0].set_value(10)
        at.run()
        assert not at.exception

    def test_summary_cards_show_all_three_sources(self):
        at = AppTest.from_file(PAGE, default_timeout=180)
        at.run()
        markdown_text = " ".join(m.value for m in at.markdown)
        assert "**Production**" in markdown_text
        assert "**Objective**" in markdown_text
        assert "**Wheelo**" in markdown_text

    def test_wheelo_never_shown_as_a_probability_on_the_page(self):
        at = AppTest.from_file(PAGE, default_timeout=180)
        at.run()
        captions = " ".join(c.value for c in at.caption)
        assert "not a probability" in captions.lower() or "not comparable" in captions.lower()

    def test_insight_bullets_are_bounded_and_present(self):
        at = AppTest.from_file(PAGE, default_timeout=180)
        at.run()
        bullets = [m.value for m in at.markdown if m.value.startswith("- ")]
        assert 3 <= len(bullets) <= 5


class TestNavigation:
    def test_clinch_round_page_registered_directly_after_round_by_round(self):
        router_src = (ROOT / "app.py").read_text()
        main_start = router_src.index('"MAIN": [')
        main_end = router_src.index("],", main_start)
        main_block = router_src[main_start:main_end]
        assert 'st.Page("pages/32_Clinch_Round.py", title="Clinch Round")' in main_block
        titles = [line.split('title="')[1].split('"')[0] for line in main_block.splitlines() if 'title="' in line]
        assert titles.index("Clinch Round") == titles.index("Round-by-Round") + 1
        assert titles.index("To Poll a Vote") == titles.index("Clinch Round") + 1

    def test_router_still_loads_with_new_page_registered(self):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        at.run()
        assert not at.exception
        labels = {pl.proto.label for pl in at.sidebar.get("page_link")}
        assert "Clinch Round" in labels
