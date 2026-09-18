"""
QA for the External Brownlow Benchmarking layer (src/external, dashboard/
external_data.py, pages 19-22). Strictly read-only against Production/
Objective -- these tests exist specifically to prove that.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.external.identity import (
    load_canonical_players, resolve_external_players, resolve_external_match_players,
    _normalise_full_surname, _first_initial,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_AVAILABLE = (ROOT / "data" / "external" / "wheelo-brownlow-predictions.csv").exists()
pytestmark = pytest.mark.skipif(not RAW_AVAILABLE, reason="external raw data not present in this environment")

HARD_CASES = [
    ("Jason Horne-Francis", "port_adelaide"),
    ("Nasiah Wanganeen-Milera", "st_kilda"),
    ("Luke Davies-Uniacke", "north_melbourne"),
    ("Darcy Byrne-Jones", "port_adelaide"),
    ("Jack Gunston", "hawthorn"),
    ("Nick Daicos", "collingwood"),
]


class TestIdentityResolution:
    def test_hard_cases_resolve_via_season_level(self):
        from src.external.wheelo_source import RAW_CSV
        from src.external.identity import WHEELO_TEAM_MAP
        raw = pd.read_csv(RAW_CSV)
        raw["team_id"] = raw["Team"].map(WHEELO_TEAM_MAP)
        season = raw.groupby(["Player", "team_id"], as_index=False)["Votes"].sum()
        resolved = resolve_external_players(season, name_col="Player", team_col="team_id")
        for name, team in HARD_CASES:
            row = resolved[(resolved["Player"] == name) & (resolved["team_id"] == team)]
            assert len(row) == 1, f"{name} missing from resolved output"
            assert row.iloc[0]["match_status"] == "resolved", f"{name} did not resolve: {row.iloc[0]['match_status']}"
            assert pd.notna(row.iloc[0]["player_id"])

    def test_chad_warner_resolved_via_match_level_disambiguation(self):
        """Chad Warner is the deliberately hard case: same surname/first-initial/
        team as a real teammate (Corey Warner). Season-level resolution alone
        must leave him ambiguous; match-level (per-round) resolution must
        resolve him in the rounds where no genuine collision exists."""
        from src.external.wheelo_source import load_wheelo_match_level
        match_level = load_wheelo_match_level()
        chad_rows = match_level[match_level["wheelo_player_name"] == "Chad Warner"]
        assert (chad_rows["match_status"] == "resolved").sum() > 0, "Chad Warner never resolves at match level"
        assert (chad_rows["match_status"] == "ambiguous").sum() > 0, (
            "expected at least one genuinely ambiguous round (both Warners active)"
        )

    def test_no_forced_ambiguous_match(self):
        canonical = load_canonical_players()
        fake_external = pd.DataFrame({"name": ["Chad Warner"], "team_id": ["sydney"]})
        resolved = resolve_external_players(fake_external, name_col="name", team_col="team_id", canonical=canonical)
        assert resolved.iloc[0]["match_status"] == "ambiguous"
        assert pd.isna(resolved.iloc[0]["player_id"]), "ambiguous match must never be force-resolved"

    def test_normalisation_helpers(self):
        assert _normalise_full_surname("Darcy Byrne-Jones") == "byrnejones"
        assert _normalise_full_surname("Nick Daicos") == "daicos"
        assert _first_initial("Chad Warner") == "c"
        assert _first_initial("Corey Warner") == "c"

    def test_noid_placeholder_rows_excluded_from_canonical(self):
        canonical = load_canonical_players()
        assert not canonical["player_id"].str.startswith("NOID").any()

    def test_canonical_player_id_unique(self):
        canonical = load_canonical_players()
        assert not canonical["player_id"].duplicated().any()


class TestWheeloIntegration:
    def test_round_numbering_matches_app_official_round(self):
        """Verified finding, not an assumption: Wheelo's own Round column for
        two real, distinct 2026 matches equals this app's official
        normalized round for the same match."""
        w = pd.read_csv(ROOT / "data" / "external" / "wheelo-brownlow-predictions.csv")
        sc = w[w["Match"] == "Syd v Carl"]
        assert set(sc["Round"].unique()) == {0}
        gws_carl = w[w["Match"] == "GWS v Carl"]
        assert set(gws_carl["Round"].unique()) == {15}

    def test_wheelo_season_totals_equal_sum_of_match_level(self):
        from src.external.wheelo_source import load_wheelo_match_level, load_wheelo_season
        match_level = load_wheelo_match_level()
        season = load_wheelo_season()
        resolved = match_level[match_level["match_status"] == "resolved"]
        for _, row in season.head(10).iterrows():
            own_matches = resolved[resolved["player_id"] == row["player_id"]]
            assert abs(own_matches["wheelo_ev"].sum() - row["wheelo_ev"]) < 1e-9

    def test_p3_probability_is_percentage_scale(self):
        from src.external.wheelo_source import load_wheelo_match_level
        match_level = load_wheelo_match_level()
        assert match_level["wheelo_p3_pct"].max() <= 100.0
        assert match_level["wheelo_p3_pct"].min() >= 0.0

    def test_no_duplicate_player_match_records(self):
        from src.external.wheelo_source import load_wheelo_match_level
        match_level = load_wheelo_match_level()
        resolved = match_level[match_level["match_status"] == "resolved"]
        dupes = resolved.duplicated(subset=["round", "player_id", "match_label"])
        assert not dupes.any()


class TestAggregation:
    def test_production_values_exactly_match_leaderboard(self):
        from src.external.aggregate import build_external_overview
        overview = build_external_overview()
        leaderboard = pd.read_csv(ROOT / "reports" / "2026_leaderboard.csv")
        leaderboard["player_id"] = leaderboard["player_id"].astype(str)
        merged = overview.merge(
            leaderboard[["player_id", "FINAL_ENSEMBLE"]], on="player_id", suffixes=("", "_lb")
        )
        merged = merged.dropna(subset=["production_ev"])
        assert (merged["production_ev"] - merged["FINAL_ENSEMBLE"]).abs().max() < 1e-9

    def test_objective_values_exactly_match_leaderboard(self):
        from src.external.aggregate import build_external_overview
        overview = build_external_overview()
        obj_lb = pd.read_csv(ROOT / "reports" / "2026_objective_leaderboard.csv")
        obj_lb["player_id"] = obj_lb["player_id"].astype(str)
        obj_lb = obj_lb[~obj_lb["player_id"].str.startswith("NOID")]
        merged = overview.merge(obj_lb[["player_id", "objective_ev"]], on="player_id", suffixes=("", "_lb"))
        merged = merged.dropna(subset=["objective_ev"])
        assert (merged["objective_ev"] - merged["objective_ev_lb"]).abs().max() < 1e-9

    def test_missing_external_data_is_na_never_zero(self):
        from src.external.aggregate import build_external_overview
        overview = build_external_overview()
        insufficient = overview[overview["n_external_sources"] == 0]
        assert not insufficient.empty
        assert insufficient["external_consensus_ev"].isna().all()
        assert (insufficient["external_consensus_ev"] == 0).sum() == 0

    def test_wheelo_only_labelled_correctly(self):
        from src.external.aggregate import build_external_overview
        overview = build_external_overview()
        wheelo_only = overview[
            overview["wheelo_ev"].notna() & overview["espn_ev"].isna() & overview["betfair_ev"].isna()
        ]
        assert not wheelo_only.empty
        assert (wheelo_only["external_source_label"] == "Wheelo benchmark").all()

    def test_agreement_categories_are_from_the_documented_set(self):
        from src.external.aggregate import build_external_overview
        overview = build_external_overview()
        allowed = {
            "STRONG CONVERGENCE", "OUR MODELS AGREE / EXTERNAL DIFFERS",
            "PRODUCTION + EXTERNAL AGREE", "OBJECTIVE + EXTERNAL AGREE",
            "HIGH DISAGREEMENT", "INSUFFICIENT EXTERNAL DATA",
        }
        assert set(overview["agreement_category"].unique()) <= allowed

    def test_external_integration_cannot_mutate_production_or_objective(self, tmp_path):
        """Byte-identical proof, not an assumption: hash Production/Objective
        leaderboards before and after building the full external overview."""
        import hashlib
        from src.external.aggregate import build_external_overview

        prod_path = ROOT / "reports" / "2026_leaderboard.csv"
        obj_path = ROOT / "reports" / "2026_objective_leaderboard.csv"
        before = (hashlib.sha256(prod_path.read_bytes()).hexdigest(), hashlib.sha256(obj_path.read_bytes()).hexdigest())
        build_external_overview()
        after = (hashlib.sha256(prod_path.read_bytes()).hexdigest(), hashlib.sha256(obj_path.read_bytes()).hexdigest())
        assert before == after


class TestESPNAndBetfair:
    def test_espn_round_columns_align_with_official_round_labels(self):
        from src.external.espn_source import ROUND_LABEL_TO_OFFICIAL
        assert ROUND_LABEL_TO_OFFICIAL["OR"] == 0
        assert ROUND_LABEL_TO_OFFICIAL["R1"] == 1
        assert ROUND_LABEL_TO_OFFICIAL["R24"] == 24

    def test_betfair_season_has_no_duplicate_player_rows(self):
        from src.external.betfair_source import load_betfair_season
        season = load_betfair_season()
        if season.empty:
            pytest.skip("no cached Betfair snapshot in this environment")
        assert not season["player_id"].duplicated().any()

    def test_no_source_parser_overwrites_another(self):
        """The overview join keeps wheelo_ev/espn_ev/betfair_ev as distinct
        columns end to end -- a bug that let one source's loader clobber
        another's column would show up as two of these being identical
        across every row, which real independent sources would not be."""
        from src.external.aggregate import build_external_overview
        overview = build_external_overview()
        both = overview.dropna(subset=["wheelo_ev", "espn_ev"])
        if len(both) > 5:
            assert not (both["wheelo_ev"] == both["espn_ev"]).all()


class TestPagesRenderWithRealData:
    def test_all_four_pages_render(self):
        from streamlit.testing.v1 import AppTest
        for f in [
            "pages/19_External_Overview.py", "pages/20_External_Player_Comparison.py",
            "pages/21_External_Winning_Order.py", "pages/22_External_Leader_After_Round.py",
        ]:
            at = AppTest.from_file(str(ROOT / f))
            at.run()
            assert not at.exception, f"{f} raised: {at.exception}"
