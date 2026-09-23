"""
Targeted tests for the 2026 post-Brownlow evaluation
(src/evaluation/settlement.py, src/evaluation/build_2026_evaluation.py,
data/evaluation/2026/*, pages/33_2026_Evaluation.py). Offline.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import settlement as stl
from src.evaluation.build_2026_evaluation import (
    COUNT_DATE_UTC, FROZEN_INPUTS, OUT, ROOT, calibration_bins, frozen_input_hashes, likelihood_band,
)

EVAL = OUT


@pytest.fixture(scope="module")
def manifest():
    return json.loads((EVAL / "manifest.json").read_text())


@pytest.fixture(scope="module")
def season():
    return pd.read_csv(EVAL / "season_players.csv", dtype={"player_id": str})


@pytest.fixture(scope="module")
def settled():
    return pd.read_csv(EVAL / "betting_settled.csv")


@pytest.fixture(scope="module")
def match_table():
    return pd.read_csv(EVAL / "match_table.csv", dtype={"actual_3_player_id": str, "model_top_player_id": str})


@pytest.fixture(scope="module")
def actual_lb():
    return pd.read_csv(ROOT / "data" / "actual" / "2026_brownlow_leaderboard.csv", dtype={"player_id": str})


# ------------------------------------------------------------------ settlement rules
class TestSettlement:
    def test_top_n_win_loss_and_dead_heat(self):
        assert stl.settle_top_n(3, 1, 5).result == "win"
        assert stl.settle_top_n(6, 1, 5).result == "loss"
        dh = stl.settle_top_n(5, 3, 5)              # 3 tied for 5th, one place left
        assert dh.result == "dead_heat" and abs(dh.fraction - 1 / 3) < 1e-9
        dh2 = stl.settle_top_n(4, 3, 5)             # 3 tied for 4th, two places left
        assert dh2.result == "dead_heat" and abs(dh2.fraction - 2 / 3) < 1e-9
        assert stl.settle_top_n(4, 2, 5).result == "win"  # both tied players fit
        assert stl.settle_top_n(None, 1, 5).result == "unsettleable"
        assert stl.settle_top_n(pd.NA, 1, 5).result == "unsettleable"

    def test_winner_and_exact_position(self):
        assert stl.settle_winner(1, 1).result == "win"
        assert stl.settle_winner(2, 1).result == "loss"
        assert stl.settle_winner(1, 2).result == "dead_heat"
        assert stl.settle_exact_position(4, 1, 4).result == "win"
        assert stl.settle_exact_position(5, 3, 5).result == "dead_heat"
        assert stl.settle_exact_position(5, 3, 6).result == "dead_heat"   # tie spans 5-7
        assert stl.settle_exact_position(5, 3, 8).result == "loss"
        assert stl.settle_exact_position(3, 1, 4).result == "loss"

    def test_over_under_and_push(self):
        assert stl.settle_over_under(28, 22.5, "over").result == "win"
        assert stl.settle_over_under(20, 22.5, "over").result == "loss"
        assert stl.settle_over_under(20, 22.5, "under").result == "win"
        assert stl.settle_over_under(22, 22.0, "over").result == "push"
        assert stl.settle_over_under(22, 22.0, "under").result == "push"
        assert stl.settle_over_under(22, None, "under").result == "unsettleable"
        assert stl.settle_over_under(22, 22.5, "sideways").result == "unsettleable"

    def test_threshold_and_h2h(self):
        assert stl.settle_threshold(25, 25).result == "win"
        assert stl.settle_threshold(24, 25).result == "loss"
        assert stl.settle_threshold(1, 1).result == "win"     # to poll a vote
        assert stl.settle_threshold(0, 1).result == "loss"
        assert stl.settle_h2h(10, 8).result == "win"
        assert stl.settle_h2h(8, 10).result == "loss"
        assert stl.settle_h2h(9, 9).result == "push"
        assert stl.settle_h2h(9, None).result == "unsettleable"

    def test_flat_unit_pnl_and_roi_math(self):
        assert stl.flat_unit_pnl(stl.Settlement("win", 1.0, ""), 2.5) == pytest.approx(1.5)
        assert stl.flat_unit_pnl(stl.Settlement("loss", 0.0, ""), 2.5) == -1.0
        assert stl.flat_unit_pnl(stl.Settlement("push", 0.0, ""), 2.5) == 0.0
        assert stl.flat_unit_pnl(stl.Settlement("dead_heat", 1 / 3, ""), 12.0) == pytest.approx(3.0)
        assert stl.flat_unit_pnl(stl.Settlement("win", 1.0, ""), None) is None
        assert stl.flat_unit_pnl(stl.Settlement("unsettleable", 0.0, ""), 2.0) is None
        s = stl.summarise_bets([1.5, -1.0, 0.0, 3.0, None], ["win", "loss", "push", "dead_heat", "unsettleable"])
        assert s["bets"] == 4 and s["wins"] == 2 and s["pushes"] == 1 and s["losses"] == 1
        assert s["hit_rate"] == pytest.approx(2 / 3)          # pushes excluded from denominator
        assert s["flat_unit_pnl"] == pytest.approx(3.5) and s["roi"] == pytest.approx(3.5 / 4)
        assert stl.implied_probability(4.0) == 0.25 and stl.implied_probability(None) is None

    def test_settled_rows_are_internally_consistent(self, settled):
        s = settled[settled["result"] != "unsettleable"]
        assert set(s["result"]) <= {"win", "loss", "push", "dead_heat"}
        assert (s.loc[s["result"] == "loss", "flat_unit_pnl"] == -1).all()
        assert (s.loc[s["result"] == "push", "flat_unit_pnl"] == 0).all()
        w = s[s["result"] == "win"]
        assert np.allclose(w["flat_unit_pnl"], w["odds"] - 1)
        dh = s[s["result"] == "dead_heat"]
        assert np.allclose(dh["flat_unit_pnl"], dh["odds"] * dh["dead_heat_fraction"] - 1)
        assert (settled.loc[settled["market_type"] == "UNMODELLED", "result"] == "unsettleable").all()
        # every selection with a captured odds and a resolved player in a modelled market is settled
        mod = settled[(settled["market_type"] != "UNMODELLED") & settled["player_id"].notna() | (settled["market_type"] == "TEAM_VOTES_OU")]
        assert (mod["result"] != "unsettleable").all()
        # both sides of every O/U pair cannot both win
        for mt in ("PLAYER_VOTES_OU", "TEAM_VOTES_OU"):
            g = s[s["market_type"] == mt].groupby(["source", "market_name"])["result"]
            assert (g.apply(lambda r: (r == "win").sum()) <= 1).all()
        # H2H pushes are genuine ties
        h = s[(s["market_type"] == "PLAYER_H2H") & (s["result"] == "push")]
        assert (h["actual_value"] == h["opponent_actual_votes"]).all()

    def test_known_2026_settlements(self, settled):
        s = settled[settled["result"] != "unsettleable"]
        daicos = s[(s["market_type"] == "WINNER") & (s["player_name"] == "Nick Daicos")]
        assert len(daicos) >= 1 and (daicos["result"] == "win").all()
        others = s[(s["market_type"] == "WINNER") & (s["player_name"] != "Nick Daicos")]
        assert (others["result"] == "loss").all()
        top5_tie = s[(s["market_type"] == "TOP_N") & (s["n"] == 5) & (s["player_name"].isin(["Patrick Cripps", "Will Ashcroft", "Harry Sheezel"]))]
        assert (top5_tie["result"] == "dead_heat").all() and np.allclose(top5_tie["dead_heat_fraction"], 1 / 3)
        gawn4 = s[(s["market_type"] == "EXACT_POSITION") & (s["player_name"] == "Max Gawn")]
        assert (gawn4["result"] == "win").all()


# ------------------------------------------------------------------ season totals / ranks / identity
def test_season_totals_and_ranks(season, actual_lb):
    assert season["player_id"].is_unique
    d = season[season["player_name"] == "Nick Daicos"].iloc[0]
    assert d["actual_votes"] == 47 and d["actual_rank"] == 1
    unmapped = int(actual_lb.loc[actual_lb["player_id"].isna(), "afl_total_actual_votes"].sum())  # Jack Ross, no canonical id
    assert unmapped == 3
    assert int(season["actual_votes"].sum()) == int(actual_lb["afl_total_actual_votes"].sum()) - unmapped
    # every resolved leaderboard player appears with the same total
    lb = actual_lb[actual_lb["player_id"].notna()].set_index("player_id")["afl_total_actual_votes"]
    got = season.set_index("player_id")["actual_votes"].reindex(lb.index)
    assert (got == lb).all()
    # min-method ranks: three players on 27 share rank 5, next rank is 8
    assert set(season.loc[season["actual_votes"] == 27, "actual_rank"]) == {5}
    assert season.loc[season["actual_votes"] == 26, "actual_rank"].iloc[0] == 8
    # errors are EV minus actual where the source covers the player
    for m in ("production", "objective", "wheelo"):
        cov = season[season[f"{m}_ev"].notna()]
        assert np.allclose(cov[f"{m}_error"], cov[f"{m}_ev"] - cov["actual_votes"])
        assert season.loc[season[f"{m}_ev"].isna(), f"{m}_error"].isna().all()


def test_player_identity_joins(season, actual_lb):
    unresolved = actual_lb[actual_lb["player_id"].isna()]
    assert unresolved["player_name"].tolist() == ["Jack Ross"]
    assert not season["player_id"].str.startswith("NOID").any()
    # a player covered by every source has one team across sources
    assert (season["in_all_three"].sum()) >= 500


def test_match_321_joins(match_table):
    assert match_table.groupby("model")["match_id"].nunique().eq(207).all()
    assert len(match_table) == 207 * 3
    for _, g in match_table.groupby("match_id"):
        assert g["actual_3_player"].nunique() == 1
    assert (match_table["n_vote_getters_in_pred_top3"] <= 3).all()
    # a hit means the top pick IS the actual 3-voter and the 3-voter's P3 rank is 1
    hit = match_table[match_table["hit_3"]]
    assert (hit["model_top_player_id"] == hit["actual_3_player_id"]).all() and (hit["actual_3_p3_rank"] == 1).all()
    assert (match_table["exact_321"] <= match_table["unordered_top3"]).all()
    # The only roster miss is Production's documented first-game (season-to-date) exclusion:
    # Jagga Smith, Carlton, Round 1. Objective and Wheelo rosters contain every 3-voter.
    miss = match_table[~match_table["actual_3_in_roster"]]
    assert set(miss["model"]) <= {"Production"} and len(miss) == 1
    assert miss["actual_3_player"].tolist() == ["Jagga Smith"] and (miss["round"] == 1).all()
    assert miss["log_loss_p3"].isna().all() and miss["brier_p3"].isna().all()


# ------------------------------------------------------------------ calibration denominators
def test_calibration_bins_denominators():
    p = pd.Series([0.05, 0.15, 0.15, 0.95, 0.95, 0.5])
    y = pd.Series([0, 1, 0, 1, 1, 0])
    table, summary = calibration_bins(p, y, "s", "m")
    assert table["n"].sum() == 6 and summary["n"] == 6
    assert table.loc[table["bin"] == "10-20%", "n"].iloc[0] == 2
    assert table.loc[table["bin"] == "10-20%", "actual_frequency"].iloc[0] == 0.5
    assert table.loc[table["bin"] == "90-100%", "n"].iloc[0] == 2
    assert summary["brier"] == pytest.approx(float(((p - y) ** 2).mean()), abs=1e-5)
    assert 0 <= summary["ece"] <= 1
    t2, s2 = calibration_bins(pd.Series([np.nan, 0.2]), pd.Series([1, 1]), "s", "m")
    assert s2["n"] == 1  # NaN predictions dropped from every denominator
    bins = pd.read_csv(EVAL / "calibration_bins.csv"); summ = pd.read_csv(EVAL / "calibration_summary.csv")
    tot = bins.groupby(["series", "model"])["n"].sum().reset_index().merge(summ, on=["series", "model"])
    assert (tot["n_x"] == tot["n_y"]).all()
    assert likelihood_band(0.9) == "Very High" and likelihood_band(0.1) == "Very Low" and likelihood_band(None) == "N/A"


# ------------------------------------------------------------------ integrity
def test_no_frozen_outputs_changed(manifest):
    assert manifest["frozen_inputs_unchanged_during_build"] is True
    now = frozen_input_hashes()
    assert now == manifest["frozen_input_hashes"], {k: v for k, v in now.items() if manifest["frozen_input_hashes"].get(k) != v}
    diff = subprocess.run(["git", "diff", "--stat", "--", "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv",
                           "reports/2026_predicted_votes.csv", "reports/2026_objective_votes.csv", "data/betting/processed/priced_opportunities.csv",
                           "data/actual/2026_brownlow_match_votes.csv"], cwd=ROOT, capture_output=True, text=True)
    assert diff.stdout.strip() == ""


def test_no_post_event_leakage(manifest):
    assert manifest["all_bookmaker_snapshots_pre_count"] is True
    assert all(t < COUNT_DATE_UTC for t in manifest["bookmaker_snapshots_retrieved_at"])
    statuses = json.loads((ROOT / "data" / "betting" / "processed" / "refresh_summary.json").read_text())["source_statuses"]
    assert all(s["retrieved_at"] < COUNT_DATE_UTC for s in statuses)
    # the evaluation code never writes outside data/evaluation and never imports the pricing/sim writers
    src = (ROOT / "src" / "evaluation" / "build_2026_evaluation.py").read_text()
    assert "to_csv(OUT" in src and "to_csv(REPORTS" not in src and "to_parquet" not in src
    assert "refresh_brownlow_odds" not in src and "run_2026_montecarlo" not in src
    # feature/model code does not read the actual votes or the evaluation outputs
    hits = subprocess.run(["grep", "-rIl", "-e", "data/actual", "-e", "data/evaluation", "-e", "src.evaluation.build_2026",
                           str(ROOT / "src" / "features"), str(ROOT / "src" / "models"), str(ROOT / "src" / "data")],
                          capture_output=True, text=True).stdout.split()
    assert hits == [], hits


def test_scorecard_and_learnings_shape():
    sc = pd.read_csv(EVAL / "scorecard.csv")
    assert set(sc["model"]) == {"Production", "Objective", "Wheelo"} and len(sc) == 6
    assert (sc["winner_predicted_correctly"]).all()
    ms = pd.read_csv(EVAL / "match_scorecard.csv")
    assert (ms["n_matches"] == 207).all() and (ms["hit_3_rate"].between(0, 1)).all()
    learn = json.loads((EVAL / "learnings.json").read_text())
    assert 8 <= len(learn) <= 12 and all({"title", "finding", "evidence", "n"} <= set(l) for l in learn)


# ------------------------------------------------------------------ page
def test_evaluation_page_renders_and_is_in_main_nav():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages" / "33_2026_Evaluation.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "2026 Evaluation"
    router = (ROOT / "app.py").read_text()
    main_block = router[router.index('"MAIN": ['):router.index("],", router.index('"MAIN": ['))]
    titles = [line.split('title="')[1].split('"')[0] for line in main_block.splitlines() if 'title="' in line]
    assert titles.index("2026 Evaluation") == titles.index("Guide & FAQs") + 1
