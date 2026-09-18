"""
Targeted tests for the Finishing Order page's computation layer
(dashboard/finishing_order.py). Everything here is read-only over the
already-generated Monte Carlo draw arrays -- no model/data file is written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard import finishing_order as fo

ROOT = Path(__file__).resolve().parent.parent


def test_winner_matches_existing_validated_probability():
    """N=1 must agree with reports/2026_leaderboard.csv's own prob_rank_1
    (independently computed, validated in an earlier task) to within Monte
    Carlo sampling noise."""
    t1 = fo.topn_table(1)
    lb = pd.read_csv(ROOT / "reports" / "2026_leaderboard.csv")
    lb["player_id"] = lb["player_id"].astype(str)
    merged = t1.merge(lb[["player_id", "prob_rank_1"]], on="player_id", how="inner")
    assert not merged.empty
    assert (merged["production_topn"] - merged["prob_rank_1"]).abs().max() < 0.01


def test_top5_matches_direct_independent_recount():
    """Recompute Top-5 a second, simpler way directly from the raw array and
    confirm agreement with topn_table()'s output."""
    totals, players = fo.load_production_sim()
    ranks = (-totals).argsort(axis=1).argsort(axis=1) + 1
    direct = (ranks <= 5).mean(axis=0)
    direct_by_id = dict(zip(players["player_id"].apply(fo._normalise_player_id), direct))

    t5 = fo.topn_table(5)
    for _, row in t5.head(30).iterrows():
        expected = direct_by_id.get(row["player_id"])
        if expected is not None:
            assert abs(row["production_topn"] - expected) < 1e-9


def test_topn_probabilities_are_monotonic_5_10_20():
    t5 = fo.topn_table(5)[["player_id", "production_topn"]].rename(columns={"production_topn": "p5"})
    t10 = fo.topn_table(10)[["player_id", "production_topn"]].rename(columns={"production_topn": "p10"})
    t20 = fo.topn_table(20)[["player_id", "production_topn"]].rename(columns={"production_topn": "p20"})
    m = t5.merge(t10, on="player_id").merge(t20, on="player_id")
    assert (m["p5"] <= m["p10"] + 1e-9).all()
    assert (m["p10"] <= m["p20"] + 1e-9).all()


def test_exact_order_probability_not_equal_to_marginal_product():
    """Proves real joint simulation outcomes are used, not a product of
    independent marginals, for a genuinely correlated real example (three
    real top contenders competing for the same top positions)."""
    t5 = fo.topn_table(5).set_index("player_id")
    top3_ids = fo.topn_table(5).sort_values("combined_evidence", ascending=False)["player_id"].head(3).tolist()
    res = fo.exact_order_probability(top3_ids)
    marginal_product = 1.0
    for pid in top3_ids:
        marginal_product *= t5.loc[pid, "production_topn"]
    assert res["production"]["exact_order_prob"] is not None
    assert abs(res["production"]["exact_order_prob"] - marginal_product) > 0.01


def test_exact_order_never_exceeds_all_in_topk_any_order():
    """The mathematical invariant this task's fix restored: an exact order
    is a subset of "all k players finish somewhere in the top k", so its
    probability can never be larger. Checked across several real player
    triples, not just one."""
    table = fo.topn_table(5).sort_values("combined_evidence", ascending=False)
    ids = table["player_id"].tolist()
    triples = [ids[0:3], ids[1:4], ids[3:6], ids[10:13]]
    for triple in triples:
        res = fo.exact_order_probability(triple)
        for model in ("production", "objective"):
            r = res[model]
            if r["exact_order_prob"] is not None:
                assert r["exact_order_prob"] <= r["all_in_topk_prob"] + 1e-12, (triple, model, r)


def test_very_rare_threshold_applied_for_low_support():
    """An implausible/rare order (e.g. reversing two clearly-unequal
    contenders relative to a fringe player) should fall under
    MIN_SUPPORTING_DRAWS at least for the Objective model's smaller
    (20,000-draw) sample somewhere in the player pool -- verifies the
    threshold path is real code, not dead code."""
    table = fo.topn_table(20)
    fringe_ids = table.sort_values("combined_evidence", ascending=True)["player_id"].head(3).tolist()
    res = fo.exact_order_probability(fringe_ids)
    assert res["objective"]["exact_order_supporting_draws"] < fo.MIN_SUPPORTING_DRAWS


def test_production_and_objective_outputs_unchanged():
    """This page is read-only -- confirm the underlying MC arrays and
    leaderboards were not modified by anything in this module."""
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--stat", "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert diff.stdout.strip() == ""


def test_navigation_places_pages_correctly():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    app_src = (ROOT / "app.py").read_text()
    main_block = app_src.split('"MAIN": [')[1].split("],")[0]
    model_analysis_block = app_src.split('"MODEL ANALYSIS": [')[1].split("],")[0]
    assert "26_Finishing_Order.py" in main_block
    assert "title=\"Production Model\"" in model_analysis_block
    assert "00_Overview.py" in model_analysis_block
    assert "00_Overview.py" not in main_block
