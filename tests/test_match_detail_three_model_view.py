"""
Regression tests for the Match Detail page's added Production/Objective/
Wheelo combined per-match table.

Audit finding this responds to: before this change, pages/6_Match_Detail.py
imported only dashboard.data (Production) -- zero references to Objective or
Wheelo anywhere in the file. This adds a combined view without touching any
model/data file or any other section of the page.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_match_detail_page_renders_with_all_three_models():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages" / "6_Match_Detail.py"), default_timeout=60)
    at.run()
    assert not at.exception
    combined_tables = [df.value for df in at.dataframe if "Prod EV" in df.value.columns]
    assert combined_tables, "combined Production/Objective/Wheelo table did not render"
    cols = set(combined_tables[0].columns)
    assert {"Prod EV", "Prod P3", "Prod P2", "Prod P1"} <= cols
    assert {"Obj EV", "Obj P3", "Obj P2", "Obj P1"} <= cols
    assert {"Wheelo Votes", "Wheelo P3%"} <= cols
    # Never a fabricated Wheelo P2/P1/P(any).
    assert not any("Wheelo P2" in c or "Wheelo P1" in c or "Wheelo P(any" in c for c in cols)


def test_match_ev_equals_3p3_plus_2p2_plus_p1():
    import dashboard.data as d

    mp = d.load_match_probabilities()
    sample = mp.dropna(subset=["p3", "p2", "p1", "expected_votes"]).head(50)
    computed = 3 * sample["p3"] + 2 * sample["p2"] + sample["p1"]
    assert (computed - sample["expected_votes"]).abs().max() < 1e-6

    obj = d.load_objective_votes()
    sample_obj = obj.dropna(subset=["p3", "p2", "p1", "expected_votes"]).head(50)
    computed_obj = 3 * sample_obj["p3"] + 2 * sample_obj["p2"] + sample_obj["p1"]
    assert (computed_obj - sample_obj["expected_votes"]).abs().max() < 1e-6


def test_no_model_or_data_files_changed():
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--stat", "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv",
         "reports/2026_match_probabilities.csv", "reports/2026_objective_votes.csv"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert diff.stdout.strip() == ""
