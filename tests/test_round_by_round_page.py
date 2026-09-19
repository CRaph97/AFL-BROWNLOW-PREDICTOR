"""
Targeted tests for pages/31_Round_By_Round.py -- the new cross-model
Round-by-Round MAIN page (Production / Objective / Wheelo cumulative
leaderboard, never blended into one score).
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "pages" / "31_Round_By_Round.py")


def _leaderboard(at: AppTest):
    tables = [df.value for df in at.dataframe if "Rank" in df.value.columns and "Player" in df.value.columns]
    assert tables, "no cumulative leaderboard rendered"
    return tables[0]


def test_page_loads_at_default_final_round_no_exception():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    assert not at.exception
    assert at.slider, "expected a round slider"


def test_early_round_r0_renders_real_data():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    sl = at.slider[0]
    sl.set_value(float(sl.min))
    at.run()
    assert not at.exception
    board = _leaderboard(at)
    assert not board.empty
    assert board.iloc[0]["Player"] == "Christian Petracca"


def test_final_round_renders_and_daicos_smith_bontempelli_ordering_visible():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    sl = at.slider[0]
    sl.set_value(float(sl.max))
    at.run()
    assert not at.exception
    board = _leaderboard(at)
    names = set(board["Player"])
    for expected in ("Nick Daicos", "Bailey Smith", "Marcus Bontempelli"):
        assert expected in names, f"{expected} not in the R{sl.max} leaderboard's visible top rows"

    daicos = board[board["Player"] == "Nick Daicos"].iloc[0]
    # Each model's own cumulative EV/rank must be shown separately -- never
    # collapsed into one blended figure.
    assert daicos["Prod cum EV"] != daicos["Obj cum EV"]
    assert daicos["Prod rank"] == "#1"


def test_summary_bullets_present_and_bounded():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    bullets = [m.value for m in at.markdown if m.value.strip().startswith("- ")]
    assert 1 <= len(bullets) <= 5


def test_trajectory_chart_supports_one_to_five_players():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    ms = at.multiselect[0]
    assert ms.max_selections == 5
    ms.set_value(["Nick Daicos"]).run()
    assert not at.exception


def test_no_model_data_or_simulation_files_changed():
    import subprocess

    diff = subprocess.run(
        ["git", "diff", "--stat", "reports/2026_leaderboard.csv", "reports/2026_objective_leaderboard.csv",
         "reports/2026_predicted_votes.csv", "reports/2026_objective_votes.csv",
         "data/betting/processed/priced_opportunities.csv", "data/processed/mc_totals_2026.npy",
         "data/processed/mc_totals_objective_2026.npy"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert diff.stdout.strip() == ""
