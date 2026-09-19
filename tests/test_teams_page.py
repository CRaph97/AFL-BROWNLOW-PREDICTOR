"""
Targeted tests for pages/30_Teams.py -- the new cross-model Teams MAIN page
(Production / Objective / Wheelo shown side by side, never blended).
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGE = str(ROOT / "pages" / "30_Teams.py")


def _select_team(at: AppTest, team_label_substr: str) -> AppTest:
    sel = at.selectbox[0]
    match = next((o for o in sel.options if team_label_substr in str(o)), None)
    assert match is not None, f"no team option contains {team_label_substr!r}: {sel.options}"
    return sel.set_value(match).run()


def test_page_loads_with_default_team_no_exception():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    assert not at.exception


def test_hawthorn_renders_and_gunston_disagreement_visible():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    at = _select_team(at, "Hawthorn")
    assert not at.exception

    tables = [df.value for df in at.dataframe if "Player" in df.value.columns]
    assert tables, "no player table rendered"
    player_table = tables[0]
    gunston = player_table[player_table["Player"] == "Jack Gunston"]
    assert not gunston.empty, "Jack Gunston not found in Hawthorn's team table"
    row = gunston.iloc[0]
    assert row["Agreement"] == "DIVERGENT"
    prod_ev = float(row["Prod EV"])
    obj_ev = float(row["Obj EV"])
    assert abs(prod_ev - obj_ev) > 5, "Gunston's well-documented Production/Objective gap should be large"
    assert prod_ev != obj_ev, "Production and Objective EVs must be shown separately, never blended"


def test_adelaide_renders_without_exception():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    at = _select_team(at, "Adelaide")
    assert not at.exception
    tables = [df.value for df in at.dataframe if "Player" in df.value.columns]
    assert tables and not tables[0].empty


def test_top_summary_metrics_kept_separate_per_model():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    at = _select_team(at, "Hawthorn")
    labels = [m.label for m in at.metric]
    assert "Production team total EV" in labels
    assert "Objective team total EV" in labels
    assert "Wheelo team total EV" in labels
    assert "Model spread (max-min total)" in labels


def test_interpretation_bullets_present_and_bounded():
    at = AppTest.from_file(PAGE, default_timeout=90)
    at.run()
    at = _select_team(at, "Hawthorn")
    bullets = [m.value for m in at.markdown if m.value.strip().startswith("- ")]
    assert 1 <= len(bullets) <= 5


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
