"""
Regression tests for the Phase 5 round-label integrity bug and its fix.

Root cause: afltables' raw `Round` column for the 2026 season labels the split Opening
Round as Round "1" and then numbers every subsequent round sequentially with no marker,
while the AFL's own official numbering treats the Opening Round as unnumbered and starts
official Round 1 the following week -- shifting every raw round from 2 onward by +1
relative to the official label. Confirmed and fixed in src/data/round_normalization_2026.py
and build_2026_extension.py; full reconciliation in reports/2026_match_identity_audit.csv.

These tests pin the two user-confirmed examples and the season-wide invariants so a
regression (e.g. someone reverting the fix, or a future season reintroducing an
un-normalised round) is caught immediately.
"""
import pandas as pd
import pytest

from src.data.round_normalization_2026 import official_round, official_round_label, fix_match_id

REPORTS = "reports"
PROCESSED = "data/processed"


@pytest.fixture(scope="module")
def identity_audit():
    return pd.read_csv(f"{REPORTS}/2026_match_identity_audit.csv")


@pytest.fixture(scope="module")
def match_probabilities():
    return pd.read_csv(f"{REPORTS}/2026_match_probabilities.csv")


@pytest.fixture(scope="module")
def model_core_2026():
    df = pd.read_parquet(f"{PROCESSED}/model_core_2026.parquet")
    return df[df["season"] == 2026]


def test_geelong_vs_collingwood_is_round_9(identity_audit):
    row = identity_audit[
        (identity_audit["official_date"] == "2026-05-09")
        & (identity_audit["official_home"] == "geelong")
        & (identity_audit["official_away"] == "collingwood")
    ]
    assert len(row) == 1
    assert int(row.iloc[0]["official_round"]) == 9
    assert int(row.iloc[0]["model_round"]) == 9
    assert int(row.iloc[0]["dashboard_round"]) == 9


def test_brisbane_vs_geelong_is_round_10(identity_audit):
    row = identity_audit[
        (identity_audit["official_date"] == "2026-05-14")
        & (identity_audit["official_home"] == "brisbane_lions")
        & (identity_audit["official_away"] == "geelong")
    ]
    assert len(row) == 1
    assert int(row.iloc[0]["official_round"]) == 10
    assert int(row.iloc[0]["model_round"]) == 10
    assert int(row.iloc[0]["dashboard_round"]) == 10


def test_shaun_mannagh_brisbane_geelong_stat_line(model_core_2026):
    row = model_core_2026[
        (model_core_2026["date"].astype(str) == "2026-05-14")
        & (model_core_2026["player_name"].str.contains("Mannagh", case=False, na=False))
    ]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["disposals"] == 30
    assert r["goals"] == 5
    assert r["goal_assists"] == 3
    assert r["round"] in ("10", 10)
    assert r["team_id"] == "geelong"
    assert r["opponent_id"] == "brisbane_lions"


def test_official_round_mapping_matches_confirmed_examples():
    assert official_round("10") == 9
    assert official_round("11") == 10
    assert official_round("1") == 0  # Opening Round
    assert official_round_label(0) == "Opening Round"
    assert official_round_label(9) == "Round 9"


def test_fix_match_id_rewrites_only_the_round_segment():
    assert fix_match_id("2026_R11_brisbane_lions_v_geelong_2026-05-14") == \
        "2026_R10_brisbane_lions_v_geelong_2026-05-14"
    assert fix_match_id("2026_R10_geelong_v_collingwood_2026-05-09") == \
        "2026_R9_geelong_v_collingwood_2026-05-09"
    # fix_match_id is now season-aware (see docs/ROUND_NORMALIZATION.md): seasons before 2024
    # never had an unnumbered Opening Round, so their match_ids are left untouched -- this is
    # correct, not a limitation. 2018 (pre-Opening-Round) must be an identity transform.
    assert fix_match_id("2018_R14_richmond_v_collingwood_2018-07-07") == \
        "2018_R14_richmond_v_collingwood_2018-07-07"
    # 2023 also needs no shift (independently confirmed against footywire's own round
    # numbering -- 2023's Opening Round was itself labelled "Round 1" by both sources).
    assert fix_match_id("2023_R9_geelong_v_collingwood_2023-05-12") == \
        "2023_R9_geelong_v_collingwood_2023-05-12"
    # 2024 and 2025 DO shift, by the same pattern as 2026 (confirmed against footywire).
    assert fix_match_id("2024_R10_adelaide_v_brisbane_lions_2024-05-12") == \
        "2024_R9_adelaide_v_brisbane_lions_2024-05-12"


def test_all_207_matches_reconciled(identity_audit):
    assert len(identity_audit) == 207
    assert identity_audit["canonical_match_id"].nunique() == 207
    assert identity_audit["identity_match"].all()
    assert identity_audit["round_match"].all()
    assert identity_audit["score_match"].all()
    assert identity_audit["prediction_match"].all()


def test_no_2026_match_appears_under_the_wrong_round(identity_audit, match_probabilities):
    mp_round = match_probabilities.drop_duplicates(subset=["match_id"])[["match_id", "round"]]
    merged = identity_audit.merge(mp_round, left_on="canonical_match_id", right_on="match_id")
    assert (merged["round"].astype(int) == merged["official_round"].astype(int)).all()


def test_prediction_match_id_equals_stats_match_id(model_core_2026, match_probabilities):
    core_ids = set(model_core_2026["match_id"])
    pred_ids = set(match_probabilities["match_id"])
    assert pred_ids <= core_ids, "every prediction match_id must exist in the underlying stats table"


def test_no_duplicate_or_missing_2026_matches(model_core_2026):
    match_meta = model_core_2026.drop_duplicates(subset=["match_id"])
    assert match_meta["match_id"].is_unique
    assert len(match_meta) == 207


def test_player_opponent_team_consistent_with_match_metadata(model_core_2026):
    bad = model_core_2026[model_core_2026["team_id"] == model_core_2026["opponent_id"]]
    assert len(bad) == 0
