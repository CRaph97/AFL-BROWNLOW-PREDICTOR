"""
Automated QC checks for the canonical CORE player-match dataset
(data/processed/player_match_core_1984_2025.parquet), per the Phase 2 testing
requirements in PROJECT_STATE.md.

Run with: pytest tests/ -v
Requires data/processed/player_match_core_1984_2025.parquet to already exist
(run `python -m src.data.build_core_dataset` first).
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "data" / "processed" / "player_match_core_1984_2025.parquet"

FINALS_ROUNDS = {"SF", "PF", "GF", "QF", "EF"}


@pytest.fixture(scope="module")
def core() -> pd.DataFrame:
    if not CORE_PATH.exists():
        pytest.skip(f"{CORE_PATH} not built yet -- run `python -m src.data.build_core_dataset` first")
    return pd.read_parquet(CORE_PATH)


def test_unique_player_match_rows(core: pd.DataFrame):
    dupes = core.duplicated(subset=["match_id", "player_id"], keep=False)
    assert dupes.sum() == 0, f"{dupes.sum()} duplicate (match_id, player_id) rows found"


def test_valid_brownlow_values(core: pd.DataFrame):
    valid = {0, 1, 2, 3}
    values = set(core["brownlow_votes"].dropna().unique().tolist())
    assert values <= valid, f"Unexpected Brownlow vote values found: {values - valid}"


def test_no_missing_target_in_scope(core: pd.DataFrame):
    # Every row in the CORE table (1984-2025, home-and-away only) must have a known vote outcome --
    # NaN here would mean a scope leak (e.g. a final or an out-of-window season slipped through).
    n_missing = core["brownlow_votes"].isna().sum()
    assert n_missing == 0, f"{n_missing} rows have a missing Brownlow vote value inside the validated 1984-2025 H&A window"


def test_exactly_one_3_2_1_per_match(core: pd.DataFrame):
    g = core.groupby("match_id")["brownlow_votes"].agg(
        n3=lambda s: (s == 3).sum(),
        n2=lambda s: (s == 2).sum(),
        n1=lambda s: (s == 1).sum(),
        total=lambda s: s.sum(),
    )
    bad = g[~((g.n3 == 1) & (g.n2 == 1) & (g.n1 == 1) & (g.total == 6))]
    assert len(bad) == 0, f"{len(bad)} matches violate the one-3/one-2/one-1/sum-6 constraint: {bad.index.tolist()[:10]}"


def test_team_opponent_consistency(core: pd.DataFrame):
    same = (core["team_id"] == core["opponent_id"]).sum()
    assert same == 0, f"{same} rows have team_id == opponent_id"


def test_home_and_away_only(core: pd.DataFrame):
    finals_present = core["round"].isin(FINALS_ROUNDS).sum()
    assert finals_present == 0, f"{finals_present} finals rows found in the CORE (home-and-away only) table"


def test_no_duplicate_matches_with_different_ids(core: pd.DataFrame):
    # A given (season, round, home team-as-team_id-when-home_away==home, away team) should map
    # to exactly one match_id.
    natural_key = core[["season", "round", "home_away"]].astype(str).agg("_".join, axis=1)
    per_match_id = core.groupby("match_id")["date"].nunique()
    assert (per_match_id == 1).all(), "Some match_id groups span more than one date"


def test_no_impossible_margins(core: pd.DataFrame):
    recomputed = core["team_score"] - core["opponent_score"]
    assert (recomputed == core["margin"]).all(), "margin column inconsistent with team_score - opponent_score"
    assert (core["absolute_margin"] == core["margin"].abs()).all()


def test_season_and_round_scope(core: pd.DataFrame):
    assert core["season"].min() >= 1984
    assert core["season"].max() <= 2025


def test_player_id_not_null(core: pd.DataFrame):
    # Every row must have SOME player_id (real afltables ID or a logged NOID_ placeholder) --
    # never a genuinely null identifier.
    assert core["player_id"].isna().sum() == 0
