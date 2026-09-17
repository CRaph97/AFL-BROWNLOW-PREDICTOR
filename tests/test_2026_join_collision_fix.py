"""
Regression tests for a live collision bug introduced by the earlier hyphenated-
surname fix in src.data.build_2026_extension.

The earlier fix (see tests/test_hyphenated_surname_join.py) made the ADVANCED
join key collapse a hyphenated surname to the segment after its last hyphen
(e.g. "Byrne-Jones" -> "jones"), to match footywire's abbreviated spelling
("B-Jones" -> "jones"). But the join key never included the player's given
name, so any teammate who happened to share that plain surname collided:
"Darcy Byrne-Jones" and "Lachie Jones" (both Port Adelaide) both normalised to
surname_key="jones", so on any shared match date the join matched BOTH of them
to whichever single footywire row came first -- silently duplicating one
player's real stats onto the other's identity. Confirmed live in
data/processed/model_advanced_2026.parquet: Byrne-Jones and Lachie Jones held
byte-identical effective_disposals/metres_gained on every shared 2026 date
(e.g. both 28.0/450.0 on 2026-06-20); the same pattern existed for
Tom Brown / Oliver Hayes-Brown (Richmond).

Fix: also require a first_name_key (the normalised first token) to agree.
footywire never abbreviates the given name, even when it abbreviates the
surname (confirmed against raw 2026 data: "Darcy B-Jones", "Lachlan Jones",
"Nasiah W-Milera", "Oliver H-Brown" all carry a full first name), so this
disambiguates the collision without needing any player-specific exception and
without breaking the original 12-player hyphenated-surname fix.

These tests run build_core_2026()/build_advanced_2026() directly against the
real raw data and assert on the in-memory result -- they do NOT write to
data/processed/ or any report file (no rebuild has been run/authorized yet).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_2026_extension import build_advanced_2026, build_core_2026, _normalise_first_name, _normalise_surname

ROOT = Path(__file__).resolve().parent.parent
RAW_AVAILABLE = (ROOT / "data" / "raw" / "fitzroy_data" / "player_stats.rda").exists()

pytestmark = pytest.mark.skipif(not RAW_AVAILABLE, reason="raw fitzRoy data not present in this environment")


@pytest.fixture(scope="module")
def merged_2026():
    core_2026, _ = build_core_2026()
    merged, validation = build_advanced_2026(core_2026)
    return merged, validation


def _row(merged: pd.DataFrame, player_name: str, date: str) -> pd.Series:
    hits = merged[(merged["player_name"] == player_name) & (merged["date"] == date)]
    assert len(hits) == 1, f"expected exactly one row for {player_name} on {date}, found {len(hits)}"
    return hits.iloc[0]


def test_normalise_first_name_extracts_first_initial():
    """Full given-name matching was tried first and reverted: it broke 584 real
    rows across 39 players because afltables/footywire don't always agree on a
    nickname ("Cam"/"Cameron", "Lachie"/"Lachlan", etc.). The first initial is
    the strongest requirement that tolerates every observed nickname pair while
    still disambiguating the two known surname collisions (D vs L, T vs O)."""
    names = pd.Series(["Darcy B-Jones", "Lachlan Jones", "Nasiah W-Milera", "Oliver H-Brown", "Cam Mackenzie", "Cameron Mackenzie"])
    assert list(_normalise_first_name(names)) == ["d", "l", "n", "o", "c", "c"]


def test_nickname_variants_share_the_same_first_initial():
    pairs = [
        ("Cam", "Cameron"), ("Lachie", "Lachlan"), ("Matt", "Matthew"),
        ("Nick", "Nicholas"), ("Tom", "Thomas"), ("Zac", "Zach"),
        ("Ollie", "Oliver"), ("Sam", "Samuel"), ("Will", "William"), ("Mitch", "Mitchell"),
    ]
    for nickname, full in pairs:
        assert _normalise_first_name(pd.Series([nickname])).iloc[0] == _normalise_first_name(pd.Series([full])).iloc[0], (
            f"{nickname} vs {full}"
        )


def test_first_name_disambiguates_a_synthetic_plain_surname_collision():
    """Two different players whose surname_key collides ("jones") must NOT
    collide once first_name_key is included."""
    a = pd.Series(["Darcy Byrne-Jones"])
    b = pd.Series(["Lachie Jones"])
    assert _normalise_surname(a).iloc[0] == _normalise_surname(b).iloc[0] == "jones"
    assert _normalise_first_name(a).iloc[0] != _normalise_first_name(b).iloc[0]


class TestNoCollisionOnRealData:
    def test_byrne_jones_and_lachie_jones_have_independent_stats(self, merged_2026):
        merged, _ = merged_2026
        shared_dates = sorted(
            set(merged.loc[merged["player_name"] == "Darcy Byrne-Jones", "date"])
            & set(merged.loc[merged["player_name"].isin(["Lachie Jones", "Lachlan Jones"]), "date"])
        )
        assert shared_dates, "test fixture assumption broken: these two should share at least one match date"
        lachie_name = merged.loc[merged["player_name"].str.contains("Jones") & merged["player_name"].str.startswith(("Lachie", "Lachlan")), "player_name"].iloc[0]
        identical_rows = []
        for date in shared_dates:
            bj = _row(merged, "Darcy Byrne-Jones", date)
            lj = _row(merged, lachie_name, date)
            if pd.notna(bj["effective_disposals"]) and pd.notna(lj["effective_disposals"]):
                if bj["effective_disposals"] == lj["effective_disposals"] and bj["metres_gained"] == lj["metres_gained"]:
                    identical_rows.append(date)
        assert not identical_rows, (
            f"Byrne-Jones and {lachie_name} still show identical stats on: {identical_rows} "
            "-- the collision is not fixed"
        )

    def test_tom_brown_and_hayes_brown_have_independent_stats(self, merged_2026):
        merged, _ = merged_2026
        shared_dates = sorted(
            set(merged.loc[merged["player_name"] == "Tom Brown", "date"])
            & set(merged.loc[merged["player_name"] == "Oliver Hayes-Brown", "date"])
        )
        assert shared_dates, "test fixture assumption broken: these two should share at least one match date"
        identical_rows = []
        for date in shared_dates:
            tb = _row(merged, "Tom Brown", date)
            hb = _row(merged, "Oliver Hayes-Brown", date)
            if pd.notna(tb["effective_disposals"]) and pd.notna(hb["effective_disposals"]):
                if tb["effective_disposals"] == hb["effective_disposals"] and tb["metres_gained"] == hb["metres_gained"]:
                    identical_rows.append(date)
        assert not identical_rows, f"Tom Brown and Hayes-Brown still show identical stats on: {identical_rows}"


HYPHENATED_PLAYERS = [
    "Alex Neal-Bullen",
    "Jason Horne-Francis",
    "Jamarra Ugle-Hagan",
    "Darcy Byrne-Jones",
    "Brandon Zerk-Thatcher",
    "Nasiah Wanganeen-Milera",
    "Callum Coleman-Jones",
    "Saad El-Hawli",
    "Archer Day-Wicks",
    "Luke Davies-Uniacke",
    "Hugo Hall-Kahan",
    "Cooper Duff-Tytler",
]


def test_all_12_hyphenated_players_still_resolve(merged_2026):
    """The collision fix must not regress the original hyphenated-surname fix:
    every one of the 12 players must still have at least one matched
    (non-null) advanced row."""
    merged, _ = merged_2026
    unresolved = []
    for name in HYPHENATED_PLAYERS:
        rows = merged[merged["player_name"] == name]
        assert len(rows) > 0, f"{name} has no CORE rows at all -- test data problem"
        if not rows["effective_disposals"].notna().any():
            unresolved.append(name)
    assert not unresolved, f"players with zero resolved advanced rows after the fix: {unresolved}"


def test_no_duplicate_or_many_to_many_joins(merged_2026):
    merged, _ = merged_2026
    dup = merged.duplicated(subset=["match_id", "player_id"], keep=False)
    assert not dup.any(), f"{dup.sum()} duplicate (match_id, player_id) rows after the join"


def test_ambiguous_same_key_teammates_are_never_matched():
    """Chad Warner and Corey Warner (real 2026 Sydney teammates) share the same
    (date, team, opponent, surname_key, first_name_key='c') identity key on
    several match dates. A single footywire row for either could not be safely
    attributed to one or the other, so both must be left unmatched on every
    date where the ambiguity applies -- never guessed, even if guessing would
    happen to look correct on the data available today."""
    core_2026, _ = build_core_2026()
    merged, _ = build_advanced_2026(core_2026)
    warner_rows = merged[merged["player_name"].isin(["Chad Warner", "Corey Warner"])]
    shared_dates = (
        warner_rows.groupby(["match_id"])["player_id"].nunique().loc[lambda s: s > 1].index
    )
    assert len(shared_dates) > 0, "test fixture assumption broken: expected at least one shared Warner match"
    ambiguous_rows = warner_rows[warner_rows["match_id"].isin(shared_dates)]
    assert ambiguous_rows["effective_disposals"].isna().all(), (
        f"ambiguous Warner rows must all be null, found: "
        f"{ambiguous_rows[['player_name', 'date', 'effective_disposals']].to_dict('records')}"
    )


def test_join_rate_did_not_regress(merged_2026):
    """The collision fix should not meaningfully reduce overall coverage --
    only the ambiguous rows (if any) should be nulled, and none were expected
    on real 2026 data given first names fully disambiguate every case found."""
    _, validation = merged_2026
    assert validation["advanced_join_match_rate_2026"] > 0.95, validation
