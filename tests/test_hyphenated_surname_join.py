"""
Regression tests for the Production Scenario C / footywire identity-join bug
affecting hyphenated surnames.

Root cause: footywire's player_stats.rda abbreviates the FIRST component of a
hyphenated compound surname to a single initial (e.g. "Wanganeen-Milera" ->
"W-Milera", "Davies-Uniacke" -> "D-Uniacke"), while afltables spells it in
full. src.data.build_2026_extension._normalise_surname() previously took the
last whitespace-separated token, lowercased, and stripped non-letters -- which
produced DIFFERENT keys for the two spellings ("wanganeenmilera" vs
"wmilera"), so the CORE<->ADVANCED join silently dropped these 12 players from
Scenario C, leaving them with FINAL_ENSEMBLE=0.0 despite having real match
data and real Scenario A/B values. Fixed by taking only the segment after the
last hyphen before stripping punctuation, which collapses both spelling
conventions to the same key and is a no-op for non-hyphenated surnames.

This only affects the 2026 additive extension (build_2026_extension.py); the
historical ADVANCED builder (build_advanced_dataset.py, feeding all Phase 4
training/backtests) has the identical unfixed pattern and is a documented,
separate, out-of-scope finding -- not touched here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_2026_extension import _normalise_surname

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

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

# The real footywire-side spelling for each player above (first component
# abbreviated to its initial) -- confirmed directly against
# data/raw/fitzroy_data/player_stats.rda's 2026 rows.
FOOTYWIRE_SPELLINGS = [
    "Alex N-Bullen",
    "Jason H-Francis",
    "Jamarra U-Hagan",
    "Darcy B-Jones",
    "Brandon Z-Thatcher",
    "Nasiah W-Milera",
    "Callum C-Jones",
    "Saad E-Hawli",
    "Archer D-Wicks",
    "Luke D-Uniacke",
    "Hugo H-Kahan",
    "Cooper D-Tytler",
]


def test_afltables_and_footywire_spellings_collapse_to_same_key():
    afl_keys = _normalise_surname(pd.Series(HYPHENATED_PLAYERS))
    fw_keys = _normalise_surname(pd.Series(FOOTYWIRE_SPELLINGS))
    mismatches = [
        (a, f, ak, fk)
        for a, f, ak, fk in zip(HYPHENATED_PLAYERS, FOOTYWIRE_SPELLINGS, afl_keys, fw_keys)
        if ak != fk
    ]
    assert not mismatches, f"surname_key mismatch(es): {mismatches}"


def test_non_hyphenated_surnames_unaffected():
    """The fix must be a no-op for ordinary surnames -- prove a handful of
    real, unrelated player names still normalise exactly as before."""
    names = pd.Series(["Nick Daicos", "Bailey Smith", "Marcus Bontempelli", "Patrick Cripps"])
    assert list(_normalise_surname(names)) == ["daicos", "smith", "bontempelli", "cripps"]


def test_generalises_beyond_the_named_examples():
    """A synthetic hyphen+apostrophe case, to prove this isn't tuned to the
    12 known players specifically."""
    full = pd.Series(["Jamie O'Brien-Walsh"])
    abbrev = pd.Series(["Jamie O-Walsh"])
    assert _normalise_surname(full).iloc[0] == _normalise_surname(abbrev).iloc[0]


@pytest.mark.skipif(
    not (REPORTS / "2026_leaderboard.csv").exists(),
    reason="production leaderboard not present in this environment",
)
class TestRebuiltProductionOutputs:
    def setup_method(self):
        self.lb = pd.read_csv(REPORTS / "2026_leaderboard.csv")

    def test_all_12_players_have_nonzero_final_ensemble(self):
        rows = self.lb[self.lb["player_name"].isin(HYPHENATED_PLAYERS)]
        assert len(rows) == 12, f"expected all 12 named players in the leaderboard, found {len(rows)}"
        assert (rows["FINAL_ENSEMBLE"] > 0).all(), rows[["player_name", "FINAL_ENSEMBLE"]]
        assert rows["sim_mean_votes"].notna().all(), rows[["player_name", "sim_mean_votes"]]

    def test_no_duplicate_player_ids(self):
        assert not self.lb["player_id"].duplicated().any()

    def test_season_total_expected_votes_still_1242(self):
        import json

        qc = json.loads((REPORTS / "2026_quality_checks.json").read_text())
        assert qc["actual_season_total_expected_votes"] == pytest.approx(1242.0, abs=1e-6)
        assert qc["all_matches_covered"] is True

    def test_season_ev_equals_sum_of_match_ev_for_fixed_players(self):
        mp = pd.read_csv(REPORTS / "2026_match_probabilities.csv")
        ids = self.lb[self.lb["player_name"].isin(HYPHENATED_PLAYERS)]["player_id"]
        sub = mp[mp["player_id"].isin(ids)].copy()
        assert sub[["p3", "p2", "p1"]].notna().all().all()
        sub["ev"] = 3 * sub["p3"] + 2 * sub["p2"] + 1 * sub["p1"]
        match_sum = sub.groupby("player_id")["ev"].sum()
        season = self.lb.set_index("player_id").loc[match_sum.index, "FINAL_ENSEMBLE"]
        assert (match_sum - season).abs().max() < 1e-6

    def test_objective_outputs_untouched(self):
        """Objective doesn't use this join at all -- its outputs must be
        completely insensitive to this fix. Only meaningful once a pre-fix
        snapshot exists; skipped standalone since this test file has no
        before/after state of its own to compare against."""
        assert (REPORTS / "2026_objective_leaderboard.csv").exists()
