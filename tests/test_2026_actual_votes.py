"""
Targeted tests for the 2026 Brownlow ACTUAL-vote ground truth
(data/actual/*, built by src/actual/build_actual_votes.py from the raw AFL
tracker snapshots in data/actual/raw/). Offline: everything here reads the
committed CSV/JSON outputs; nothing hits the network.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.actual.build_actual_votes import (
    FROZEN_PREDICTION_FILES,
    HOME_AWAY_ROUNDS,
    KNOWN_UNRESOLVED_AFL_PLAYERS,
    ROOT,
    frozen_file_hashes,
    load_canonical_matches,
    lookup_name,
)

ACTUAL = ROOT / "data" / "actual"
RAW = ACTUAL / "raw"


@pytest.fixture(scope="module")
def match_votes():
    return pd.read_csv(ACTUAL / "2026_brownlow_match_votes.csv", dtype={"player_id": "string"})


@pytest.fixture(scope="module")
def leaderboard():
    return pd.read_csv(ACTUAL / "2026_brownlow_leaderboard.csv", dtype={"player_id": "string"})


@pytest.fixture(scope="module")
def player_round():
    return pd.read_csv(ACTUAL / "2026_brownlow_player_round.csv", dtype={"player_id": "string"})


@pytest.fixture(scope="module")
def validation():
    return json.loads((ACTUAL / "2026_brownlow_validation.json").read_text())


@pytest.fixture(scope="module")
def fetch_status():
    return json.loads((RAW / "fetch_status.json").read_text())


# ---------------------------------------------------------------- per-match structure
def test_every_match_has_exactly_six_votes(match_votes):
    totals = match_votes.groupby("match_id")["actual_brownlow_votes"].sum()
    assert (totals == 6).all(), totals[totals != 6]
    assert len(totals) == 207
    assert len(match_votes) == 621


def test_every_match_has_one_3_one_2_one_1(match_votes):
    sets = match_votes.groupby("match_id")["actual_brownlow_votes"].apply(lambda s: tuple(sorted(s, reverse=True)))
    assert (sets == (3, 2, 1)).all(), sets[sets != (3, 2, 1)]
    assert not match_votes.duplicated(["match_id", "actual_brownlow_votes"]).any()
    assert not match_votes.duplicated(["match_id", "afl_player_id"]).any()
    assert set(match_votes["actual_brownlow_votes"].unique()) == {1, 2, 3}


def test_vote_getter_played_for_home_or_away_team(match_votes):
    ok = (match_votes["player_team"] == match_votes["home_team"]) | (match_votes["player_team"] == match_votes["away_team"])
    assert ok.all(), match_votes[~ok]


# ---------------------------------------------------------------- totals reconciliation
def test_reconstructed_totals_equal_afl_leaderboard(match_votes, leaderboard):
    recon = match_votes.groupby("afl_player_id")["actual_brownlow_votes"].sum()
    lb = leaderboard.set_index("afl_player_id")
    assert set(recon.index) == set(lb.index)
    diff = lb["afl_total_actual_votes"] - recon.reindex(lb.index)
    assert (diff == 0).all(), lb.loc[diff != 0, ["player_name", "afl_total_actual_votes"]]
    assert (lb["afl_total_actual_votes"] == lb["reconstructed_total_votes"]).all()
    assert int(lb["afl_total_actual_votes"].sum()) == 207 * 6


def test_nick_daicos_total_is_47_and_winner(leaderboard, match_votes):
    d = leaderboard[leaderboard["player_name"] == "Nick Daicos"]
    assert len(d) == 1
    assert int(d["afl_total_actual_votes"].iloc[0]) == 47
    assert bool(d["winner"].iloc[0])
    assert int(d["rank"].iloc[0]) == 1
    assert int(match_votes.loc[match_votes["player_name"] == "Nick Daicos", "actual_brownlow_votes"].sum()) == 47


def test_player_round_actuals_reconcile_to_match_votes(player_round, match_votes):
    pr_tot = player_round.groupby("afl_player_id")["actual_votes"].sum(min_count=1).fillna(0)
    mv_tot = match_votes.groupby("afl_player_id")["actual_brownlow_votes"].sum()
    assert (pr_tot.reindex(mv_tot.index) == mv_tot).all()
    # every voted (player, round) in player_round points at the same AFL match as match_votes
    voted = player_round[player_round["actual_votes"].fillna(0) > 0][["afl_player_id", "round", "afl_match_id", "actual_votes"]]
    merged = voted.merge(match_votes[["afl_player_id", "round", "afl_match_id", "actual_brownlow_votes"]],
                         on=["afl_player_id", "round", "afl_match_id"], how="left")
    assert merged["actual_brownlow_votes"].notna().all()
    assert (merged["actual_brownlow_votes"] == merged["actual_votes"]).all()


# ---------------------------------------------------------------- pagination completeness
def test_leaderboard_pagination_was_exhausted(fetch_status, leaderboard):
    assert fetch_status["load_more_button_present_at_end"] is False
    clicks = fetch_status["pagination"]
    assert len(clicks) > 1, "pagination never clicked"
    rows = [c["rows_visible"] for c in clicks]
    assert rows == sorted(rows) and rows[-1] > rows[0]
    assert all("Show next" in c["label"] for c in clicks[1:])
    api_n = len(json.loads((RAW / "afl_bfawards_leaderboard_CD_S2026014.json").read_text())["body"]["leaderboard"])
    assert fetch_status["dom_rows_after_pagination"] == api_n == len(leaderboard) == 183
    # predictor pages were captured for every leaderboard player (one XHR per page of 15)
    pages = json.loads((RAW / "aflapi_award_brownlow_predictor_pages.json").read_text())
    pred_ids = {p["providerId"] for pg in pages for p in pg["body"]["players"]}
    assert pred_ids == set(leaderboard["afl_player_id"])


def test_dom_snapshot_agrees_with_api(validation):
    assert validation["checks"]["dom_leaderboard_matches_api"]["ok"] is True


# ---------------------------------------------------------------- identity
def test_identity_uniqueness(match_votes, leaderboard):
    assert leaderboard["afl_player_id"].is_unique
    assert not leaderboard["player_id"].dropna().duplicated().any()
    res = match_votes[match_votes["player_id"].notna()]
    assert (res.groupby("afl_player_id")["player_id"].nunique() == 1).all()
    assert (res.groupby("player_id")["afl_player_id"].nunique() == 1).all()
    # a canonical id resolved for a player never changes team across the season
    assert (res.groupby("player_id")["player_team"].nunique() == 1).all()


def test_unresolved_identities_are_flagged_never_guessed(match_votes):
    unresolved = match_votes[~match_votes["identity_status"].str.startswith("resolved")]
    assert unresolved["player_id"].isna().all()
    assert set(unresolved["afl_player_id"]) <= set(KNOWN_UNRESOLVED_AFL_PLAYERS), unresolved
    resolved = match_votes[match_votes["identity_status"].str.startswith("resolved")]
    assert resolved["player_id"].notna().all()
    assert not resolved["player_id"].str.startswith("NOID").any()


def test_known_identity_edge_cases(match_votes):
    chad = match_votes[match_votes["player_name"] == "Chad Warner"]
    assert (chad["player_id"] == "12797").all()
    assert set(chad["identity_status"]) <= {"resolved", "resolved_exact_full_name"}
    # Round 4: Corey Warner also played for Sydney, so surname+initial is ambiguous and
    # only the exact-full-name path may resolve it (never the first-initial key).
    assert chad.loc[chad["round"] == 4, "identity_status"].tolist() == ["resolved_exact_full_name"]
    bjw = match_votes[match_votes["player_name"] == "Bailey J. Williams"]
    assert len(bjw) >= 1 and (bjw["player_id"] == "12836").all() and (bjw["player_team"] == "west_coast").all()
    ross = match_votes[match_votes["player_name"] == "Jack Ross"]
    assert len(ross) >= 1 and ross["player_id"].isna().all() and (ross["identity_status"] == "unresolved").all()


def test_lookup_name_strips_middle_initial_only():
    assert lookup_name("Bailey J. Williams") == "Bailey Williams"
    assert lookup_name("Bailey J Williams") == "Bailey Williams"
    assert lookup_name("Nasiah Wanganeen-Milera") == "Nasiah Wanganeen-Milera"
    assert lookup_name("Jacob van Rooyen") == "Jacob van Rooyen"
    assert lookup_name("Nick Daicos") == "Nick Daicos"


# ---------------------------------------------------------------- round coverage / normalisation
def test_round_coverage_matches_canonical_season(match_votes):
    canon = load_canonical_matches()
    got = match_votes.groupby("round")["match_id"].nunique().to_dict()
    want = canon.groupby("round")["match_id"].nunique().to_dict()
    assert sorted(got) == HOME_AWAY_ROUNDS == list(range(25))
    assert got == want
    assert set(match_votes["match_id"]) == set(canon["match_id"])
    assert (match_votes.loc[match_votes["round"] == 0, "round_label"] == "Opening Round").all()
    assert int((match_votes["round"] == 0).sum()) == 15  # 5 Opening Round matches x 3 votes
    assert match_votes["afl_match_id"].nunique() == 207
    # the date embedded in the canonical match_id equals the AFL local-venue date
    assert (match_votes["match_id"].str[-10:] == match_votes["date"]).all()


def test_player_round_grid_is_complete(player_round, leaderboard):
    assert len(player_round) == len(leaderboard) * 25
    assert not player_round.duplicated(["afl_player_id", "round"]).any()
    assert set(player_round["status"].unique()) <= {"played", "bye", "did_not_play"}
    byes = player_round[player_round["status"] == "bye"]
    assert byes["actual_votes"].isna().all() and byes["afl_predicted_votes"].isna().all()
    # every player has at least one bye in a 25-round / 24-game season
    assert (player_round.groupby("afl_player_id")["status"].apply(lambda s: (s == "bye").sum()) >= 1).all()


# ---------------------------------------------------------------- training integrity
def test_frozen_prediction_files_not_contaminated(validation):
    # 1. every frozen file still hashes exactly as it did when the actual votes were built
    now = frozen_file_hashes()
    recorded = validation["frozen_file_hashes"]
    assert now == recorded, {k: (recorded.get(k), now[k]) for k in now if recorded.get(k) != now[k]}
    # 2. no frozen CSV grew an "actual" column
    for p in FROZEN_PREDICTION_FILES:
        if p.suffix == ".csv":
            cols = pd.read_csv(p, nrows=0).columns
            assert not any("actual" in c.lower() for c in cols), (p, list(cols))
    # 3. nothing under the feature/model/simulation code reads the actual-vote outputs
    hits = subprocess.run(
        ["grep", "-rIl", "-e", "data/actual", "-e", "brownlow_match_votes", "-e", "src.actual",
         str(ROOT / "src" / "features"), str(ROOT / "src" / "models"), str(ROOT / "src" / "data"),
         str(ROOT / "src" / "simulation")],
        capture_output=True, text=True,
    ).stdout.split()
    assert hits == [], hits


def test_validation_report_all_hard_checks_pass(validation):
    assert validation["all_hard_checks_pass"] is True
    assert validation["counts"] == {"players": 183, "matches": 207, "vote_rows": 621,
                                    "player_round_rows": 4575, "eligible_players": 169}
    inel = validation["checks"]["ineligible_players_report"]
    assert inel["count"] == 14
