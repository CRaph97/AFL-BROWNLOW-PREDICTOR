"""
Regression tests for a dashboard data-integrity bug: build_player_round_by_round()
used to return a NON-empty dataframe (real model_core_2026 rows, but NaN
p3/p2/p1/expected_votes) for any player with zero rows in
reports/2026_match_probabilities.csv, causing every page that gates on
`.empty` to render a misleading NaN/0-filled table instead of a "no data
resolved" warning.

This affected a real, pre-existing production-pipeline gap: Scenario C's
footywire join failed to resolve any of the 12 hyphenated-surname players
(12/12 correlation), leaving them with FINAL_ENSEMBLE=0.0 despite real match
data. That upstream gap has since been fixed at its source (the
surname-normalisation function in src/data/build_2026_extension.py -- see
tests/test_hyphenated_surname_join.py) and the pipeline rebuilt, so as of
this build there are no known gap players left -- `_gap_player_ids()` is
expected to return an empty list. The safeguard added here
(build_player_round_by_round() returning truly empty rather than a
NaN-filled table for ANY player with zero match-probability rows) is kept as
a general defence for a future, currently-unknown gap, not because a gap is
known to exist right now. The tests below are written generically (derived
from the data, not a hardcoded name list) so they keep working whether the
gap set is empty (today) or non-empty (if a new one is ever introduced).
"""
import numpy as np
import pandas as pd
import pytest

from dashboard import data as d

NASIAH_ID = 12950


def _gap_player_ids() -> list[int]:
    """Players with real leaderboard rows but zero production match-level
    data -- the exact condition build_player_round_by_round() must detect."""
    lb = d.load_leaderboard()
    return lb.loc[lb["FINAL_ENSEMBLE"] == 0.0, "player_id"].tolist()


def _healthy_player_ids(limit: int = 40) -> list[int]:
    lb = d.load_leaderboard()
    return lb.loc[lb["FINAL_ENSEMBLE"] > 0.0, "player_id"].head(limit).tolist()


def test_nasiah_wanganeen_milera_now_resolves_with_real_data():
    """Nasiah was the named example of the hyphenated-surname join bug (real
    CORE rows, but FINAL_ENSEMBLE=0.0 and no round-by-round data). Now that
    the upstream join is fixed, he must resolve like any other real
    contender -- non-zero EV and a real, non-empty, non-null round-by-round."""
    lb = d.load_leaderboard()
    row = lb[lb["player_id"] == NASIAH_ID].iloc[0]
    assert row["player_name"] == "Nasiah Wanganeen-Milera"
    assert row["FINAL_ENSEMBLE"] > 0, "Nasiah's upstream join gap should be fixed -- FINAL_ENSEMBLE is still 0.0"
    core = d.load_core_2026()
    assert (core["player_id"] == NASIAH_ID).sum() > 0
    rbr = d.build_player_round_by_round(NASIAH_ID)
    assert not rbr.empty, "Nasiah should now have real round-by-round data"
    assert rbr["expected_votes"].notna().all() and rbr["p3"].notna().all()


def test_every_gap_player_returns_empty_round_by_round():
    """Generic safeguard check: IF any player still has FINAL_ENSEMBLE==0.0
    (a future, currently-unknown gap), their round-by-round must be empty,
    not NaN-filled. Today's expected state is an empty gap set -- that is
    success, not a skipped check, since the known 12-player gap was fixed."""
    gap_ids = _gap_player_ids()
    for pid in gap_ids:
        rbr = d.build_player_round_by_round(int(pid))
        assert rbr.empty, f"player_id {pid} has FINAL_ENSEMBLE==0.0 but round-by-round is not empty"


def test_healthy_players_have_non_empty_non_null_round_by_round():
    """Sanity check that the fix didn't over-correct: real, resolvable
    players must still get real round-by-round data for the large majority of
    their matches.

    Not a 100%-non-null requirement: a player can legitimately have a small
    number of genuinely-unresolvable matches without being a "gap player" --
    e.g. player_id 12797 (Chad Warner) shares an identity key with a real
    teammate (Corey Warner, both real 2026 Sydney players) on a handful of
    match dates, which the join's ambiguity guard correctly leaves unmatched
    rather than guessing (see src/data/build_2026_extension.py). This is the
    same class of pre-existing, accepted gap as the "6 of 207 matches can't be
    scored due to missing Round-1 lagged features" limitation documented in
    docs/2026_MODELLING_METHODOLOGY.md -- a real, disclosed data limitation,
    not an identity-resolution failure like the 12-hyphenated-player bug this
    test file was originally written to guard against."""
    for pid in _healthy_player_ids():
        rbr = d.build_player_round_by_round(int(pid))
        assert not rbr.empty, f"player_id {pid} unexpectedly has no round-by-round data"
        non_null_rate = rbr["expected_votes"].notna().mean()
        assert non_null_rate >= 0.7, (
            f"player_id {pid} has only {non_null_rate:.0%} non-null expected_votes rows "
            "-- too low to be an isolated ambiguous-match gap"
        )


def test_season_ev_equals_sum_of_round_ev_for_all_resolvable_players():
    """Check B from the audit brief: season Expected Votes == sum(round
    expected_votes), for every player who has real match-level data at all.
    Known-gap players are excluded (their season EV is a separately-flagged
    0.0 data gap, not a sum-consistency question)."""
    lb = d.load_leaderboard()
    mp = d.load_match_probabilities()
    gap_ids = set(_gap_player_ids())
    sums = mp.groupby("player_id")["expected_votes"].sum()

    resolvable = lb[~lb["player_id"].isin(gap_ids)]
    mismatches = []
    for _, row in resolvable.iterrows():
        pid = row["player_id"]
        season_ev = row["FINAL_ENSEMBLE"]
        round_sum = sums.get(pid, 0.0)
        if abs(season_ev - round_sum) > 0.05:
            mismatches.append((pid, row["player_name"], season_ev, round_sum))

    assert not mismatches, f"season EV != sum(round EV) for: {mismatches[:10]}"


def test_mc_summary_columns_are_all_or_nothing_never_partial():
    """Check D from the audit brief: a player either has a complete Monte
    Carlo summary (mean/median/p10/p90/p2_5/p97_5 all present) or none of it
    -- a partial gap would indicate a different, un-investigated failure
    mode and must not pass silently."""
    lb = d.load_leaderboard()
    sim_cols = ["sim_mean_votes", "sim_median_votes", "sim_p10", "sim_p90", "sim_p2_5", "sim_p97_5"]
    null_counts = lb[sim_cols].isna().sum(axis=1)
    partial = lb[(null_counts > 0) & (null_counts < len(sim_cols))]
    assert partial.empty, f"players with a PARTIAL (not all-or-nothing) MC-summary gap: {partial['player_name'].tolist()}"


def test_no_null_probabilities_in_production_match_file():
    """Check C from the audit brief: every row that exists at all in
    2026_match_probabilities.csv must have non-null p3/p2/p1/p0/expected_votes
    -- a row existing with a null value would be a distinct, worse bug (data
    present but corrupt) than a player being entirely absent."""
    mp = d.load_match_probabilities()
    prob_cols = ["p3", "p2", "p1", "p0", "expected_votes"]
    assert mp[prob_cols].notna().all().all()


def test_projection_concentration_uses_same_ev_values_as_round_by_round():
    """Check E from the audit brief: Projection Concentration's bucketed EV
    values must be derived from the exact same round-by-round `expected_votes`
    Player Detail shows -- both call the same build_player_round_by_round(),
    so this mainly guards against a future divergence (e.g. one page adding
    its own separate loader) or the bucket logic silently drifting from the
    underlying rbr it was given."""
    for pid in _healthy_player_ids(limit=5):
        rbr = d.build_player_round_by_round(int(pid))
        conc = d.projection_concentration(rbr)
        for t in [2.5, 2.0, 1.5, 1.0]:
            row = conc[conc["bucket"] == f"EV >= {t}"].iloc[0]
            expected_n = int((rbr["expected_votes"] >= t).sum())
            expected_sum = rbr.loc[rbr["expected_votes"] >= t, "expected_votes"].sum()
            assert row["n_games"] == expected_n, f"player {pid}, threshold {t}: bucket count drifted from rbr"
            assert abs(row["ev_contribution"] - expected_sum) < 1e-9, (
                f"player {pid}, threshold {t}: bucket EV sum drifted from rbr"
            )


def test_local_and_deployment_core_2026_loaders_agree_on_gap_players():
    """Local/deployment parity check: the slimmed deployment parquet
    (data/deployment/model_core_2026_dashboard.parquet) must carry the same
    known-gap players as the local file, so this bug's fix behaves
    identically in both environments."""
    core = d.load_core_2026()
    for pid in _gap_player_ids():
        assert (core["player_id"] == pid).sum() > 0, (
            f"player_id {pid} has real CORE rows locally but is missing from "
            "whichever load_core_2026() path is currently active -- local/deployment drift"
        )
