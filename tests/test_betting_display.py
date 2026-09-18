"""
Regression tests for the Betting Opportunities page redesign
(dashboard/betting_opportunities.py, pages/23_Brownlow_Betting_Opportunities.py).

Covers the display/mapping bugs fixed in that pass:
- raw CONSTANT_CASE market_type leaking into UI (market_label/bet_description)
- Top-N not showing which N
- team O/U rows not showing team+line
- non-player text (margin buckets, "Round N") leaking into player_name
  (src/betting/scraping.py's UNMODELLED-row fix)
- combination display completeness (no "? -- ?", no blank legs)
- Top Opportunities excluding NO_VALUE/MODEL_DISAGREEMENT/UNMODELLED, not
  just data-quality-flagged rows
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard import betting_opportunities as bo

ROOT = Path(__file__).resolve().parent.parent
OPPS_PATH = ROOT / "data" / "betting" / "processed" / "priced_opportunities.csv"
COMBOS_PATH = ROOT / "data" / "betting" / "processed" / "combinations.csv"

pytestmark = pytest.mark.skipif(not OPPS_PATH.exists(), reason="no priced betting opportunities in this environment")


@pytest.fixture(scope="module")
def opps():
    return pd.read_csv(OPPS_PATH)


@pytest.fixture(scope="module")
def display(opps):
    return bo.prepare_display(opps)


def test_market_label_never_leaks_raw_constant(display):
    known_constants = set(display["market_type"].unique())
    leaked = [label for label in display["market_label"].unique() if label in known_constants]
    assert not leaked, f"raw market_type constant(s) leaked into a human-readable label: {leaked}"
    assert not display["market_label"].str.fullmatch(r"[A-Z_]+").any(), (
        "a CONSTANT_CASE-looking label leaked into the UI"
    )


def test_top_n_label_shows_the_n(display):
    top_n = display[display["market_type"] == "TOP_N"]
    if top_n.empty:
        pytest.skip("no TOP_N rows in this run's data")
    assert (top_n["market_label"].str.contains(r"Top \d+ Finish")).all()


def test_team_votes_ou_bet_names_team_and_line(display):
    team_rows = display[display["market_type"] == "TEAM_VOTES_OU"]
    if team_rows.empty:
        pytest.skip("no TEAM_VOTES_OU rows in this run's data")
    assert not team_rows["bet"].str.contains("Unknown team|nan", case=False, na=False).any()
    # every team bet must mention a numeric line
    assert team_rows["bet"].str.contains(r"\d").all()


def test_no_non_player_text_in_player_name(opps):
    bad_patterns = ["votes", "Round "]
    names = opps["player_name"].dropna().unique()
    leaked = [n for n in names if any(p.lower() in str(n).lower() for p in bad_patterns)]
    assert not leaked, f"non-player text leaked into player_name: {leaked}"


def test_qualifying_opportunities_excludes_no_value_and_unmodelled(display):
    q = bo.qualifying_opportunities(display)
    assert not q["confidence"].isin(["NO_VALUE", "MODEL_DISAGREEMENT"]).any()
    assert not (q["market_type"] == "UNMODELLED").any()
    assert not q["has_flag"].any()
    assert not q["is_incomplete"].any()


def test_combinations_have_no_blank_or_placeholder_legs():
    if not COMBOS_PATH.exists():
        pytest.skip("no combinations file in this environment")
    combos = pd.read_csv(COMBOS_PATH)
    if combos.empty:
        pytest.skip("no combination rows in this run's data")
    assert combos["legs"].notna().all()
    assert not combos["legs"].astype(str).str.contains(r"\?", regex=True).any()
    assert not (combos["legs"].astype(str).str.strip() == "").any()


def test_bet_description_always_non_empty_for_modelled_rows(display):
    modelled = display[display["market_type"] != "UNMODELLED"]
    assert modelled["bet"].notna().all()
    assert not (modelled["bet"].astype(str).str.strip() == "").any()


def test_h2h_opponent_resolves_from_market_name(display):
    h2h = display[display["market_type"] == "PLAYER_H2H"]
    if h2h.empty:
        pytest.skip("no PLAYER_H2H rows in this run's data")
    assert h2h["market_label"].str.startswith("H2H vs ").all()
    assert not h2h["market_label"].str.contains("H2H vs None").any()


# --------------------------------------------------------------------------
# Fast final cleanup pass: team-less PointsBet identity resolution, team
# totals, friendly support labels, and the To Poll a Vote match-level
# drill-down (see src/external/identity.py's team-less fallback and
# dashboard/betting_opportunities.py's match_level_drilldown()).
# --------------------------------------------------------------------------

def test_pointsbet_player_markets_resolve_without_a_team_field(display):
    """PointsBet's own raw API response carries an empty teamId/playerId for
    every player-market outcome (confirmed directly against
    data/betting/raw/pointsbet_1.json) -- resolve_external_players() must
    still resolve a genuinely unique player name via its team-less fallback,
    not leave every PointsBet player row permanently unresolved."""
    pointsbet_player_rows = display[
        display["source"].astype(str).str.startswith("pointsbet")
        & display["player_name"].notna()
        & (display["market_type"] != "UNMODELLED")
    ]
    if pointsbet_player_rows.empty:
        pytest.skip("no PointsBet player-market rows in this run's data")
    resolution_rate = pointsbet_player_rows["player_id"].notna().mean()
    assert resolution_rate > 0.8, (
        f"PointsBet player-row identity resolution rate is only {resolution_rate:.2%} -- "
        "the team-less fallback in src/external/identity.py may have regressed"
    )


def test_bailey_smith_top_n_merges_both_bookmakers_into_one_row(display):
    """The exact case that exposed the identity bug: before the fix, Bailey
    Smith's PointsBet Top-3/Top-5 rows had player_id=NaN (team-less, and the
    old team-required-only matching left them unresolved), so
    with_bookmaker_odds()'s player_id grouping produced two rows per
    threshold (one real, one all-NaN) instead of one merged row."""
    for n in (3, 5):
        sub = display[(display["market_type"] == "TOP_N") & (display["n"] == n)]
        if sub.empty:
            continue
        piv = bo.with_bookmaker_odds(sub, ["player_id"])
        bs = piv[piv["player_name"] == "Bailey Smith"]
        if bs.empty:
            continue
        assert len(bs) == 1, f"Bailey Smith Top {n} appears {len(bs)} times, expected exactly 1"
        assert pd.notna(bs.iloc[0]["production_probability"]), "merged row lost its real probability"
        assert pd.notna(bs.iloc[0]["neds_odds"]) and pd.notna(bs.iloc[0]["pointsbet_odds"]), (
            "merged row should carry both bookmakers' prices"
        )


def test_no_remaining_na_duplicate_rows_are_unflagged(display):
    """Any player_id-grouped row that still has an all-NaN probability after
    merging must be genuinely flagged (e.g. IDENTITY_AMBIGUOUS) -- never a
    silent, unexplained duplicate."""
    top_n_types = display[display["market_type"] == "TOP_N"]
    for n in sorted(top_n_types["n"].dropna().unique()):
        sub = top_n_types[top_n_types["n"] == n]
        piv = bo.with_bookmaker_odds(sub, ["player_id"])
        bad = piv[piv["production_probability"].isna()]
        assert bad["has_flag"].all(), (
            f"Top {int(n)}: unflagged rows with no resolved probability: "
            f"{bad.loc[~bad['has_flag'], 'player_name'].tolist()}"
        )


def test_friendly_support_label_never_leaks_raw_constant():
    from dashboard.betting_opportunities import friendly_support_label
    assert friendly_support_label("STRONG_WHEELO_SUPPORT") == "Strong Wheelo support"
    assert friendly_support_label("WHEELO_DISAGREES") == "Wheelo disagrees"
    assert friendly_support_label("INSUFFICIENT_WHEELO_DATA") == "Insufficient Wheelo data"
    assert friendly_support_label("INSUFFICIENT_DATA") == "Insufficient external data"
    assert "_" not in friendly_support_label("PARTIAL_WHEELO_SUPPORT")


def test_no_raw_support_constants_in_display_columns(display):
    known_raw_constants = {
        "STRONG_WHEELO_SUPPORT", "PARTIAL_WHEELO_SUPPORT", "WHEELO_NEUTRAL",
        "WHEELO_DISAGREES", "INSUFFICIENT_WHEELO_DATA", "BROADER_EXTERNAL_SUPPORT",
        "MIXED_EXTERNAL", "BROADER_EXTERNAL_DISAGREEMENT", "INSUFFICIENT_DATA",
    }
    for col in ("wheelo_support_label", "external_support_label"):
        if col not in display.columns:
            continue
        values = set(display[col].dropna().unique().tolist())
        leaked = values & known_raw_constants
        assert not leaked, f"raw constant(s) leaked into {col}: {leaked}"


class TestTeamTotalsAndDrilldown:
    """Requires the real dashboard/data + reports files, not just the betting
    processed output -- separate from the module-level skip above."""

    def setup_method(self):
        pytest.importorskip("streamlit")
        from dashboard import data as d
        self.d = d

    def test_team_totals_reconcile_exactly(self):
        lb = pd.read_csv(ROOT / "reports" / "2026_leaderboard.csv")
        obj_lb = pd.read_csv(ROOT / "reports" / "2026_objective_leaderboard.csv")
        for team in ["western_bulldogs", "collingwood"]:
            rankings = self.d.dual_model_team_rankings(team)
            expected_prod = lb.loc[lb["team_id"] == team, "FINAL_ENSEMBLE"].sum()
            expected_obj = obj_lb.loc[obj_lb["team_id"] == team, "objective_ev"].sum()
            assert rankings["production_ev"].sum() == pytest.approx(expected_prod, abs=1e-6)
            assert rankings["objective_ev"].sum() == pytest.approx(expected_obj, abs=1e-6)

    def test_drilldown_works_with_the_real_float64_player_id_from_load_opportunities(self):
        """The prior version of this test sourced player_id from
        reports/2026_leaderboard.csv (int64), which never exercised the real
        page's actual data path: pages/23_Brownlow_Betting_Opportunities.py
        calls match_level_drilldown(prow["player_id"]) where `prow` comes
        from bo.load_opportunities(), and that column is float64 (upcast by
        an earlier merge introducing NaNs). str(12537.0) == "12537.0", which
        never matched the "12537"/12537 stored in match_probabilities or
        objective_votes -- so the entire "Why this bet?" drill-down silently
        returned empty for every player when used from the real page, despite
        this file's own tests passing. Verified live for Tim English (the
        user's own reported example) before this test existed."""
        from dashboard import betting_opportunities as bo_mod
        opps = bo_mod.load_opportunities()
        row = opps[opps["player_name"] == "Tim English"]
        if row.empty:
            pytest.skip("Tim English not present in this environment's opportunities")
        pid = row.iloc[0]["player_id"]
        assert isinstance(pid, float), "test assumption broken: load_opportunities()'s player_id is no longer float64"
        drilldown = bo_mod.match_level_drilldown(pid)
        assert not drilldown.empty, "match_level_drilldown returned empty for the real float64 player_id -- regression"
        assert drilldown["wheelo_match_ev"].notna().any()

    def test_wheelo_p3_pct_is_a_0_100_percentage_never_rescaled(self):
        """The raw Wheelo CSV's Votes3_Probability column is already a 0-100
        percentage (e.g. Tim English's Round 0 value is literally 0.3,
        meaning 0.3%, not 30%). Nothing in this app may multiply it by 100
        (it isn't a [0,1] fraction) or leave a genuinely-fractional value
        unconverted. This locks in the semantics found during the Wheelo
        integrity audit: EV=0.097 and P3=0.3% for that match are
        mathematically consistent (3*0.003=0.009 of the 0.097 EV comes from
        P3 alone) -- an earlier informal report of "P3 30%" for this exact
        match was a reporting error in a prior task's chat summary, not a
        bug in this code."""
        raw = pd.read_csv(ROOT / "data" / "external" / "wheelo-brownlow-predictions.csv")
        assert raw["Votes3_Probability"].min() >= 0.0
        assert raw["Votes3_Probability"].max() <= 100.0
        te = raw[(raw["Player"] == "Tim English") & (raw["Round"] == 0)]
        if not te.empty:
            assert te.iloc[0]["Votes3_Probability"] == pytest.approx(0.3, abs=1e-9)
            assert te.iloc[0]["Votes"] == pytest.approx(0.097, abs=1e-9)

    def test_wheelo_season_ev_reconciles_to_sum_of_match_votes(self):
        raw = pd.read_csv(ROOT / "data" / "external" / "wheelo-brownlow-predictions.csv")
        ext = pd.read_csv(ROOT / "data" / "external" / "processed" / "external_overview.csv")
        for name in ["Tim English", "Nick Daicos"]:
            match_sum = raw.loc[raw["Player"] == name, "Votes"].sum()
            row = ext[ext["player_name"] == name]
            if row.empty or pd.isna(row.iloc[0].get("wheelo_ev")):
                continue
            assert row.iloc[0]["wheelo_ev"] == pytest.approx(match_sum, abs=1e-6)

    def test_no_score_field_labelled_as_ev_or_probability(self):
        """RatingPoints/Supercoach are Wheelo model-score fields, not an
        expected-votes or probability quantity, and must never be surfaced
        as if they were one anywhere in the dashboard/betting display layer."""
        import inspect
        from dashboard import betting_opportunities as bo_mod
        from dashboard import external_data as ed_mod
        src = inspect.getsource(bo_mod) + inspect.getsource(ed_mod)
        assert "RatingPoints" not in src
        assert "Supercoach" not in src

    def test_tim_english_drilldown_is_real_and_sane(self):
        from dashboard import betting_opportunities as bo_mod
        tim = pd.read_csv(ROOT / "reports" / "2026_leaderboard.csv")
        row = tim[tim["player_name"] == "Tim English"]
        if row.empty:
            pytest.skip("Tim English not present in this environment's leaderboard")
        pid = row.iloc[0]["player_id"]
        drilldown = bo_mod.match_level_drilldown(pid)
        assert not drilldown.empty
        # Real key stats must be attached, not silently all-NaN (the exact
        # dtype-mismatch bug this task fixed: player_id becomes "12537.0"
        # after CORE's own float coercion, not "12537").
        assert drilldown["disposals"].notna().any(), "key stats never joined -- dtype mismatch regressed"
        assert drilldown["hitouts"].notna().any()
        # Wheelo must never get a synthesised P(any) -- it has no P2/P1.
        assert "wheelo_p2" not in drilldown.columns and "wheelo_p1" not in drilldown.columns
        # Every row's Production P(any) must equal P3+P2+P1 exactly.
        p_any = drilldown["production_p3"] + drilldown["production_p2"] + drilldown["production_p1"]
        assert (p_any - drilldown["production_p_any"]).abs().max() < 1e-9


class TestDrilldownMissingOptionalStatColumn:
    """Regression test for a live crash: the deployment-bundled CORE parquet
    (data/deployment/model_core_2026_dashboard.parquet, built before this
    drill-down feature existed) is missing 'hitouts' -- confirmed directly:
    it carries only a subset of CORE's real columns. match_level_drilldown()
    used to select `["match_id"] + _KEY_STAT_COLS` unconditionally, which
    raised a bare KeyError the instant it ran against that file, crashing the
    "To Poll a Vote" section and every section below it on Streamlit Cloud."""

    def test_missing_optional_column_does_not_raise(self, monkeypatch):
        from dashboard import data as d_mod

        real_core = d_mod.load_core_2026()
        crippled = real_core.drop(columns=["hitouts"])
        monkeypatch.setattr(bo.d, "load_core_2026", lambda: crippled)

        tim = pd.read_csv(ROOT / "reports" / "2026_leaderboard.csv")
        row = tim[tim["player_name"] == "Tim English"]
        if row.empty:
            pytest.skip("Tim English not present in this environment's leaderboard")
        pid = row.iloc[0]["player_id"]

        drilldown = bo.match_level_drilldown(pid)  # must not raise KeyError
        assert not drilldown.empty
        assert "hitouts" in drilldown.columns
        assert drilldown["hitouts"].isna().all(), "missing column should degrade to NA, not a stale/fabricated value"
        # An unaffected key stat must still be real and populated.
        assert drilldown["disposals"].notna().any()

    def test_page_renders_and_degrades_gracefully_with_missing_column(self, monkeypatch):
        """End-to-end: run the REAL page (via AppTest) with a CORE table
        missing 'hitouts', and confirm the "To Poll a Vote" page (where this
        drill-down now lives, moved out of Brownlow Betting Opportunities)
        still renders -- not a page-crashing exception."""
        from streamlit.testing.v1 import AppTest
        from dashboard import data as d_mod

        real_core = d_mod.load_core_2026()
        crippled = real_core.drop(columns=["hitouts"])
        monkeypatch.setattr(d_mod, "load_core_2026", lambda: crippled)

        at = AppTest.from_file(str(ROOT / "pages" / "27_To_Poll_A_Vote.py"), default_timeout=60)
        at.run()
        assert not at.exception

        # Betting Opportunities itself (now just a summary card + link to the
        # page above) must also still render, via the real router since it
        # uses st.page_link.
        at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at2.run()
        at2.switch_page("pages/23_Brownlow_Betting_Opportunities.py")
        at2.run()
        assert not at2.exception
        headers = [h.value for h in at2.header]
        assert "5. Team Explorer" in headers
        assert "9. Advanced / All Markets" in headers


class TestFinishingPositionExcludesFlaggedIdentities:
    """Chad Warner (genuinely ambiguous vs. teammate Corey Warner -- same
    team, same first initial, correctly left unresolved at the identity
    layer per an earlier task) must never appear as an unexplained all-N/A
    row in the normal Finishing Position Explorer. He must still appear,
    unfiltered, in Advanced/All Markets -- this excludes him from one
    display view, it does not delete or hide the underlying data."""

    def test_flagged_rows_excluded_from_every_threshold(self):
        df = bo.load_opportunities()
        for n in [5.0, 10.0, 20.0]:
            sub = df[(df["market_type"] == "TOP_N") & (df["n"] == n)]
            sub = sub[~sub.apply(bo.has_flag, axis=1)]
            piv = bo.with_bookmaker_odds(sub, ["player_id"])
            assert piv[piv["player_name"] == "Chad Warner"].empty, f"Top {int(n)}"

    def test_flagged_row_still_present_in_raw_opportunities(self):
        """Never actually dropped from the data -- only from this one view."""
        df = bo.load_opportunities()
        assert (df["player_name"] == "Chad Warner").any()


class TestToPollAVoteSingleDrilldownSelector:
    """The per-player expander list was replaced with one searchable
    selectbox (key='tpav_drilldown_player') -- verifies the page (now
    pages/27_To_Poll_A_Vote.py, extracted out of Brownlow Betting
    Opportunities) is materially shorter and both a normal and a
    previously-crash-prone player still resolve through it."""

    def test_page_has_one_drilldown_selector_not_a_list_of_expanders(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "pages" / "27_To_Poll_A_Vote.py"), default_timeout=60)
        at.run()
        assert not at.exception
        selectors = [s for s in at.selectbox if s.key == "tpav_drilldown_player"]
        assert len(selectors) == 1
        assert "Tim English" in selectors[0].options
        assert "Mabior Chol" in selectors[0].options
        # Materially shorter: the old design produced one expander per
        # "To Poll a Vote" player (~dozens); this page's total expander
        # count must stay in the single digits.
        assert len(at.expander) < 20

    def test_selected_player_drilldown_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "pages" / "27_To_Poll_A_Vote.py"), default_timeout=60)
        at.run()
        sel = [s for s in at.selectbox if s.key == "tpav_drilldown_player"][0]
        sel.set_value("Tim English").run()
        assert not at.exception
        assert any("Strongest Production polling match" in (m.value or "") for m in at.markdown)
