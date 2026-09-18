"""
Data-loading and derivation layer for the 2026 Brownlow review dashboard.

STRICT RULE: this module is READ-ONLY over the frozen, audited Phase 4 / 2026
production outputs (reports/2026_*.csv, docs/2026_*.md,
data/processed/model_core_2026.parquet). It never re-fits a model, never
renormalises or rescales a probability, and never changes a predicted vote.
Every function here either (a) loads a production file unchanged, or (b)
derives a purely presentational field (e.g. a human-readable "win by 12"
string, or a top-3 driver label) from columns that already exist in those
files. Derivations are documented inline with the exact source columns used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
PROCESSED = ROOT / "data" / "processed"
DEPLOYMENT = ROOT / "data" / "deployment"


def _resolve_path(local_path: Path, deployment_filename: str) -> Path:
    """Centralised local-file-with-cloud-fallback resolution.

    `data/processed/` is a local-only, gitignored directory (see .gitignore)
    containing large intermediate research artefacts that were never meant to
    be committed. On Streamlit Community Cloud that directory doesn't exist at
    all after a fresh clone, so any loader reading from it directly raises
    FileNotFoundError there. This helper makes every such loader prefer the
    real local file when present (unchanged local-dev behaviour) and fall back
    to a bundled, git-tracked file in data/deployment/ otherwise. The
    deployment file is either an exact copy or an exact column/row subset of
    the same real data -- never a re-derived or altered value -- see
    data/deployment/README.md (build provenance) for how each one was made.
    """
    if local_path.exists():
        return local_path
    fallback = DEPLOYMENT / deployment_filename
    if fallback.exists():
        return fallback
    return local_path  # let the original FileNotFoundError surface with the intended path

# The 8 raw-stat families that have a `_match_z` column in model_core_2026 --
# this is the exact set the prior audit fork used to build
# reports/2026_daicos_round_by_round.csv's "primary_drivers" text. Reused
# here so every player gets drivers built the same, real, documented way.
Z_SCORE_STATS = [
    "disposals",
    "contested_possessions",
    "clearances",
    "tackles",
    "goals",
    "inside_50s",
    "contested_marks",
    "marks",
]


def highlight_objective_stats_nav() -> None:
    """Give each sidebar nav SECTION (MAIN / MODEL ANALYSIS / EXTERNAL
    VALIDATION / ADVANCED) its own subtle accent colour, applied on every
    page (not just while a given page is active) -- name kept for zero-touch
    compatibility with its 27 existing call sites; this now styles the whole
    section-grouped nav, not just one page's link, and the single-page green
    "Objective Stats Model" highlight it used to render has been removed.

    Streamlit's automatic multipage sidebar nav is re-rendered by the
    framework on every page run, and any <style> tag injected by a page
    script is torn down when navigating to a different page's script -- so
    this must be called once near the top of every page (including app.py)
    for the styling to persist while browsing other pages.

    IMPORTANT, found via headless-browser (Playwright) investigation, not
    assumed: this app's sidebar nav DOM is not always grouped the same way.
    Landing on the root page shows the expected 4-section grouped structure
    (one wrapper per section, headers included). But navigating directly to
    a specific page's URL (e.g. opening /Objective_Stats_Model fresh, the
    common case for a bookmark or a refresh) can render a completely FLAT,
    ungrouped list of all pages instead -- with no section-header elements at
    all, in old filename-numeric order, and even including the page that's
    marked visibility="hidden" in app.py's DOM (though still not visibly
    shown, per its own hidden state). A position-based selector
    (`ul > div:nth-of-type(N)`) is therefore NOT reliable across every real
    entry path into this app, so it is deliberately NOT used here -- this is
    a genuine multipage-routing/rendering quirk worth a dedicated look
    separately; it is out of scope for a styling-only change to fix.

    What IS reliable in every render observed: each link's `href` always
    ends with that page's real URL path segment. So every page is matched
    individually by an exact, "/"-anchored suffix (never a bare substring --
    e.g. "/Overview" would otherwise also match "/External_Overview", and
    "/Betting_Opportunities" would otherwise also match
    "/Brownlow_Betting_Opportunities"; anchoring on the leading "/" rules out
    both). Section headers are still coloured too, best-effort, via the same
    positional selector that worked when headers ARE present -- if they
    aren't rendered in a given load (per the quirk above), that rule simply
    matches nothing, which is a harmless no-op, never an error.
    """
    accents = {
        "MAIN": "#3b6fa0",  # navy/blue
        "MODEL_ANALYSIS": "#7c5cbf",  # purple
        "EXTERNAL": "#b8860b",  # amber/gold
        "ADVANCED": "#64748b",  # slate/grey
    }
    # Must mirror app.py's page->section membership exactly (checked against
    # its current MAIN/MODEL ANALYSIS/EXTERNAL VALIDATION/ADVANCED lists).
    section_pages = {
        "MAIN": ["/", "Technical_Summary", "Guide_And_FAQs", "Finishing_Order",
                 "Brownlow_Betting_Opportunities", "To_Poll_A_Vote", "Match_Detail",
                 "Brownlow_Night_Tracker"],
        "MODEL_ANALYSIS": ["Overview", "Objective_Stats_Model", "Model_Agreement",
                            "Order_Scenarios", "Player_H2H", "Multi_Player_Comparison",
                            "Team_Player_Rankings"],
        "EXTERNAL": ["External_Overview", "External_Player_Comparison",
                     "External_Winning_Order", "External_Leader_After_Round"],
        "ADVANCED": ["Player_Detail", "Scenario_Comparison", "Model_Disagreement",
                     "Round_View", "Defender_Bias_Watchlist", "Projection_Concentration",
                     "Uncertainty", "Round_By_Round_Leaderboard", "Team_Breakdown"],
    }
    rules = []
    for section, colour in accents.items():
        selectors = ", ".join(
            f'[data-testid="stSidebarNavLink"][href$="/"]' if p == "/"
            else f'[data-testid="stSidebarNavLink"][href$="/{p}"]'
            for p in section_pages[section]
        )
        rules.append(f"""
        {selectors} {{ border-radius: 8px; }}
        """)
        for p in section_pages[section]:
            sel = '[href$="/"]' if p == "/" else f'[href$="/{p}"]'
            rules.append(f"""
            [data-testid="stSidebarNavLink"]{sel}:hover {{ background-color: {colour}14; }}
            [data-testid="stSidebarNavLink"]{sel}[aria-current="page"] {{ background-color: {colour}2E; }}
            """)
        # Best-effort section-header colouring (see docstring: only applies
        # when the grouped DOM structure is actually present).
        idx = list(accents).index(section) + 1
        rules.append(f"""
        [data-testid="stSidebarNav"] ul > div:nth-of-type({idx}) [data-testid="stNavSectionHeader"] {{
            color: {colour};
            border-left: 3px solid {colour};
            padding-left: 8px;
        }}
        """)
    st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def gradient_style(s: pd.Series, rgb: tuple[int, int, int] = (74, 144, 217)) -> list:
    """Lightweight replacement for pandas Styler.background_gradient (which
    requires matplotlib, a dependency we deliberately avoid here). Linearly
    interpolates a single-hue background from the column's min to its max."""
    vals = pd.to_numeric(s, errors="coerce")
    lo, hi = vals.min(), vals.max()
    span = (hi - lo) or 1.0
    styles = []
    for v in vals:
        if pd.isna(v):
            styles.append("")
            continue
        t = max(0.0, min(1.0, (v - lo) / span))
        alpha = 0.12 + 0.55 * t
        styles.append(f"background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})")
    return styles


def _display_team(team_id: str) -> str:
    if not isinstance(team_id, str):
        return str(team_id)
    return team_id.replace("_", " ").title()


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_leaderboard.csv")


@st.cache_data
def load_simulation_summary() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_simulation_summary.csv")


@st.cache_data
def load_match_probabilities() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_match_probabilities.csv")


@st.cache_data
def load_predicted_votes() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_predicted_votes.csv")


@st.cache_data
def load_scenario_comparison() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_scenario_comparison.csv")


@st.cache_data
def load_round_contenders() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_round_by_round_contenders.csv")


@st.cache_data
def load_model_disagreement() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_model_disagreement.csv")


@st.cache_data
def load_daicos_round_by_round() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_daicos_round_by_round.csv")


@st.cache_data
def load_top10_probability_audit() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_top10_probability_audit.csv")


@st.cache_data
def load_quality_checks() -> dict:
    import json

    with open(REPORTS / "2026_quality_checks.json") as f:
        return json.load(f)


@st.cache_data
def load_core_2026() -> pd.DataFrame:
    """Player-match rows for the 2026 season only, from the frozen production
    feature table used to build the audited forecast."""
    path = _resolve_path(PROCESSED / "model_core_2026.parquet", "model_core_2026_dashboard.parquet")
    df = pd.read_parquet(path)
    df = df[df["season"] == 2026].reset_index(drop=True)
    # player_id is stored as string in this parquet but as int64 in the
    # reports/2026_*.csv files -- normalise to numeric so joins/filters by
    # player_id work consistently across every source used by this dashboard.
    # A small number of rows carry an unresolved placeholder id (e.g.
    # "NOID2026_144", a pre-existing identity-resolution gap documented in
    # Phase 2/4, not introduced here) -- these become NaN and are simply
    # excluded from any player_id-keyed lookup, which is the correct behaviour
    # since they were never resolvable to a real player anyway.
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    return df


@st.cache_data
def load_defender_watchlist() -> pd.DataFrame:
    """Parses the existing markdown table in docs/2026_FINAL_AUDIT.md ("Specific
    2026 games where an elite defensive performance looks undervalued") rather
    than hardcoding the list, so the dashboard always reflects the actual
    audited file."""
    text = (DOCS / "2026_FINAL_AUDIT.md").read_text()
    marker = "Specific 2026 games where an elite defensive performance looks undervalued"
    idx = text.index(marker)
    # Take only the contiguous run of "|"-prefixed lines immediately after the
    # marker (the one table this heading introduces) -- stop at the first
    # non-table line so later, differently-themed tables in the same doc
    # (which may coincidentally share a column count) are never picked up.
    remainder_lines = text[idx:].splitlines()
    table_lines = []
    started = False
    for ln in remainder_lines:
        if ln.strip().startswith("|"):
            table_lines.append(ln)
            started = True
        elif started:
            break
    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for ln in table_lines[2:]:  # skip header + separator row
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(cells)
    watchlist = pd.DataFrame(rows, columns=header)
    for col in ("round", "disposals", "contested_marks", "one_percenters", "expected_votes"):
        if col in watchlist.columns:
            watchlist[col] = pd.to_numeric(watchlist[col], errors="coerce")
    watchlist["reason_flagged"] = (
        "Elite defensive output (top 5% league-wide by disposals + 2x contested marks "
        "+ 0.5x one-percenters) with model expected votes < 0.5 -- consistent with the "
        "Phase 4 finding that key defenders are identified as 3-vote winners only 12.5% "
        "of the time vs. 62% for midfielders."
    )
    return watchlist


def match_result_string(row: pd.Series) -> str:
    """Derives a 'win by 12' / 'loss by 6' / 'draw by 0' string from the
    existing win_loss_draw + absolute_margin columns in model_core_2026 --
    the same derivation used to build 2026_daicos_round_by_round.csv."""
    outcome = str(row.get("win_loss_draw", "")).lower()
    margin = row.get("absolute_margin", 0)
    try:
        margin = int(round(float(margin)))
    except (TypeError, ValueError):
        margin = 0
    if outcome.startswith("w"):
        return f"win by {margin}"
    if outcome.startswith("l"):
        return f"loss by {margin}"
    return f"draw by {margin}"


def primary_drivers_string(row: pd.Series, top_n: int = 3) -> str:
    """Top-N stat families by within-match z-score (model_core_2026's
    `<stat>_match_z` columns), formatted 'stat=raw_value (z=z_value)' --
    identical method to the audited reports/2026_daicos_round_by_round.csv,
    generalised to any player-match row."""
    scored = []
    for stat in Z_SCORE_STATS:
        zcol = f"{stat}_match_z"
        if zcol not in row.index or pd.isna(row[zcol]):
            continue
        scored.append((stat, row.get(stat), row[zcol]))
    scored.sort(key=lambda t: t[2], reverse=True)
    parts = []
    for stat, raw, z in scored[:top_n]:
        label = stat.replace("_", " ")
        parts.append(f"{label}={raw:g} (z={z:.1f})")
    return "; ".join(parts) if parts else "insufficient stat history"


@st.cache_data
def build_player_round_by_round(player_id: int) -> pd.DataFrame:
    """Generic, reusable version of the Daicos-specific audit CSV -- works for
    any player_id. Joins model_core_2026 (raw stats, result, z-scores) with
    reports/2026_match_probabilities.csv (p3/p2/p1/p0/expected_votes) and
    reports/2026_predicted_votes.csv (deterministic pick). No probability is
    recomputed or renormalised -- p3/p2/p1/p0/expected_votes are copied
    verbatim from the frozen production file.

    A small number of players (a pre-existing, documented production-pipeline
    gap -- see docs/2026_OBJECTIVE_MODEL.md section 11 and
    docs/2026_BROWNLOW_REVIEW_REPORT.md section 7 -- affects
    Alex Neal-Bullen, Jason Horne-Francis, Darcy Byrne-Jones,
    Nasiah Wanganeen-Milera, and Luke Davies-Uniacke as of this build) have
    real rows in model_core_2026 (they played real 2026 matches) but ZERO
    rows in reports/2026_match_probabilities.csv / 2026_predicted_votes.csv,
    because their Scenario C/ensemble/simulation step never resolved. Without
    the guard below, the left-merges here would silently keep all of that
    player's CORE rows with p3/p2/p1/expected_votes = NaN -- a non-empty
    dataframe that looks like real (if unimpressive) data, which is exactly
    the bug this function used to have: every page that gates on `.empty` to
    show a "no data resolved" warning instead rendered a NaN/0-filled table
    for these players. Detect the "no real production probability data at
    all for this player" case explicitly and return truly empty (same
    columns) so that existing gate works as intended, generically, for any
    player this affects now or in the future -- not a name-specific patch."""
    core = load_core_2026()
    prow = core[core["player_id"] == player_id].copy()
    if prow.empty:
        return prow

    mp = load_match_probabilities()
    mp_player = mp[mp["player_id"] == player_id][
        ["match_id", "p3", "p2", "p1", "p0", "expected_votes"]
    ]
    if mp_player.empty:
        # Real player, real CORE rows, but no production probability data
        # exists anywhere for them -- return empty (not NaN-filled) so
        # callers' `.empty` checks correctly show the "no data" warning.
        return prow.iloc[0:0]
    pv = load_predicted_votes()
    pv_player = pv[pv["player_id"] == player_id][["match_id", "predicted_votes"]]

    out = prow.merge(mp_player, on="match_id", how="left").merge(
        pv_player, on="match_id", how="left"
    )
    out["predicted_votes"] = out["predicted_votes"].fillna(0).astype(int)
    out["result"] = out.apply(match_result_string, axis=1)
    out["primary_drivers"] = out.apply(primary_drivers_string, axis=1)
    out["opponent_display"] = out["opponent_id"].map(_display_team)
    out = out.sort_values("round")
    return out[
        [
            "round",
            "opponent_id",
            "opponent_display",
            "result",
            "disposals",
            "goals",
            "p3",
            "p2",
            "p1",
            "p0",
            "expected_votes",
            "predicted_votes",
            "primary_drivers",
        ]
    ].rename(columns={"predicted_votes": "deterministic_pick"})


def classify_significant_games(rbr: pd.DataFrame) -> dict:
    """Auto-classifies a player's round-by-round rows into the categories the
    brief asks for. Thresholds are simple, explicit and documented here (not
    fit to data) -- purely a presentational bucketing of existing p3/p2/p1/
    expected_votes/z-score columns, never a new model."""
    if rbr.empty:
        return {k: rbr for k in (
            "high_confidence_3", "likely_2_3", "borderline",
            "surprising", "zero_despite_strong",
        )}

    def top_z(drivers: str) -> float:
        m = re.search(r"z=(-?\d+\.?\d*)", drivers)
        return float(m.group(1)) if m else 0.0

    zscore = rbr["primary_drivers"].map(top_z)

    high_conf_3 = rbr[rbr["p3"] >= 0.55]
    likely_2_3 = rbr[(rbr["expected_votes"] >= 1.5) & (rbr["expected_votes"] < 2.5) & ~rbr.index.isin(high_conf_3.index)]
    borderline = rbr[(rbr["expected_votes"] >= 0.8) & (rbr["expected_votes"] < 1.5)]
    surprising = rbr[(rbr["expected_votes"] >= 1.0) & (rbr["deterministic_pick"] == 0)]
    zero_despite_strong = rbr[(zscore >= 1.5) & (rbr["expected_votes"] < 0.5)]

    return {
        "high_confidence_3": high_conf_3,
        "likely_2_3": likely_2_3,
        "borderline": borderline,
        "surprising": surprising,
        "zero_despite_strong": zero_despite_strong,
    }


def projection_concentration(rbr: pd.DataFrame) -> pd.DataFrame:
    """Generic EV-bucket breakdown (section 11 / "Daicos audit view",
    generalised to any player). Buckets are cumulative-from-above (a game
    with EV=3.0 counts in every threshold at or below it) except the final
    zero-vote row."""
    thresholds = [2.5, 2.0, 1.5, 1.0]
    rows = []
    for t in thresholds:
        sub = rbr[rbr["expected_votes"] >= t]
        rows.append({"bucket": f"EV >= {t}", "n_games": len(sub), "ev_contribution": sub["expected_votes"].sum()})
    zero = rbr[rbr["expected_votes"] < 0.05]
    rows.append({"bucket": "EV ~ 0", "n_games": len(zero), "ev_contribution": zero["expected_votes"].sum()})
    return pd.DataFrame(rows)


@st.cache_data
def match_meta_table() -> pd.DataFrame:
    """One row per match_id with both teams' scores/venue/date, derived by
    collapsing model_core_2026's per-player-per-team rows down to team level.
    No probability or vote data is touched here."""
    core = load_core_2026()
    team_rows = core.drop_duplicates(subset=["match_id", "team_id"])[
        ["match_id", "season", "round", "date", "venue", "team_id", "team_score"]
    ]
    meta = []
    for match_id, g in team_rows.groupby("match_id"):
        if len(g) != 2:
            continue
        g = g.sort_values("team_score", ascending=False)
        winner, loser = g.iloc[0], g.iloc[1]
        margin = winner["team_score"] - loser["team_score"]
        meta.append(
            {
                "match_id": match_id,
                "season": winner["season"],
                "round": winner["round"],
                "date": winner["date"],
                "venue": winner["venue"],
                "team_a": winner["team_id"],
                "team_a_score": winner["team_score"],
                "team_b": loser["team_id"],
                "team_b_score": loser["team_score"],
                "margin": margin,
                "result_label": (
                    f"{_display_team(winner['team_id'])} {winner['team_score']:g} - "
                    f"{loser['team_score']:g} {_display_team(loser['team_id'])}"
                    + (" (draw)" if margin == 0 else f" ({_display_team(winner['team_id'])} by {margin:g})")
                ),
            }
        )
    return pd.DataFrame(meta).sort_values(["round", "match_id"]).reset_index(drop=True)


@st.cache_data
def load_scenario_predictions_2026() -> pd.DataFrame:
    """Round-level, per-scenario expected-vote predictions
    (data/processed/scenario_predictions_2026.parquet) -- the real match-level
    detail behind the season-aggregate reports/2026_scenario_comparison.csv.
    Contains A_historical, A_with_reputation, B_recent_era, C_stats_assisted
    only (the D structural-break sensitivity bands and FINAL_ENSEMBLE are
    computed at the season level via a probability-space blend documented in
    docs/2026_MODELLING_METHODOLOGY.md -- not a simple weighted average of
    these columns, so this dashboard does not attempt to reconstruct a
    round-level "final ensemble EV" from them, to avoid presenting an
    unverified number as if it were the audited pipeline's output)."""
    path = _resolve_path(PROCESSED / "scenario_predictions_2026.parquet", "scenario_predictions_2026.parquet")
    df = pd.read_parquet(path)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    return df


@st.cache_data
def load_contender_probabilities(model_label: str) -> pd.DataFrame:
    """Winner/Top2/Top3/Top5/Top7/Top10 probability + mean_votes per player,
    for the Production or Objective model's season Monte Carlo.

    Locally, computes this fresh from the real raw per-simulation totals
    (data/processed/mc_totals[_objective]_2026.npy, 100,000 / 20,000 draws) via
    the exact same src.models.order_scenarios.contender_probabilities used to
    build reports/2026_order_scenarios.csv. Those .npy files are large
    (113MB / 30MB) local-only research artefacts, not meant to be shipped to
    Streamlit Cloud -- so when they're not present, this instead reads a
    precomputed CSV in data/deployment/ containing this exact function's
    output, computed once from the same real arrays (see
    data/deployment/README.md). Either path returns identical numbers; no
    probability is invented or recomputed differently for deployment.
    """
    from src.models.order_scenarios import contender_probabilities

    npy_name = "mc_totals_2026.npy" if model_label == "Production" else "mc_totals_objective_2026.npy"
    idx_name = "2026_mc_player_index.csv" if model_label == "Production" else "2026_objective_mc_player_index.csv"
    npy_path = PROCESSED / npy_name
    if npy_path.exists():
        totals = np.load(npy_path)
        players = pd.read_csv(REPORTS / idx_name)
        return contender_probabilities(totals, players)

    fallback_name = ("contender_probabilities_production.csv" if model_label == "Production"
                      else "contender_probabilities_objective.csv")
    return pd.read_csv(DEPLOYMENT / fallback_name)


@st.cache_data
def round_level_disagreement_table(player_ids: tuple[int, ...]) -> pd.DataFrame:
    """Per player-round: A_historical / B_recent_era / C_stats_assisted
    expected votes (real production values) and a simple, transparent
    round-level disagreement = max - min across those three scenarios."""
    sp = load_scenario_predictions_2026()
    sp = sp[sp["player_id"].isin(player_ids) & sp["scenario"].isin(["A_historical", "B_recent_era", "C_stats_assisted"])]
    pivot = sp.pivot_table(
        index=["player_id", "player_name", "team_id", "round", "match_id"],
        columns="scenario", values="expected_votes",
    ).reset_index()
    scen_cols = ["A_historical", "B_recent_era", "C_stats_assisted"]
    pivot["round_disagreement"] = pivot[scen_cols].max(axis=1) - pivot[scen_cols].min(axis=1)
    return pivot.sort_values("round_disagreement", ascending=False)


@st.cache_data
def team_list() -> list[str]:
    return sorted(load_leaderboard()["team_id"].unique().tolist())


@st.cache_data
def team_breakdown(team_id: str) -> pd.DataFrame:
    """Per-player season summary for one team, built entirely from existing
    reports/2026_leaderboard.csv columns (FINAL_ENSEMBLE, sim_median_votes,
    projected_*_vote_games) -- no recomputation. Adds a "Share of Team
    Expected Votes" column (player FINAL_ENSEMBLE / team total FINAL_ENSEMBLE)."""
    lb = load_leaderboard()
    team = lb[lb["team_id"] == team_id].copy()
    team_total = team["FINAL_ENSEMBLE"].sum()
    team["share_of_team_ev"] = team["FINAL_ENSEMBLE"] / team_total if team_total else 0.0
    return team.sort_values("FINAL_ENSEMBLE", ascending=False).reset_index(drop=True)


def team_concentration(team_breakdown_df: pd.DataFrame) -> float:
    """Herfindahl-style concentration of a team's expected votes across its
    players: sum of each player's (share of team EV)^2. Ranges from ~1/n
    (perfectly even) to 1.0 (one player holds the entire team total)."""
    return float((team_breakdown_df["share_of_team_ev"] ** 2).sum())


# Minimum expected votes for a team player-match row to be considered
# "material" enough to list in the Team Breakdown round-by-round table --
# reuses the same 0.8 EV floor that classify_significant_games() already
# uses as its "borderline polling game" threshold, so no new cutoff is
# invented here.
TEAM_ROUND_EV_FLOOR = 0.8


@st.cache_data
def team_round_by_round(team_id: str) -> pd.DataFrame:
    """Round, opponent, player, EV, deterministic pick, p3/p2/p1 for every
    player-match row belonging to `team_id` with expected_votes >=
    TEAM_ROUND_EV_FLOOR. Joins reports/2026_match_probabilities.csv (for
    every player's p3/p2/p1/expected_votes) with match_meta_table() (for the
    opponent) and reports/2026_predicted_votes.csv (for the deterministic
    pick, defaulting to 0 for rows with no deterministic vote)."""
    mp = load_match_probabilities()
    team_rows = mp[(mp["team_id"] == team_id) & (mp["expected_votes"] >= TEAM_ROUND_EV_FLOOR)].copy()

    meta = match_meta_table()[["match_id", "team_a", "team_b"]]
    team_rows = team_rows.merge(meta, on="match_id", how="left")
    team_rows["opponent_id"] = team_rows.apply(
        lambda r: r["team_b"] if r["team_a"] == team_id else r["team_a"], axis=1
    )
    team_rows["opponent_display"] = team_rows["opponent_id"].map(_display_team)

    pv = load_predicted_votes()[["match_id", "player_id", "predicted_votes"]]
    team_rows = team_rows.merge(pv, on=["match_id", "player_id"], how="left")
    team_rows["predicted_votes"] = team_rows["predicted_votes"].fillna(0).astype(int)

    return team_rows.sort_values(["round", "expected_votes"], ascending=[True, False])[
        [
            "round",
            "opponent_display",
            "player_name",
            "expected_votes",
            "predicted_votes",
            "p3",
            "p2",
            "p1",
        ]
    ].rename(columns={"opponent_display": "opponent", "predicted_votes": "most_likely_votes"})


def relative_ranks_for_match(match_id: str) -> pd.DataFrame:
    """Team-relative rank/share columns already computed in model_core_2026
    (e.g. disposals_team_rank, clearances_team_share) for every player in a
    given match -- used by the match detail view."""
    core = load_core_2026()
    cols = ["player_id", "player_name", "team_id", "role"]
    for stat in ("disposals", "contested_possessions", "clearances", "tackles", "goals"):
        cols += [f"{stat}_team_rank", f"{stat}_team_share"]
    return core[core["match_id"] == match_id][cols]


# --------------------------------------------------------------------------
# 2026 Objective Stats Model (Experimental) -- loaders only. All scoring/
# weighting logic lives in src/models/objective_stats_model.py and
# src/models/build_2026_objective_outputs.py, kept separate from this
# production-dashboard data layer; these functions just read the CSVs that
# pipeline already produced, exactly like every other loader in this file.
# --------------------------------------------------------------------------

@st.cache_data
def load_objective_leaderboard() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_objective_leaderboard.csv")


@st.cache_data
def load_objective_vs_production() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_objective_vs_production.csv")


@st.cache_data
def load_objective_votes() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_objective_votes.csv")


@st.cache_data
def load_objective_match_scores() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_objective_match_scores.csv")


@st.cache_data
def load_dual_model_comparison() -> pd.DataFrame:
    """Canonical, single-source-of-truth Production-vs-Objective per-player
    comparison. Every page comparing the two models (H2H, multi-player,
    team rankings, and -- ideally -- Model Agreement) should read this
    instead of re-joining reports/2026_leaderboard.csv and
    reports/2026_objective_leaderboard.csv inline, to avoid two copies of the
    same join logic drifting apart.

    player_id is the join key: Production's leaderboard stores it as int64,
    Objective's as str, but the underlying values are the same real ids
    (confirmed: both pipelines assign ids from the same 2026 CORE player
    identities) -- normalised to str once, here, at the loader boundary.

    This is an OUTER join, not inner: Objective's 2026 CORE set includes ~167
    players excluded from Production's ensemble by the season-to-date/Round-1
    exclusion documented in docs/2026_DATA_VALIDATION.md (a player's first
    game of the season has no prior-round rolling-average feature to score
    on). Those players get NaN production_ev/production_rank -- callers MUST
    render that as an explicit "not available in Production" state, never a
    bare 0, since 0 would misrepresent "no votes" as "unscoreable"."""
    prod = load_leaderboard()[["player_id", "player_name", "team_id", "rank", "FINAL_ENSEMBLE"]].copy()
    obj = load_objective_leaderboard()[["player_id", "player_name", "team_id", "rank", "objective_ev"]].copy()
    prod["player_id"] = prod["player_id"].astype(str)
    obj["player_id"] = obj["player_id"].astype(str)
    prod = prod.rename(columns={
        "player_name": "player_name_prod", "team_id": "team_id_prod",
        "rank": "production_rank", "FINAL_ENSEMBLE": "production_ev",
    })
    obj = obj.rename(columns={
        "player_name": "player_name_obj", "team_id": "team_id_obj",
        "rank": "objective_rank", "objective_ev": "objective_ev",
    })
    merged = prod.merge(obj, on="player_id", how="outer")

    assert not merged["player_id"].duplicated().any(), (
        "duplicate player_id after Production/Objective merge -- identity join is unsafe"
    )
    team_mismatch = merged[merged["team_id_prod"].notna() & merged["team_id_obj"].notna() & (merged["team_id_prod"] != merged["team_id_obj"])]
    assert team_mismatch.empty, f"team_id disagrees between models for shared players: {team_mismatch['player_id'].tolist()}"

    merged["player_name"] = merged["player_name_prod"].fillna(merged["player_name_obj"])
    merged["team_id"] = merged["team_id_prod"].fillna(merged["team_id_obj"])
    merged["in_production"] = merged["production_ev"].notna()
    merged["in_objective"] = merged["objective_ev"].notna()
    merged["model_difference"] = merged["production_ev"] - merged["objective_ev"]
    merged["absolute_difference"] = merged["model_difference"].abs()
    merged["average_ev"] = merged[["production_ev", "objective_ev"]].mean(axis=1, skipna=True)

    out = merged[[
        "player_id", "player_name", "team_id",
        "production_ev", "production_rank", "in_production",
        "objective_ev", "objective_rank", "in_objective",
        "model_difference", "absolute_difference", "average_ev",
    ]].sort_values("player_id").reset_index(drop=True)
    return out


def dual_model_team_rankings(team_id: str) -> pd.DataFrame:
    """The dual-model comparison restricted to one team, with TEAM-relative
    ranks computed here (on full-precision EV, not a display-rounded value) --
    distinct from production_rank/objective_rank above, which are each
    model's GLOBAL (whole-competition) rank."""
    df = load_dual_model_comparison()
    sub = df[df["team_id"] == team_id].copy()
    sub["production_team_rank"] = sub["production_ev"].rank(ascending=False, method="min")
    sub["objective_team_rank"] = sub["objective_ev"].rank(ascending=False, method="min")
    return sub


@st.cache_data
def build_player_objective_round_by_round(player_id) -> pd.DataFrame:
    """Round-by-round objective vs. production EV/3-2-1/difference for one
    player, joining the objective votes file with the production predicted-
    votes file on match_id + player_id (both real, already-computed sources --
    no probability is recomputed here)."""
    obj = load_objective_votes()
    obj = obj[obj["player_id"].astype(str) == str(player_id)].copy()
    prod = load_predicted_votes()
    prod = prod[prod["player_id"].astype(str) == str(player_id)][
        ["match_id", "expected_votes", "predicted_votes"]
    ].rename(columns={"expected_votes": "production_ev", "predicted_votes": "production_pred_votes"})
    merged = obj.merge(prod, on="match_id", how="left")
    merged["ev_difference"] = merged["expected_votes"] - merged["production_ev"]
    return merged.rename(columns={
        "expected_votes": "objective_ev", "objective_pred_votes": "objective_pred_votes",
    }).sort_values("round")
