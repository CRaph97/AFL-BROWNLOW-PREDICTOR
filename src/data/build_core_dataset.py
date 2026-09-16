"""
Build the canonical CORE player-match dataset (Phase 2B schema) from the raw
afltables-sourced fitzRoy data (afldata.rda), for the validated target window
of 1984-2025 home-and-away matches (see docs/TARGET_VALIDATION.md for why this
window and not something wider).

Grain: ONE ROW = ONE PLAYER IN ONE MATCH.

This script does NOT compute any match-relative / derived features (per the
Phase 2B instruction to keep the raw schema ready for that, but not build it
yet). It only: filters to scope, canonicalises identifiers, renames to a
documented schema, and validates the target variable's structural integrity.

Raw data is never modified in place: this reads from data/raw/ and writes to
data/interim/ (long, all-columns) and data/processed/ (the canonical schema).
"""
import json
from pathlib import Path

import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "fitzroy_data"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "config"

FINALS_ROUNDS = {"SF", "PF", "GF", "QF", "EF"}
CORE_START_SEASON = 1984
CORE_END_SEASON = 2025  # 2026 excluded: in-progress season, Brownlow votes not yet revealed (see docs/TARGET_VALIDATION.md)

# afldata.rda column -> canonical snake_case column (Phase 2B schema groups noted in comments)
COLUMN_RENAME = {
    # identifiers / context
    "Season": "season",
    "Round": "round",
    "Date": "date",
    "Venue": "venue",
    "Home.team": "home_team_raw",
    "Away.team": "away_team_raw",
    "Playing.for": "team_raw",
    "Home.Away": "home_away",
    "First.name": "player_first_name",
    "Surname": "player_surname",
    "ID": "player_id_afltables",
    "Jumper.No.": "jumper_number",
    # match context
    "Home.score": "home_score",
    "Away.score": "away_score",
    # player stats (only fields reliably present across the window; NaN where structurally unavailable)
    "Kicks": "kicks",
    "Marks": "marks",
    "Handballs": "handballs",
    "Disposals": "disposals",
    "Goals": "goals",
    "Behinds": "behinds",
    "Hit.Outs": "hitouts",
    "Tackles": "tackles",
    "Rebounds": "rebound_50s",
    "Inside.50s": "inside_50s",
    "Clearances": "clearances",
    "Clangers": "clangers",
    "Frees.For": "frees_for",
    "Frees.Against": "frees_against",
    "Contested.Possessions": "contested_possessions",
    "Uncontested.Possessions": "uncontested_possessions",
    "Contested.Marks": "contested_marks",
    "Marks.Inside.50": "marks_inside_50",
    "One.Percenters": "one_percenters",
    "Bounces": "bounces",
    "Goal.Assists": "goal_assists",
    "Time.on.Ground": "time_on_ground_pct",
    "Substitute": "substitute_status",
    # target
    "Brownlow.Votes": "brownlow_votes",
}

KEEP_RAW_COLUMNS = list(COLUMN_RENAME.keys())


def load_team_mapping() -> dict:
    df = pd.read_csv(CONFIG_DIR / "team_mapping.csv")
    return dict(zip(df["source_name"], df["canonical_team_id"]))


def build() -> pd.DataFrame:
    raw = pyreadr.read_r(RAW_DIR / "afldata.rda")["afldata"]

    df = raw[KEEP_RAW_COLUMNS].rename(columns=COLUMN_RENAME).copy()
    df["brownlow_votes"] = pd.to_numeric(df["brownlow_votes"], errors="coerce")

    # --- scope filter: home-and-away matches, 1984-2025 (see docs/TARGET_VALIDATION.md) ---
    is_final = df["round"].isin(FINALS_ROUNDS)
    in_window = df["season"].between(CORE_START_SEASON, CORE_END_SEASON)
    core = df[in_window & ~is_final].copy()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    core.to_parquet(INTERIM_DIR / "player_match_afltables_1984_2025.parquet", index=False)

    # --- canonical identifiers ---
    team_map = load_team_mapping()
    unmapped_teams = set(core["team_raw"]) | set(core["home_team_raw"]) | set(core["away_team_raw"])
    unmapped_teams -= set(team_map.keys())
    if unmapped_teams:
        raise ValueError(f"Unmapped team names found -- update config/team_mapping.csv: {unmapped_teams}")

    core["team_id"] = core["team_raw"].map(team_map)
    core["home_team_id"] = core["home_team_raw"].map(team_map)
    core["away_team_id"] = core["away_team_raw"].map(team_map)
    core["opponent_id"] = core.apply(
        lambda r: r["away_team_id"] if r["team_id"] == r["home_team_id"] else r["home_team_id"], axis=1
    )

    # Player identity: prefer the afltables numeric ID (stable across trades/seasons -- see
    # docs/TARGET_VALIDATION.md identity checks). A small number of very recent rows (79 rows,
    # all 2025, at time of writing) have a missing ID because the debutant hasn't been added to
    # the afltables ID crosswalk yet. These are NOT silently fuzzy-matched: each gets a row-unique
    # placeholder ID and is logged to reports/identity_review.csv for manual review, per the
    # explicit instruction to never silently resolve ambiguous player identities.
    core = core.reset_index(drop=True)
    missing_id_mask = core["player_id_afltables"].isna()
    core["player_id"] = core["player_id_afltables"].astype("Int64").astype(str)
    if missing_id_mask.any():
        placeholder_ids = [f"NOID_{i}" for i in core.index[missing_id_mask]]
        core.loc[missing_id_mask, "player_id"] = placeholder_ids

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        review_cols = ["season", "round", "date", "team_raw", "player_first_name", "player_surname", "player_id"]
        core.loc[missing_id_mask, review_cols].to_csv(REPORTS_DIR / "identity_review_missing_id.csv", index=False)
        print(f"WARNING: {missing_id_mask.sum()} rows had no afltables player ID -- "
              f"logged to reports/identity_review_missing_id.csv for manual review, "
              f"placeholder IDs assigned (not merged with any other player).")

    core["player_name"] = core["player_first_name"] + " " + core["player_surname"]

    # --- match context ---
    core["team_score"] = core.apply(lambda r: r["home_score"] if r["team_id"] == r["home_team_id"] else r["away_score"], axis=1)
    core["opponent_score"] = core.apply(lambda r: r["away_score"] if r["team_id"] == r["home_team_id"] else r["home_score"], axis=1)
    core["margin"] = core["team_score"] - core["opponent_score"]
    core["absolute_margin"] = core["margin"].abs()
    core["win_loss_draw"] = core["margin"].apply(lambda m: "win" if m > 0 else ("loss" if m < 0 else "draw"))

    core["match_id"] = (
        core["season"].astype(str) + "_R" + core["round"].astype(str) + "_"
        + core["home_team_id"] + "_v_" + core["away_team_id"] + "_" + core["date"].astype(str)
    )

    # --- target-variable structural integrity (see docs/TARGET_VALIDATION.md) ---
    _validate_target(core)

    final_columns = [
        # identifiers
        "season", "round", "match_id", "date", "venue",
        "player_id", "player_name", "team_id", "opponent_id", "home_away",
        # match context
        "team_score", "opponent_score", "margin", "win_loss_draw", "absolute_margin",
        # player stats
        "kicks", "marks", "handballs", "disposals", "goals", "behinds", "hitouts",
        "tackles", "rebound_50s", "inside_50s", "clearances", "clangers",
        "frees_for", "frees_against", "contested_possessions", "uncontested_possessions",
        "contested_marks", "marks_inside_50", "one_percenters", "bounces", "goal_assists",
        "time_on_ground_pct", "substitute_status", "jumper_number",
        # target
        "brownlow_votes",
    ]
    out = core[final_columns].copy()
    out["brownlow_votes"] = out["brownlow_votes"].astype("Int64")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet", index=False)
    print(f"Wrote {len(out):,} rows, {out['match_id'].nunique():,} matches -> "
          f"data/processed/player_match_core_1984_2025.parquet")
    return out


def _validate_target(core: pd.DataFrame) -> None:
    """Assert the one-3/one-2/one-1/sum-6 constraint holds for every match. Raises on failure --
    this is a hard gate, not a warning, because everything downstream depends on it."""
    g = core.groupby("match_id")["brownlow_votes"].agg(
        n3=lambda s: (s == 3).sum(),
        n2=lambda s: (s == 2).sum(),
        n1=lambda s: (s == 1).sum(),
        total=lambda s: s.sum(),
    )
    bad = g[~((g.n3 == 1) & (g.n2 == 1) & (g.n1 == 1) & (g.total == 6))]
    if len(bad) > 0:
        bad_path = REPORTS_DIR / "target_integrity_failures.csv"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        bad.to_csv(bad_path)
        raise AssertionError(
            f"{len(bad)} matches violate the one-3/one-2/one-1/sum-6 Brownlow constraint. "
            f"Details written to {bad_path}. Do not proceed without resolving or explicitly "
            f"documenting these as historical exceptions."
        )
    print(f"Target integrity check passed: all {len(g):,} matches have exactly one 3, one 2, "
          f"one 1 vote-getter and sum to 6.")


if __name__ == "__main__":
    build()
