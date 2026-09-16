"""
Join footywire-sourced advanced player stats (player_stats.rda, 2010-2026) onto
the CORE canonical dataset (player_match_core_1984_2025.parquet).

CRITICAL: this join deliberately does NOT use (season, round, team) as the key.
Phase 2 investigation found that footywire's round numbering is NOT always
consistent with afltables' round numbering in the same season -- e.g. the
2024-05-12 Adelaide v Brisbane Lions match is "Round 10" in afldata (afltables)
but "Round 9" in player_stats (footywire), because footywire's source data
counts the season-opening "Round 0" while afltables labels it "Round 1",
shifting every later round number by one. This is documented in
docs/DATA_COVERAGE.md and docs/TARGET_VALIDATION.md.

The robust, source-agnostic join key used here is (season, date, canonical
team_id, canonical opponent_id) -- a specific calendar date pins down a unique
match far more reliably than a round label across sources.

Player-level join uses (match key, player_surname + team) since player_stats
does not carry the same afltables numeric player ID. First-name-only or
surname-only join keys risk collisions for common surnames -- see
docs/DATA_DICTIONARY.md for the exact key and its known limitations (this is a
best-effort join for the "ADVANCED" feature layer, not the target-bearing CORE
table, so a small unmatched-row rate here does not threaten target integrity).
"""
from pathlib import Path

import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "fitzroy_data"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "config"

ADV_COLUMN_RENAME = {
    "CP": "adv_contested_possessions",
    "UP": "adv_uncontested_possessions",
    "ED": "effective_disposals",
    "DE": "disposal_efficiency_pct",
    "CM": "adv_contested_marks",
    "MI5": "adv_marks_inside_50",
    "One.Percenters": "adv_one_percenters",
    "BO": "adv_bounces",
    "TOG": "adv_time_on_ground_pct",
    "CCL": "centre_clearances",
    "SCL": "stoppage_clearances",
    "SI": "score_involvements",
    "MG": "metres_gained",
    "TO": "turnovers",
    "ITC": "intercepts",  # footywire's generic "Intercepts" -- NOT confirmed identical to Champion
                          # Data's 2026 "intercept possessions" definition. See docs/2026_STATS_MIRROR.md.
    "T5": "tackles_inside_50",
    "GA": "adv_goal_assists",
    "AF": "afl_fantasy_points",
    "SC": "supercoach_points",
}


def load_team_mapping() -> dict:
    df = pd.read_csv(CONFIG_DIR / "team_mapping.csv")
    return dict(zip(df["source_name"], df["canonical_team_id"]))


def build() -> pd.DataFrame:
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    adv_raw = pyreadr.read_r(RAW_DIR / "player_stats.rda")["player_stats"]

    team_map = load_team_mapping()
    unmapped = (set(adv_raw["Team"]) | set(adv_raw["Opposition"])) - set(team_map.keys())
    if unmapped:
        raise ValueError(f"Unmapped team names in player_stats -- update config/team_mapping.csv: {unmapped}")

    adv = adv_raw.rename(columns=ADV_COLUMN_RENAME).copy()
    adv["team_id"] = adv["Team"].map(team_map)
    adv["opponent_id"] = adv["Opposition"].map(team_map)
    def normalise_surname(series: pd.Series) -> pd.Series:
        # Last whitespace-separated token, lowercased, with punctuation stripped. Needed because
        # afltables and footywire do not always spell hyphenated/apostrophe surnames identically
        # (e.g. "O'Halloran" vs "OHalloran") -- see docs/DATA_COVERAGE.md identity-resolution notes.
        return series.str.split().str[-1].str.lower().str.replace(r"[^a-z]", "", regex=True)

    adv["date"] = pd.to_datetime(adv["Date"]).dt.strftime("%Y-%m-%d")
    adv["surname_key"] = normalise_surname(adv["Player"])

    core = core.copy()
    core["date_str"] = pd.to_datetime(core["date"]).dt.strftime("%Y-%m-%d")
    core["surname_key"] = normalise_surname(core["player_name"])

    adv_cols = ["date", "team_id", "opponent_id", "surname_key"] + list(ADV_COLUMN_RENAME.values())
    merged = core.merge(
        adv[adv_cols],
        left_on=["date_str", "team_id", "opponent_id", "surname_key"],
        right_on=["date", "team_id", "opponent_id", "surname_key"],
        how="left",
        suffixes=("", "_adv"),
    )

    # de-dup safety: the join key can in principle collide for two team-mates sharing a surname
    # in the same match -- detect and log rather than silently keep duplicated rows.
    dup_mask = merged.duplicated(subset=["match_id", "player_id"], keep=False) & merged["effective_disposals"].notna()
    if dup_mask.any():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        merged.loc[dup_mask].to_csv(REPORTS_DIR / "advanced_join_surname_collisions.csv", index=False)
        n_before = len(merged)
        # keep first match only per (match_id, player_id) to avoid row-count inflation from a
        # many-to-one join match (e.g. two team-mates sharing a surname colliding onto one row)
        merged = merged.drop_duplicates(subset=["match_id", "player_id"], keep="first")
        print(f"WARNING: {dup_mask.sum()} rows involved in a surname-collision in the advanced-stats "
              f"join ({n_before - len(merged)} duplicate rows dropped) -- logged to "
              f"reports/advanced_join_surname_collisions.csv, kept first match only.")

    match_rate = merged["effective_disposals"].notna().mean()
    within_2010_2025 = merged[merged["season"].between(2010, 2025)]
    match_rate_in_scope = within_2010_2025["effective_disposals"].notna().mean()
    print(f"Advanced-stat join match rate (all seasons 1984-2025): {match_rate:.1%}")
    print(f"Advanced-stat join match rate (2010-2025, where footywire data exists): {match_rate_in_scope:.1%}")

    out_cols = [c for c in merged.columns if c not in ("date_adv", "date_str", "surname_key")]
    out = merged[out_cols]
    out.to_parquet(PROCESSED_DIR / "player_match_advanced_2010_2025.parquet", index=False)
    print(f"Wrote {len(out):,} rows -> data/processed/player_match_advanced_2010_2025.parquet")
    return out


if __name__ == "__main__":
    build()
