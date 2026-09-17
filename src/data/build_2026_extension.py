"""
Phase 5, section 5: extend the validated CORE/ADVANCED pipelines to include the
2026 home-and-away season as an INFERENCE-ONLY extension.

Deliberately NOT a modification of build_core_dataset.py / build_advanced_dataset.py
(those remain the untouched, validated Phase 2 pipeline producing the 1984-2025 /
2010-2025 canonical files used for all Phase 4 training and backtesting). This
script reuses their column-mapping/canonicalisation logic and produces SEPARATE,
ADDITIVE output files:
    data/processed/player_match_core_1984_2026.parquet
    data/processed/player_match_advanced_2010_2026.parquet
2026 rows carry brownlow_votes = NaN (votes not yet revealed -- the count happens
after the home-and-away season, before finals; as of this run finals are already
underway) -- this is asserted, not just assumed. No target-integrity check is run
for 2026 (there is no target yet); instead we assert every 2026 match has the
expected number of afltables rows (i.e. a normal completed match).

2026 introduced a new finals structure with a "Wildcard Final" round in addition
to the usual EF/QF/SF -- excluded here exactly like every other finals round,
since Brownlow votes are never awarded for finals in any season.
"""
import json
from pathlib import Path

import pandas as pd
import pyreadr

from src.data.build_core_dataset import COLUMN_RENAME, KEEP_RAW_COLUMNS, load_team_mapping as load_team_mapping_core
from src.data.build_advanced_dataset import ADV_COLUMN_RENAME
from src.data.round_normalization_2026 import official_round

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "fitzroy_data"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

SEASON_2026 = 2026
FINALS_ROUNDS_2026 = {"SF", "PF", "GF", "QF", "EF", "Wildcard Final"}


def _normalise_surname(series: pd.Series) -> pd.Series:
    # footywire's player_stats.rda abbreviates the FIRST component of a hyphenated
    # compound surname to a single initial (e.g. "Wanganeen-Milera" -> "W-Milera",
    # "Davies-Uniacke" -> "D-Uniacke"), while afltables spells it in full. Confirmed
    # against all 12 affected 2026 players (Wanganeen-Milera, Horne-Francis,
    # Neal-Bullen, Ugle-Hagan, Byrne-Jones, Zerk-Thatcher, Coleman-Jones, El-Hawli,
    # Day-Wicks, Davies-Uniacke, Hall-Kahan, Duff-Tytler). Taking only the segment
    # after the last hyphen collapses both conventions to the same key ("milera")
    # without affecting any non-hyphenated surname (a no-op there).
    last_token = series.str.split().str[-1]
    last_hyphen_segment = last_token.str.split("-").str[-1]
    return last_hyphen_segment.str.lower().str.replace(r"[^a-z]", "", regex=True)


def build_core_2026() -> pd.DataFrame:
    raw = pyreadr.read_r(RAW_DIR / "afldata.rda")["afldata"]
    df = raw[KEEP_RAW_COLUMNS].rename(columns=COLUMN_RENAME).copy()
    df["brownlow_votes"] = pd.to_numeric(df["brownlow_votes"], errors="coerce")

    is_final = df["round"].astype(str).isin(FINALS_ROUNDS_2026)
    core = df[(df["season"] == SEASON_2026) & ~is_final].copy()

    # afltables' raw round label is off by one from the AFL's official round number for
    # every round after the split Opening Round -- see round_normalization_2026.py for the
    # full root-cause writeup (confirmed via the Phase 5 2026-integrity audit). Keep the raw
    # value for traceability, but `round` downstream must be the official number.
    core["round_raw_source"] = core["round"].astype(str)
    core["round"] = core["round"].astype(str).map(official_round).astype(str)

    assert core["brownlow_votes"].fillna(0).eq(0).all(), (
        "2026 rows unexpectedly carry non-zero Brownlow votes -- the count may have been revealed; "
        "re-check whether the assumption 'votes not yet public' still holds before proceeding."
    )
    core["brownlow_votes"] = pd.NA

    team_map = load_team_mapping_core()
    unmapped = (set(core["team_raw"]) | set(core["home_team_raw"]) | set(core["away_team_raw"])) - set(team_map.keys())
    if unmapped:
        raise ValueError(f"Unmapped 2026 team names -- update config/team_mapping.csv: {unmapped}")

    core["team_id"] = core["team_raw"].map(team_map)
    core["home_team_id"] = core["home_team_raw"].map(team_map)
    core["away_team_id"] = core["away_team_raw"].map(team_map)
    core["opponent_id"] = core.apply(
        lambda r: r["away_team_id"] if r["team_id"] == r["home_team_id"] else r["home_team_id"], axis=1
    )

    core = core.reset_index(drop=True)
    missing_id_mask = core["player_id_afltables"].isna()
    core["player_id"] = core["player_id_afltables"].astype("Int64").astype(str)
    n_missing_id = int(missing_id_mask.sum())
    if missing_id_mask.any():
        placeholder_ids = [f"NOID2026_{i}" for i in core.index[missing_id_mask]]
        core.loc[missing_id_mask, "player_id"] = placeholder_ids

    core["player_name"] = core["player_first_name"] + " " + core["player_surname"]

    core["team_score"] = core.apply(lambda r: r["home_score"] if r["team_id"] == r["home_team_id"] else r["away_score"], axis=1)
    core["opponent_score"] = core.apply(lambda r: r["away_score"] if r["team_id"] == r["home_team_id"] else r["home_score"], axis=1)
    core["margin"] = core["team_score"] - core["opponent_score"]
    core["absolute_margin"] = core["margin"].abs()
    core["win_loss_draw"] = core["margin"].apply(lambda m: "win" if m > 0 else ("loss" if m < 0 else "draw"))

    core["match_id"] = (
        core["season"].astype(str) + "_R" + core["round"].astype(str) + "_"
        + core["home_team_id"] + "_v_" + core["away_team_id"] + "_" + core["date"].astype(str)
    )

    final_columns = [
        "season", "round", "match_id", "date", "venue",
        "player_id", "player_name", "team_id", "opponent_id", "home_away",
        "team_score", "opponent_score", "margin", "win_loss_draw", "absolute_margin",
        "kicks", "marks", "handballs", "disposals", "goals", "behinds", "hitouts",
        "tackles", "rebound_50s", "inside_50s", "clearances", "clangers",
        "frees_for", "frees_against", "contested_possessions", "uncontested_possessions",
        "contested_marks", "marks_inside_50", "one_percenters", "bounces", "goal_assists",
        "time_on_ground_pct", "substitute_status", "jumper_number",
        "brownlow_votes",
    ]
    out = core[final_columns].copy()
    out["brownlow_votes"] = out["brownlow_votes"].astype("Int64")

    validation = {
        "n_rows": len(out),
        "n_matches": int(out["match_id"].nunique()),
        "n_missing_afltables_id": n_missing_id,
        "rounds": sorted(out["round"].unique().tolist(), key=lambda r: int(r)),
        "date_min": str(out["date"].min()),
        "date_max": str(out["date"].max()),
        "teams": sorted(out["team_id"].unique().tolist()),
        "n_teams": int(out["team_id"].nunique()),
    }
    return out, validation


def build_advanced_2026(core_2026: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    adv_raw = pyreadr.read_r(RAW_DIR / "player_stats.rda")["player_stats"]
    adv_raw = adv_raw[adv_raw["Season"] == SEASON_2026].copy()

    team_map = load_team_mapping_core()
    unmapped = (set(adv_raw["Team"]) | set(adv_raw["Opposition"])) - set(team_map.keys())
    if unmapped:
        raise ValueError(f"Unmapped 2026 team names in player_stats -- update config/team_mapping.csv: {unmapped}")

    adv = adv_raw.rename(columns=ADV_COLUMN_RENAME).copy()
    adv["team_id"] = adv["Team"].map(team_map)
    adv["opponent_id"] = adv["Opposition"].map(team_map)
    adv["date"] = pd.to_datetime(adv["Date"]).dt.strftime("%Y-%m-%d")
    adv["surname_key"] = _normalise_surname(adv["Player"])

    core = core_2026.copy()
    core["date_str"] = pd.to_datetime(core["date"]).dt.strftime("%Y-%m-%d")
    core["surname_key"] = _normalise_surname(core["player_name"])

    adv_cols = ["date", "team_id", "opponent_id", "surname_key"] + list(ADV_COLUMN_RENAME.values())
    merged = core.merge(
        adv[adv_cols],
        left_on=["date_str", "team_id", "opponent_id", "surname_key"],
        right_on=["date", "team_id", "opponent_id", "surname_key"],
        how="left",
        suffixes=("", "_adv"),
    )
    dup_mask = merged.duplicated(subset=["match_id", "player_id"], keep=False) & merged["effective_disposals"].notna()
    if dup_mask.any():
        merged = merged.drop_duplicates(subset=["match_id", "player_id"], keep="first")

    match_rate = merged["effective_disposals"].notna().mean()
    out_cols = [c for c in merged.columns if c not in ("date_adv", "date_str", "surname_key")]
    out = merged[out_cols]
    validation = {"advanced_join_match_rate_2026": float(match_rate)}
    return out, validation


def main():
    hist_core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    core_2026, val_core = build_core_2026()
    extended_core = pd.concat([hist_core, core_2026], ignore_index=True)
    extended_core.to_parquet(PROCESSED_DIR / "player_match_core_1984_2026.parquet", index=False)
    print(f"player_match_core_1984_2026.parquet: {len(extended_core):,} rows "
          f"({len(core_2026):,} new 2026 rows, {val_core['n_matches']} matches)")

    hist_adv = pd.read_parquet(PROCESSED_DIR / "player_match_advanced_2010_2025.parquet")
    adv_2026, val_adv = build_advanced_2026(core_2026)
    # align columns: advanced historical table has the same schema as core plus adv_* columns
    common_cols = [c for c in hist_adv.columns if c in adv_2026.columns]
    extended_adv = pd.concat([hist_adv[common_cols], adv_2026[common_cols]], ignore_index=True)
    extended_adv.to_parquet(PROCESSED_DIR / "player_match_advanced_2010_2026.parquet", index=False)
    print(f"player_match_advanced_2010_2026.parquet: {len(extended_adv):,} rows, "
          f"2026 footywire join match rate: {val_adv['advanced_join_match_rate_2026']:.1%}")

    validation = {**val_core, **val_adv}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "2026_data_extension_validation.json").write_text(json.dumps(validation, indent=2, default=str))
    print(json.dumps(validation, indent=2, default=str))


if __name__ == "__main__":
    main()
