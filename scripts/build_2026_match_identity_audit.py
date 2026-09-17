"""
Phase 5 audit deliverable: full 207-match reconciliation of the 2026 home-and-away season
against the raw afltables fixture (season, round, date, teams, scores), post round-label fix.

"Official" round/date/home/away/scores here means: read directly from the raw afltables
source (data/raw/fitzroy_data/afldata.rda), with the round re-derived via the documented
Opening Round correction (src/data/round_normalization_2026.py) -- this is the authoritative
source for round numbers used throughout this project (the only other candidate, footywire's
player_stats.rda, does not carry a round column at all -- see build_2026_extension.py). Score
and team/date fields are taken directly from afldata.rda with no transformation, since Phase 2
already validated afltables score/date integrity match-for-match against live AFL Tables pages
(docs/TARGET_VALIDATION.md).

Writes reports/2026_match_identity_audit.csv, one row per one of the 207 real 2026
home-and-away matches.
"""
from pathlib import Path

import pandas as pd
import pyreadr

from src.data.build_core_dataset import load_team_mapping
from src.data.round_normalization_2026 import official_round

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "fitzroy_data"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

FINALS_ROUNDS_2026 = {"SF", "PF", "GF", "QF", "EF", "Wildcard Final"}
FLAGGED_EXAMPLES = {
    ("2026-05-09", "geelong", "collingwood"): "geelong vs collingwood",
    ("2026-05-14", "brisbane_lions", "geelong"): "brisbane vs geelong",
}


def build_official_fixture() -> pd.DataFrame:
    raw = pyreadr.read_r(RAW_DIR / "afldata.rda")["afldata"]
    df = raw[raw["Season"] == 2026].copy()
    is_final = df["Round"].astype(str).isin(FINALS_ROUNDS_2026)
    df = df[~is_final]

    fixture = df.drop_duplicates(subset=["Date", "Home.team", "Away.team"])[
        ["Date", "Round", "Home.team", "Away.team", "Home.score", "Away.score"]
    ].copy()
    fixture.columns = ["official_date", "source_round", "official_home_raw", "official_away_raw",
                        "official_home_score", "official_away_score"]
    fixture["official_date"] = fixture["official_date"].astype(str)
    fixture["official_round"] = fixture["source_round"].astype(str).map(official_round)

    team_map = load_team_mapping()
    fixture["official_home"] = fixture["official_home_raw"].map(team_map)
    fixture["official_away"] = fixture["official_away_raw"].map(team_map)
    fixture["canonical_match_id"] = (
        "2026_R" + fixture["official_round"].astype(str) + "_"
        + fixture["official_home"] + "_v_" + fixture["official_away"] + "_" + fixture["official_date"]
    )
    return fixture.sort_values(["official_round", "official_date"]).reset_index(drop=True)


def main():
    fixture = build_official_fixture()
    assert len(fixture) == 207, f"expected 207 official 2026 home-and-away matches, found {len(fixture)}"

    core = pd.read_parquet(PROCESSED_DIR / "model_core_2026.parquet")
    core_2026 = core[core["season"] == 2026]
    match_probs = pd.read_csv(REPORTS_DIR / "2026_match_probabilities.csv")
    predicted_votes = pd.read_csv(REPORTS_DIR / "2026_predicted_votes.csv")

    # model-side per-match round + reconstructed scores (home team's own team_score/opponent_score row)
    model_side = core_2026.merge(
        fixture[["canonical_match_id", "official_home"]], left_on="match_id", right_on="canonical_match_id", how="right"
    )
    home_rows = model_side[model_side["team_id"] == model_side["official_home"]]
    model_by_match = home_rows.groupby("match_id").agg(
        model_round=("round", "first"),
        model_home_score=("team_score", "first"),
        model_away_score=("opponent_score", "first"),
    ).reset_index()

    dashboard_round = match_probs.drop_duplicates(subset=["match_id"])[["match_id", "round"]].rename(
        columns={"round": "dashboard_round"}
    )
    has_prediction = predicted_votes.drop_duplicates(subset=["match_id"])[["match_id"]].assign(prediction_match=True)

    audit = fixture.merge(model_by_match, left_on="canonical_match_id", right_on="match_id", how="left")
    audit = audit.merge(dashboard_round, left_on="canonical_match_id", right_on="match_id", how="left", suffixes=("", "_dash"))
    audit = audit.merge(has_prediction, left_on="canonical_match_id", right_on="match_id", how="left", suffixes=("", "_pred"))
    audit["prediction_match"] = audit["prediction_match"].fillna(False)

    audit["identity_match"] = audit["model_round"].notna()  # match_id resolved in model_core_2026 at all
    audit["round_match"] = (audit["model_round"].astype("Int64").astype(str) == audit["official_round"].astype(str)) & \
                            (audit["dashboard_round"].astype("Int64").astype(str) == audit["official_round"].astype(str))
    audit["score_match"] = (audit["model_home_score"] == audit["official_home_score"]) & \
                            (audit["model_away_score"] == audit["official_away_score"])

    def note(r):
        key = (r["official_date"], r["official_home"], r["official_away"])
        base = FLAGGED_EXAMPLES.get(key, "")
        if base:
            base = f"user-flagged example ({base}); "
        if r["source_round"] != r["official_round"]:
            return base + f"source_round={r['source_round']} corrected to official_round={r['official_round']} (Opening Round offset)"
        return base + "source_round already equals official_round (Opening Round itself, or -- N/A here since round 1 always shifts)"

    audit["notes"] = audit.apply(note, axis=1)

    out = audit[[
        "official_round", "official_date", "official_home", "official_away",
        "official_home_score", "official_away_score", "canonical_match_id",
        "source_round", "model_round", "dashboard_round",
        "identity_match", "round_match", "score_match", "prediction_match", "notes",
    ]].sort_values(["official_round", "official_date"])

    out.to_csv(REPORTS_DIR / "2026_match_identity_audit.csv", index=False)

    n = len(out)
    n_id = int(out["identity_match"].sum())
    n_round = int(out["round_match"].sum())
    n_score = int(out["score_match"].sum())
    n_pred = int(out["prediction_match"].sum())
    print(f"reports/2026_match_identity_audit.csv: {n} rows")
    print(f"identity_match: {n_id}/{n}, round_match: {n_round}/{n}, score_match: {n_score}/{n}, prediction_match: {n_pred}/{n}")
    for key, label in FLAGGED_EXAMPLES.items():
        row = out[out["canonical_match_id"].str.contains(key[0]) & out["official_home"].eq(key[1]) & out["official_away"].eq(key[2])]
        print(f"\n{label}:")
        print(row.to_string(index=False))


if __name__ == "__main__":
    main()
