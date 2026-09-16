"""
Phase 5, sections 6/7/11/12/15: final 2026 match probabilities, deterministic
predicted votes, the leaderboard, contender round-by-round reports, match-level
explanations for material games, and the pre-delivery quality checks.

All figures here come from the FINAL_ENSEMBLE scenario (data/processed/
all_2026_scenarios_and_ensemble.parquet) for point estimates and probabilities,
and the Monte Carlo simulation (reports/2026_simulation_summary.csv) for
uncertainty intervals -- consistent with section 10's instruction that the
final production forecast is the documented ensemble, with Monte Carlo used
for the season-total distributional statistics.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

MATERIAL_P3_THRESHOLD = 0.30  # "meaningful polling performance" cutoff for match-level explanations
TOP_N_CONTENDERS = 20

DRIVER_STATS = [
    ("disposals", "disposals"), ("contested_possessions", "contested possessions"),
    ("clearances", "clearances"), ("tackles", "tackles"), ("goals", "goals"),
    ("inside_50s", "inside 50s"), ("contested_marks", "contested marks"), ("marks", "marks"),
]


def quality_checks(match_probs: pd.DataFrame, core_2026: pd.DataFrame) -> dict:
    checks = {}

    sums = match_probs.groupby("match_id")[["p3", "p2", "p1"]].sum()
    checks["max_abs_p3_sum_error"] = float((sums["p3"] - 1).abs().max())
    checks["max_abs_p2_sum_error"] = float((sums["p2"] - 1).abs().max())
    checks["max_abs_p1_sum_error"] = float((sums["p1"] - 1).abs().max())

    ev_sum = match_probs.groupby("match_id")["expected_votes"].sum()
    checks["max_abs_ev_sum_error_vs_6"] = float((ev_sum - 6).abs().max())

    n_matches_covered = match_probs["match_id"].nunique()
    n_matches_total = core_2026["match_id"].nunique()
    checks["n_2026_matches_total"] = int(n_matches_total)
    checks["n_2026_matches_covered_by_ensemble"] = int(n_matches_covered)
    checks["all_matches_covered"] = bool(n_matches_covered == n_matches_total)

    checks["n_duplicate_player_match_rows"] = int(
        match_probs.duplicated(subset=["match_id", "player_id"]).sum()
    )

    checks["expected_season_total_votes"] = float(6 * n_matches_covered)
    checks["actual_season_total_expected_votes"] = float(ev_sum.sum())

    checks["n_2026_core_rows_missing_key_stats"] = int(
        core_2026[core_2026["season"] == 2026][["disposals", "goals", "clearances"]].isna().any(axis=1).sum()
    )
    return checks


def deterministic_match_picks(match_probs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_id, g in match_probs.groupby("match_id"):
        pred_3 = g.loc[g["p3"].idxmax()]
        remaining = g[g["player_id"] != pred_3["player_id"]]
        pred_2 = remaining.loc[remaining["p2"].idxmax()]
        remaining2 = remaining[remaining["player_id"] != pred_2["player_id"]]
        pred_1 = remaining2.loc[remaining2["p1"].idxmax()]
        for votes, row in [(3, pred_3), (2, pred_2), (1, pred_1)]:
            rows.append({
                "match_id": match_id, "season": row["season"], "round": row["round"],
                "player_id": row["player_id"], "player_name": row["player_name"], "team_id": row["team_id"],
                "predicted_votes": votes, "p3": row["p3"], "p2": row["p2"], "p1": row["p1"],
                "expected_votes": row["expected_votes"],
            })
    out = pd.DataFrame(rows)
    check = out.groupby("match_id")["predicted_votes"].sum()
    assert (check == 6).all(), "deterministic picks do not sum to 6 in every match"
    return out


def match_explanation(row: pd.Series, match_df: pd.DataFrame) -> str:
    team_rows = match_df[match_df["team_id"] == row["team_id"]]
    drivers = []
    for col, label in DRIVER_STATS:
        z_col = f"{col}_match_z"
        if z_col in row.index and pd.notna(row[z_col]) and row[z_col] >= 1.5:
            drivers.append(f"{label}={int(row[col])} (match z={row[z_col]:.1f})")
    win_str = "winning team" if row.get("win_loss_draw") == "win" else ("losing team" if row.get("win_loss_draw") == "loss" else "draw")
    gap = row.get("disposals_gap_best_team", np.nan)
    teammate_note = ""
    if pd.notna(gap) and gap < -5:
        best_teammate = team_rows.loc[team_rows["disposals"].idxmax(), "player_name"] if len(team_rows) else "a teammate"
        teammate_note = f" Uncertainty: teammate {best_teammate} also had a big disposal count."
    drivers_str = "; ".join(drivers) if drivers else "no single dominant statistical category"
    return f"{win_str}; {drivers_str}.{teammate_note}"


def contender_round_by_round(top_players: pd.DataFrame, match_probs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid in top_players["player_id"]:
        p_rows = match_probs[match_probs["player_id"] == pid].sort_values("round", key=lambda s: s.astype(int))
        for _, r in p_rows.iterrows():
            most_likely = max([("3", r["p3"]), ("2", r["p2"]), ("1", r["p1"]), ("0", r["p0"])], key=lambda x: x[1])
            confidence = "HIGH" if most_likely[1] >= 0.5 else ("MEDIUM" if most_likely[1] >= 0.3 else "LOW")
            rows.append({
                "player_id": pid, "player_name": r["player_name"], "season": r["season"], "round": r["round"],
                "match_id": r["match_id"], "team_id": r["team_id"],
                "most_likely_votes": most_likely[0], "confidence": confidence,
                "expected_votes": r["expected_votes"], "p3": r["p3"], "p2": r["p2"], "p1": r["p1"], "p0": r["p0"],
            })
    return pd.DataFrame(rows)


def run():
    all_scenarios = pd.read_parquet(PROCESSED_DIR / "all_2026_scenarios_and_ensemble.parquet")
    ensemble = all_scenarios[all_scenarios["scenario"] == "FINAL_ENSEMBLE"].copy()
    core_2026 = pd.read_parquet(PROCESSED_DIR / "model_core_2026.parquet")
    core_2026 = core_2026[core_2026["season"] == 2026].copy()

    match_cols = ["match_id", "season", "round", "player_id", "player_name", "team_id",
                  "p3", "p2", "p1", "p0", "expected_votes"]
    match_probs = ensemble[match_cols].copy()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    match_probs.to_csv(REPORTS_DIR / "2026_match_probabilities.csv", index=False)
    print(f"Wrote reports/2026_match_probabilities.csv ({len(match_probs):,} rows)")

    picks = deterministic_match_picks(match_probs)
    picks.to_csv(REPORTS_DIR / "2026_predicted_votes.csv", index=False)
    print(f"Wrote reports/2026_predicted_votes.csv ({len(picks):,} rows, {picks['match_id'].nunique()} matches)")

    checks = quality_checks(match_probs, core_2026)
    (REPORTS_DIR / "2026_quality_checks.json").write_text(json.dumps(checks, indent=2))
    print("=== QUALITY CHECKS ===")
    print(json.dumps(checks, indent=2))

    scenario_cmp = pd.read_csv(REPORTS_DIR / "2026_scenario_comparison.csv", dtype={"player_id": str})
    sim_summary = pd.read_csv(REPORTS_DIR / "2026_simulation_summary.csv", dtype={"player_id": str})
    picks["player_id"] = picks["player_id"].astype(str)

    leaderboard = scenario_cmp.merge(
        sim_summary[["player_id", "sim_mean_votes", "sim_median_votes", "sim_p10", "sim_p90",
                     "sim_p2_5", "sim_p97_5"] + [c for c in sim_summary.columns if c.startswith("prob_")]],
        on="player_id", how="left",
    )
    n_3_games = picks[picks["predicted_votes"] == 3].groupby("player_id").size().rename("projected_3_vote_games")
    n_2_games = picks[picks["predicted_votes"] == 2].groupby("player_id").size().rename("projected_2_vote_games")
    n_1_games = picks[picks["predicted_votes"] == 1].groupby("player_id").size().rename("projected_1_vote_games")
    leaderboard = leaderboard.merge(n_3_games, on="player_id", how="left").merge(n_2_games, on="player_id", how="left").merge(n_1_games, on="player_id", how="left")
    for c in ["projected_3_vote_games", "projected_2_vote_games", "projected_1_vote_games"]:
        leaderboard[c] = leaderboard[c].fillna(0).astype(int)

    leaderboard = leaderboard.sort_values("FINAL_ENSEMBLE", ascending=False).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    leaderboard.to_csv(REPORTS_DIR / "2026_leaderboard.csv", index=False)
    print(f"Wrote reports/2026_leaderboard.csv ({len(leaderboard)} players)")

    top = leaderboard.head(TOP_N_CONTENDERS)
    rbr = contender_round_by_round(top, match_probs)
    rbr.to_csv(REPORTS_DIR / "2026_round_by_round_contenders.csv", index=False)
    print(f"Wrote reports/2026_round_by_round_contenders.csv ({len(rbr)} rows, {top.shape[0]} contenders)")

    material = match_probs[match_probs["p3"] >= MATERIAL_P3_THRESHOLD].copy()
    explanations = []
    for match_id, g_match in core_2026[core_2026["match_id"].isin(material["match_id"])].groupby("match_id"):
        mat_rows = material[material["match_id"] == match_id]
        for _, mr in mat_rows.iterrows():
            core_row = g_match[g_match["player_id"] == mr["player_id"]]
            if len(core_row) == 0:
                continue
            core_row = core_row.iloc[0]
            explanations.append({
                "match_id": match_id, "season": mr["season"], "round": mr["round"],
                "player_name": mr["player_name"], "team_id": mr["team_id"],
                "p3": mr["p3"], "p2": mr["p2"], "p1": mr["p1"], "p0": mr["p0"],
                "expected_votes": mr["expected_votes"],
                "most_likely": "3" if mr["p3"] > mr["p2"] and mr["p3"] > mr["p1"] else ("2" if mr["p2"] > mr["p1"] else "1"),
                "key_drivers": match_explanation(core_row, g_match),
            })
    explanations_df = pd.DataFrame(explanations).sort_values(["season", "round", "p3"], ascending=[True, True, False])
    explanations_df.to_csv(REPORTS_DIR / "2026_match_explanations.csv", index=False)
    print(f"Wrote reports/2026_match_explanations.csv ({len(explanations_df)} material games, p3>={MATERIAL_P3_THRESHOLD})")

    return checks, leaderboard


if __name__ == "__main__":
    run()
