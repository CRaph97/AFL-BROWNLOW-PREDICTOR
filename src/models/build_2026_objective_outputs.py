"""
Orchestration for the 2026 Objective Stats Model (Experimental). Completely
separate from the production 2026 pipeline (train_2026_scenarios.py,
build_2026_ensemble.py, build_2026_outputs.py, run_2026_montecarlo.py) -- does
not import from or modify any of those files. See objective_stats_model.py for
the scoring logic and docs/2026_OBJECTIVE_MODEL.md for the full methodology.

Run: python -m src.models.build_2026_objective_outputs
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.objective_stats_model import (
    CONTEXT_COLS,
    GROUP_WEIGHTS,
    UNAVAILABLE_INPUTS,
    assert_no_forbidden_inputs,
    build_match_probabilities,
    compute_objective_scores,
    load_2026_input_frame,
    top_drivers,
)

REPORTS = Path("reports")
DOCS = Path("docs")

GROUP_COLS = {
    "POSSESSION_QUALITY": "grp_possession_quality",
    "CONTEST": "grp_contest",
    "CLEARANCE": "grp_clearance",
    "SCORING": "grp_scoring",
    "SCORE_CREATION": "grp_score_creation",
    "TERRITORY": "grp_territory",
    "DEFENCE": "grp_defence",
    "PRESSURE": "grp_pressure",
    "RUCK": "grp_ruck",
    "TEAM_RESULT": "grp_team_result",
}


def deterministic_match_picks(scored: pd.DataFrame) -> pd.DataFrame:
    """Greedy argmax 3-2-1 allocation, guaranteed three distinct players per match.
    Same generic algorithm pattern used by the production pipeline's
    build_2026_outputs.py::deterministic_match_picks -- pure allocation logic with
    no fitted parameters, reimplemented locally to keep this model's code fully
    separate from the production orchestration module."""
    rows = []
    for match_id, g in scored.groupby("match_id", sort=False):
        pred_3 = g.loc[g["p3"].idxmax()]
        remaining = g[g["player_id"] != pred_3["player_id"]]
        pred_2 = remaining.loc[remaining["p2"].idxmax()]
        remaining2 = remaining[remaining["player_id"] != pred_2["player_id"]]
        pred_1 = remaining2.loc[remaining2["p1"].idxmax()]
        rows.append({"match_id": match_id, "player_id": pred_3["player_id"], "objective_pred_votes": 3})
        rows.append({"match_id": match_id, "player_id": pred_2["player_id"], "objective_pred_votes": 2})
        rows.append({"match_id": match_id, "player_id": pred_1["player_id"], "objective_pred_votes": 1})
    return pd.DataFrame(rows)


def run_sensitivity_analysis(df_input: pd.DataFrame, base_leaderboard: pd.DataFrame, top_n: int = 20) -> dict:
    """Perturbs each group weight by +/-25% (renormalising the other 9 groups to
    keep the 100-point budget fixed) and reports how stable the top-N leaderboard
    is. This is a check that no single arbitrary coefficient determines the
    result -- not a search for 'better' weights, and never touches historical
    Brownlow votes."""
    base_top = set(base_leaderboard.head(top_n)["player_id"])
    results = {}
    for group in GROUP_WEIGHTS:
        for direction, factor in [("minus25pct", 0.75), ("plus25pct", 1.25)]:
            perturbed = dict(GROUP_WEIGHTS)
            delta = perturbed[group] * (factor - 1.0)
            perturbed[group] = perturbed[group] * factor
            others = [g for g in perturbed if g != group]
            total_others = sum(GROUP_WEIGHTS[g] for g in others)
            for g in others:
                perturbed[g] = perturbed[g] - delta * (GROUP_WEIGHTS[g] / total_others)
            assert abs(sum(perturbed.values()) - 100.0) < 1e-6

            scored = compute_objective_scores(df_input, weights=perturbed)
            match_probs = build_match_probabilities(scored)
            season = (
                match_probs.groupby(["player_id", "player_name", "team_id"], as_index=False)
                .agg(objective_ev=("expected_votes", "sum"))
                .sort_values("objective_ev", ascending=False)
            )
            top_set = set(season.head(top_n)["player_id"])
            overlap = len(base_top & top_set)
            results[f"{group}_{direction}"] = {
                "top20_overlap_with_base": overlap,
                "top20_overlap_pct": round(overlap / top_n, 3),
            }
    return results


def main():
    df = load_2026_input_frame()

    feature_matrix_cols = list(df.columns)
    forbidden = assert_no_forbidden_inputs(
        [c for c in feature_matrix_cols if c not in ("player_id", "match_id")]
    )
    # brownlow_votes (the label itself) and the lagged reputation/role columns exist
    # in the *loaded* frame (it's the same CORE/ADVANCED table other code uses) but
    # must never reach compute_objective_scores' actual inputs. Verify that here.
    used_cols = set(CONTEXT_COLS) | {
        "disposals", "disposals_match_z", "contested_possessions", "contested_possessions_match_z",
        "contested_marks", "contested_marks_match_z", "clearances", "clearances_match_z",
        "goals", "behinds", "inside_50s", "inside_50s_match_z", "marks", "marks_match_z",
        "tackles", "tackles_match_z", "one_percenters", "rebound_50s", "hitouts",
        "frees_for", "frees_against", "clangers",
        "effective_disposals", "disposal_efficiency_pct", "score_involvements", "goal_assists",
        "metres_gained", "centre_clearances", "intercepts", "turnovers", "tackles_inside_50",
    }
    forbidden_in_used = assert_no_forbidden_inputs(used_cols)
    assert not forbidden_in_used, f"Forbidden historical/reputation/role columns in use: {forbidden_in_used}"

    scored = compute_objective_scores(df)
    match_probs = build_match_probabilities(scored)

    # --- match-level output ---------------------------------------------------
    match_out_cols = [
        "match_id", "round", "date", "team_id", "opponent_id", "player_id", "player_name",
        "team_score", "opponent_score", "margin",
    ] + list(GROUP_COLS.values()) + ["objective_score", "p3", "p2", "p1", "p0", "expected_votes"]
    match_scores = match_probs[match_out_cols].copy()
    match_scores["primary_drivers"] = match_probs.apply(lambda r: top_drivers(r, GROUP_COLS), axis=1)
    match_scores.to_csv(REPORTS / "2026_objective_match_scores.csv", index=False)

    # --- deterministic picks + season votes ------------------------------------
    picks = deterministic_match_picks(match_probs)
    votes_per_match = match_probs.merge(picks, on=["match_id", "player_id"], how="left")
    votes_per_match["objective_pred_votes"] = votes_per_match["objective_pred_votes"].fillna(0).astype(int)
    votes_per_match[
        ["match_id", "round", "date", "team_id", "opponent_id", "player_id", "player_name",
         "objective_score", "p3", "p2", "p1", "p0", "expected_votes", "objective_pred_votes"]
    ].to_csv(REPORTS / "2026_objective_votes.csv", index=False)

    season = (
        votes_per_match.groupby(["player_id", "player_name", "team_id"], as_index=False)
        .agg(
            objective_ev=("expected_votes", "sum"),
            objective_3_games=("objective_pred_votes", lambda s: int((s == 3).sum())),
            objective_2_games=("objective_pred_votes", lambda s: int((s == 2).sum())),
            objective_1_games=("objective_pred_votes", lambda s: int((s == 1).sum())),
            n_matches=("match_id", "nunique"),
        )
        .sort_values("objective_ev", ascending=False)
        .reset_index(drop=True)
    )
    season.insert(0, "rank", season.index + 1)
    season.to_csv(REPORTS / "2026_objective_leaderboard.csv", index=False)

    # --- comparison with production leaderboard ---------------------------------
    prod = pd.read_csv(REPORTS / "2026_leaderboard.csv")[["player_id", "player_name", "team_id", "rank", "FINAL_ENSEMBLE"]]
    prod = prod.rename(columns={"rank": "Production Rank", "FINAL_ENSEMBLE": "Production EV"})
    prod["player_id"] = prod["player_id"].astype(str)
    comp = season.rename(columns={"rank": "Objective Rank", "objective_ev": "Objective EV"})
    comp["player_id"] = comp["player_id"].astype(str)
    comp = comp.merge(prod, on=["player_id", "player_name", "team_id"], how="outer")
    comp["Difference"] = comp["Objective EV"] - comp["Production EV"]
    comp["Rank Difference"] = comp["Production Rank"] - comp["Objective Rank"]
    comp = comp.rename(columns={"player_name": "Player", "team_id": "Team"})
    comp = comp[["Player", "Team", "Production EV", "Objective EV", "Difference",
                 "Production Rank", "Objective Rank", "Rank Difference"]]
    comp = comp.sort_values("Objective EV", ascending=False)
    comp.to_csv(REPORTS / "2026_objective_vs_production.csv", index=False)

    # --- QC -----------------------------------------------------------------
    n_matches = match_probs["match_id"].nunique()
    p_sum_err = 0.0
    for col in ["p3", "p2", "p1"]:
        s = match_probs.groupby("match_id")[col].sum()
        p_sum_err = max(p_sum_err, (s - 1.0).abs().max())
    ev_total = match_probs.drop_duplicates(["match_id", "player_id"])["expected_votes"].sum()
    dup = match_probs.duplicated(["match_id", "player_id"]).sum()
    pick_counts = picks.groupby("match_id")["player_id"].nunique()
    all_distinct = (pick_counts == 3).all()

    qc = {
        "n_2026_matches_total": int(n_matches),
        "n_matches_expected": 207,
        "all_207_matches_present": bool(n_matches == 207),
        "max_abs_p3_p2_p1_sum_error": float(p_sum_err),
        "season_total_expected_votes": float(ev_total),
        "expected_season_total": float(6 * n_matches),
        "season_total_matches_expected": bool(abs(ev_total - 6 * n_matches) < 1e-6),
        "n_duplicate_player_match_rows": int(dup),
        "deterministic_picks_always_3_distinct_players": bool(all_distinct),
        "forbidden_columns_in_scoring_inputs": forbidden_in_used,
        "unavailable_candidate_inputs": UNAVAILABLE_INPUTS,
        "group_weights": GROUP_WEIGHTS,
    }
    with open(REPORTS / "2026_objective_quality_checks.json", "w") as f:
        json.dump(qc, f, indent=2)

    # --- sensitivity analysis -------------------------------------------------
    sensitivity = run_sensitivity_analysis(df, season)
    with open(REPORTS / "2026_objective_sensitivity.json", "w") as f:
        json.dump(sensitivity, f, indent=2)

    print("QC:", json.dumps(qc, indent=2))
    print("\nTop 20:\n", season.head(20).to_string(index=False))
    return qc, sensitivity, season, comp


if __name__ == "__main__":
    main()
