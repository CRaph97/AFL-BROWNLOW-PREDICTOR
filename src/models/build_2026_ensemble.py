"""
Phase 5, sections 3D/9/10: structural-break sensitivity variants (Scenario D)
and the final production ensemble.

=== DESIGN CORRECTION (2026-09-17) ===
An earlier version of this module blended Scenario A/B/C's WITHIN-MATCH
UNIT-VARIANCE-STANDARDISED utilities before re-deriving probabilities. That
was wrong: a Plackett-Luce utility's SCALE (unlike its additive per-match
shift) is not arbitrary -- it directly encodes how sharply that fitted model
discriminates between players (the softmax "temperature"). Forcing every
scenario's utility to unit variance before averaging discards each model's own
learned confidence and, because averaging several imperfectly-correlated
unit-variance signals mechanically shrinks the combined variance, produced
artificially FLATTENED, compressed probabilities (verified: it cut the
projected leader's expected votes roughly in half versus any individual
scenario, an obviously wrong direction for an ensemble of broadly agreeing
models).

**Fix: ensemble in PROBABILITY space, not utility space** -- a linear opinion
pool. Each scenario already produces a fully coherent p3/p2/p1 per match (sums
to 1 by construction, from `_attach_pl_probabilities`). Restrict every
scenario to the common player set for that match, RENORMALISE each scenario's
p3/p2/p1 to sum to 1 over that common set (the same post-hoc-renormalisation
principle already used for Model 0/2b in Phase 4), then take a weighted
average of PROBABILITIES across scenarios. A weighted average of vectors that
each already sum to 1, with weights summing to 1, sums to 1 automatically --
coherence is preserved exactly, with no rescaling artefact and no need to
re-invoke the Plackett-Luce marginalisation at all.

Scenario D (structural-break sensitivity): p_D = (1-alpha)*p_A + alpha*p_C,
per p3/p2/p1 independently. alpha is a documented judgement call (see
docs/2026_STRUCTURAL_BREAK.md), not a fitted parameter -- there are no 2026
votes to fit it against.

Final ensemble (section 10): p_E = 0.45*p_A + 0.20*p_B + 0.35*p_C. Weights are
a documented judgement call (Phase 4 OOS validation strength + directness of
relevance to the actual 2026 rule change), not a statistical estimate.

Each scenario's RAW utility (utility_raw, the model's own fitted linear
utility, NOT the unit-variance version) is carried through unchanged for
`run_2026_montecarlo.py`, which samples the ensemble as a genuine GENERATIVE
MIXTURE (for each match x simulation, draw which scenario's model applies
according to the ensemble weights, then Gumbel-max sample from THAT model's
own utility) -- the correct way to draw joint 3-2-1 samples from a probability
mixture of Plackett-Luce models, exactly reproducing the mixture's marginals
in the simulation limit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

ALPHA = {"D_low": 0.15, "D_medium": 0.35, "D_high": 0.60}
ENSEMBLE_WEIGHTS = {"A_historical": 0.45, "B_recent_era": 0.20, "C_stats_assisted": 0.35}
PROB_COLS = ["p3", "p2", "p1"]


def build_common_frame(scenario_preds: pd.DataFrame) -> pd.DataFrame:
    """Restrict to player-match rows present in ALL of A/B/C, then renormalise each scenario's
    own p3/p2/p1 to sum to 1 over exactly that common set (players missing from one scenario --
    e.g. ADVANCED's slightly stricter footywire-join completeness -- would otherwise leave that
    scenario's probability mass summing to < 1 within the reduced set)."""
    frames = {}
    for scen, key in [("A", "A_historical"), ("B", "B_recent_era"), ("C", "C_stats_assisted")]:
        cols = ["match_id", "player_id", "player_name", "team_id", "season", "round"] + PROB_COLS + ["utility_raw"]
        frames[scen] = scenario_preds[scenario_preds["scenario"] == key][cols].rename(
            columns={c: f"{c}_{scen}" for c in PROB_COLS + ["utility_raw"]}
        )

    merged = frames["A"].merge(
        frames["B"][["match_id", "player_id"] + [f"{c}_B" for c in PROB_COLS] + ["utility_raw_B"]],
        on=["match_id", "player_id"], how="inner",
    ).merge(
        frames["C"][["match_id", "player_id"] + [f"{c}_C" for c in PROB_COLS] + ["utility_raw_C"]],
        on=["match_id", "player_id"], how="inner",
    )

    for scen in ["A", "B", "C"]:
        for c in PROB_COLS:
            col = f"{c}_{scen}"
            g = merged.groupby("match_id")[col]
            merged[col] = merged[col] / g.transform("sum")

    n_matches_a = frames["A"]["match_id"].nunique()
    n_matches_common = merged["match_id"].nunique()
    print(f"Common A/B/C player-match frame: {len(merged):,} rows, {n_matches_common} matches "
          f"(of {n_matches_a} in Scenario A alone -- {n_matches_a - n_matches_common} matches lost "
          f"to incomplete ADVANCED/reputation coverage)")

    excluded_matches = set(frames["A"]["match_id"]) - set(merged["match_id"])
    if excluded_matches:
        frames["A"][frames["A"]["match_id"].isin(excluded_matches)].to_csv(
            REPORTS_DIR / "2026_excluded_matches.csv", index=False)
    return merged


def _blend_to_long(common: pd.DataFrame, weights: dict, scenario_name: str, prefix: str) -> pd.DataFrame:
    out = common[["match_id", "player_id", "player_name", "team_id", "season", "round"]].copy()
    for c in PROB_COLS:
        out[c] = sum(w * common[f"{c}_{scen}"] for scen, w in zip(["A", "B", "C"], weights))
    out["p0"] = (1 - out["p3"] - out["p2"] - out["p1"]).clip(lower=0)
    out["expected_votes"] = 3 * out["p3"] + 2 * out["p2"] + 1 * out["p1"]
    out["scenario"] = scenario_name
    return out


def run():
    scenario_preds = pd.read_parquet(PROCESSED_DIR / "scenario_predictions_2026.parquet")
    common = build_common_frame(scenario_preds)

    outputs = []
    for name, alpha in ALPHA.items():
        # weights over (A, B, C) with B's weight fixed at 0 -- D blends only A and C, per the brief
        outputs.append(_blend_to_long(common, [1 - alpha, 0.0, alpha], name, name))

    ensemble_weights_ordered = [ENSEMBLE_WEIGHTS["A_historical"], ENSEMBLE_WEIGHTS["B_recent_era"], ENSEMBLE_WEIGHTS["C_stats_assisted"]]
    outputs.append(_blend_to_long(common, ensemble_weights_ordered, "FINAL_ENSEMBLE", "FINAL_ENSEMBLE"))

    blended = pd.concat(outputs, ignore_index=True)

    # carry each scenario's raw utility (aligned to the common player set) for Monte Carlo's
    # generative-mixture sampling
    utility_frame = common[["match_id", "player_id"] + [f"utility_raw_{s}" for s in ["A", "B", "C"]]].copy()
    utility_frame.to_parquet(PROCESSED_DIR / "common_scenario_utilities_2026.parquet", index=False)

    all_scenarios = pd.concat([scenario_preds, blended], ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_scenarios.to_parquet(PROCESSED_DIR / "all_2026_scenarios_and_ensemble.parquet", index=False)
    print(f"Wrote {len(all_scenarios):,} total rows across all scenarios+ensemble -> "
          f"data/processed/all_2026_scenarios_and_ensemble.parquet")

    ev_check = blended[blended["scenario"] == "FINAL_ENSEMBLE"].groupby("player_id")["expected_votes"].sum()
    print(f"Sanity check -- FINAL_ENSEMBLE top-5 season EV: {ev_check.sort_values(ascending=False).head(5).to_dict()}")

    # --- section 9: scenario comparison table for leading contenders ---
    ev_by_scenario = all_scenarios.groupby(["player_id", "player_name", "team_id", "scenario"])["expected_votes"].sum().reset_index()
    pivot = ev_by_scenario.pivot_table(index=["player_id", "player_name", "team_id"], columns="scenario", values="expected_votes", fill_value=0.0).reset_index()
    core_scenarios = [c for c in ["A_historical", "B_recent_era", "C_stats_assisted", "D_low", "D_medium", "D_high"] if c in pivot.columns]
    if "C_stats_assisted" in pivot.columns and "A_historical" in pivot.columns:
        pivot["structural_break_sensitivity"] = (pivot[["A_historical", "C_stats_assisted"]].max(axis=1)
                                                  - pivot[["A_historical", "C_stats_assisted"]].min(axis=1))
    pivot["model_disagreement_range"] = pivot[core_scenarios].max(axis=1) - pivot[core_scenarios].min(axis=1)
    if "A_with_reputation" in pivot.columns:
        pivot["reputation_effect"] = pivot["A_with_reputation"] - pivot["A_historical"]
    pivot = pivot.sort_values("FINAL_ENSEMBLE", ascending=False)
    pivot.to_csv(REPORTS_DIR / "2026_scenario_comparison.csv", index=False)
    print(f"Wrote reports/2026_scenario_comparison.csv ({len(pivot)} players)")

    top_disagreement = pivot.sort_values("model_disagreement_range", ascending=False).head(30)
    top_disagreement.to_csv(REPORTS_DIR / "2026_model_disagreement.csv", index=False)
    print(f"Wrote reports/2026_model_disagreement.csv (top 30 by disagreement range)")

    return all_scenarios, pivot


if __name__ == "__main__":
    run()
