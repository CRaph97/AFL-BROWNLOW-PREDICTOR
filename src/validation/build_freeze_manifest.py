"""
Build data/deployment/2027_freeze_manifest.json: the single machine-readable
definition of the intended 2027 architecture. No model is trained here.
Run after rolefree_compare:  python -m src.validation.build_freeze_manifest
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.models.performance_ml.ranker import DEFAULT_XGB
from src.validation import registry
from src.validation.run_experiments import STRUCTURAL_FAMILIES, STATS_ONLY_FAMILIES, family_features

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "data" / "experiments" / "analysis"
OUT = ROOT / "data" / "deployment" / "2027_freeze_manifest.json"
SEED = 20270101


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _latest(reg: pd.DataFrame, name: str) -> dict:
    r = reg[reg["name"] == name].sort_values("timestamp").iloc[-1]
    return {"experiment_id": r["experiment_id"], "git_commit": r["git_commit"], "timestamp": r["timestamp"], "n_features": int(r["n_features"]), "status": r["status"]}


def build() -> dict:
    reg = registry.load()
    decision = json.loads((AN / "a_decision.json").read_text())
    a_rolefree = decision["decision"].startswith("A role-free")
    a_families = [f for f in STRUCTURAL_FAMILIES if not (a_rolefree and f in ("role", "role_interactions"))]
    a_name = "A_structural_pl_rolefree" if a_rolefree else "A_structural_pl"
    b_params = {**DEFAULT_XGB, "max_depth": 6, "min_child_weight": 20, "n_estimators": 400, "learning_rate": 0.05}
    weights = pd.read_csv(AN / "ensemble_weights.csv").sort_values("test_season").iloc[-1]
    feat_reg = ROOT / "data" / "features" / "feature_registry.json"; feat_man = json.loads((ROOT / "data" / "features" / "build_manifest.json").read_text())
    denylist = ROOT / "data" / "canonical" / "contaminated_artefacts.json"
    m = {
        "manifest_version": "1.0.0", "built_at": datetime.now(timezone.utc).isoformat(), "git_commit": registry.git_commit(),
        "status": "ARCHITECTURE FROZEN -- no 2027 model trained (2027 input rows do not exist yet)",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
                        "blas": "accelerate (numpy.show_config)", "thread_settings": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "xgboost_n_jobs": 2}},
        "seeds": {"global": SEED, "xgboost_random_state": SEED, "simulator": SEED, "bootstrap": SEED},
        "feature_schema": {"build_version": feat_man["build_version"], "registry_path": str(feat_reg.relative_to(ROOT)), "registry_sha256": _sha(feat_reg),
                           "n_registered_columns": feat_man["n_feature_columns"], "input_hashes": feat_man["input_hashes"], "builder": "src/features/point_in_time.py"},
        "denylist": {"path": str(denylist.relative_to(ROOT)), "sha256": _sha(denylist), "denied_feature_columns": json.loads(denylist.read_text())["denied_feature_columns"]},
        "models": {
            "A_structural": {"class": "src.models.structural.pl_model.StructuralPL", "families": a_families, "n_features": len(family_features(a_families)),
                             "hyperparameters": {"l2": 1.0, "optimizer": "L-BFGS-B analytic gradient (fast_pl)", "maxiter": 500, "imputation": "train median"},
                             "window": "expanding 2003-2026", "role_free_decision": decision["decision"], "role_free_reason": decision["reason"], "source_experiment": _latest(reg, a_name)},
            "B_performance_ml": {"class": "src.models.performance_ml.ranker.PerformanceRanker", "families": STRUCTURAL_FAMILIES, "n_features": len(family_features(STRUCTURAL_FAMILIES)),
                                 "hyperparameters": b_params, "temperature": "fit on last training season by PL likelihood, then refit on all training seasons",
                                 "window": "expanding 2003-2026", "source_experiment": _latest(reg, "B_performance_xgb_rank_tuned"), "tuning": "analysis/ranker_tuning_summary.csv (pilot inner holdouts 2013-2015)"},
            "C_stats_only": {"class": "src.models.stats_only.learned_pl.StatsOnlyPL", "families": STATS_ONLY_FAMILIES, "n_features": len(family_features(STATS_ONLY_FAMILIES)),
                             "hyperparameters": {"l2": 1.0}, "guard": "src.models.stats_only.learned_pl.assert_stats_only", "window": "expanding 2003-2026", "source_experiment": _latest(reg, "C_stats_only_pl"), "role": "benchmark and ensemble component"},
        },
        "ensemble": {"method": "log-linear pool of component P3 (utility = sum_m w_m log p3_m), Plackett-Luce marginalisation for P2/P1", "module": "src.models.ensemble.stacking",
                     "weights_fit": "minimise OOF per-match P3 log loss on walk-forward OOF predictions of seasons <= 2026 only (softmax-parametrised Nelder-Mead, multi-start)",
                     "components": ["A_structural", "B_performance_ml", "C_stats_only"], "weights_latest_walk_forward_row": {"test_season": int(weights["test_season"]), "A": float(weights["w_A_structural_pl"]), "B": float(weights["w_B_performance_xgb_rank_tuned"]), "C": float(weights["w_C_stats_only_pl"])},
                     "note": "final 2027 weights are refit on all OOF seasons 2012-2026 at freeze; expected close to the 2026 row", "evidence": "analysis/audit_same_period_*.csv, audit_ensemble_weight_proof.csv"},
        "calibration": {"method": "Platt: logistic regression on logit(P3), renormalised within match; fit on OOF seasons <= 2026", "estimator": "sklearn.linear_model.LogisticRegression(C=1e6)", "applies_to": "headline ensemble and B", "evidence": "analysis/audit_calibration_by_season.csv"},
        "headline": {"probability_model": "Platt-calibrated learned ensemble (A+B+C)", "primary_standalone": "B_performance_ml", "structural_challenger": "A_structural (3-vote pick model, reported alongside)", "stats_only_challenger": "C_stats_only (benchmark)", "wheelo": "external corroboration and evaluation only; never a training target or ensemble input"},
        "cutoff_rules": ["training labels: seasons 2003-2026; 2026 labels from data/actual (label_source recorded)", "no 2027 vote or any post-match information enters a feature; every prior_matches/prior_seasons feature is dated before the match",
                         "denied columns (denylist) may never be consumed by any 2027 pipeline", "role for 2027 rows = lagged proxy only", "no in-season refit; per-round runs are inference only", "frozen output directory immutable after Round 1"],
        "simulator_contract": {"module": "src.simulation.season_sim", "entry": "simulate_matches(preds, n_sims, seed)", "input_columns": ["match_id", "round", "player_id", "player_name", "team_id", "p3"], "n_sims_production": 100000, "seed": SEED,
                               "outputs": ["totals", "players", "votes_by_round"], "queries": ["summarise", "h2h", "exact_order", "team_leader", "clinch_round"], "validated_against": "reports/2026_simulation_summary.csv (validate_against_frozen)"},
        "source_experiments": {n: _latest(reg, n) for n in ("baseline_phase4_pl_legacy", "A_structural_pl", "A_structural_pl_rolefree", "B_performance_xgb_rank_tuned", "C_stats_only_pl", "C_stats_only_xgb_rank") if (reg["name"] == n).any()},
        "artefact_hashes": {str(p.relative_to(ROOT)): _sha(p) for p in [AN / "comparison_pooled.csv", AN / "ensemble_weights.csv", AN / "champion.csv", AN / "a_decision.json", ROOT / "data" / "experiments" / "registry.jsonl"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(m, indent=1, default=str))
    return m


if __name__ == "__main__":
    m = build(); print("wrote", OUT); print(json.dumps(m["headline"], indent=1)); print(m["models"]["A_structural"]["role_free_decision"])
