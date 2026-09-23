"""
2027 R&D experiment driver. Walk-forward fits over the point-in-time feature
store, OOF predictions to data/experiments/oof/<name>.parquet, per-season
metrics to data/experiments/metrics/<name>.csv and a registry record.

Suites:
  reproduce  -- push the FROZEN 2026 Production / Objective / Wheelo match
                probabilities through the new metric code and check they
                reproduce the frozen 2026 evaluation (metric-code validation).
  baseline   -- Phase-4/Production Plackett-Luce configuration (legacy FULL
                CORE features, recent-8 window) re-run walk-forward 2012-2026.
  candidates -- Structural (A), Performance ML (B), Stats-only (C) on the 2027
                feature families, expanding + recent-8 windows, 2012-2026.
  ablation   -- leave-one-family-out for the Structural model, 2017-2026.
Usage: python -m src.validation.run_experiments --suite candidates [--workers 4]
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import feature_sets as fs
from src.validation import metrics as M
from src.validation import registry
from src.validation.walk_forward import make_folds, assert_no_overlap

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "features" / "player_match_features.parquet"
REG = json.loads((ROOT / "data" / "features" / "feature_registry.json").read_text())
EXP = ROOT / "data" / "experiments"
OOF_DIR, MET_DIR = EXP / "oof", EXP / "metrics"

TEST_SEASONS = list(range(2012, 2027))
ABLATION_SEASONS = list(range(2017, 2027))
KEEP = ["season", "match_id", "player_id", "player_name", "team_id", "brownlow_votes", "p3", "p2", "p1", "p0", "expected_votes"]

SAME_MATCH_FAMILIES = ["raw", "match_relative", "match_relative_ext", "team_relative", "context", "teammate", "dominance", "nonlinear"]
STRUCTURAL_FAMILIES = SAME_MATCH_FAMILIES + ["role", "role_interactions", "lagged_form", "baseline_relative", "team_strength", "reputation_pit"]
ML_FAMILIES = STRUCTURAL_FAMILIES
STATS_ONLY_FAMILIES = SAME_MATCH_FAMILIES
LEGACY_FULL = (fs.FAMILIES["raw"] + fs.FAMILIES["match_relative"] + fs.FAMILIES["context"] + fs.FAMILIES["teammate"]
               + fs.FAMILIES["role"] + fs.FAMILIES["nonlinear"] + fs.FAMILIES["lagged_form"] + fs.FAMILIES["win_margin_interaction"])


def family_features(families: list[str]) -> list[str]:
    cols = []
    for f in families:
        cols += REG[f]["columns"]
    return list(dict.fromkeys(cols))


def build_model(kind: str, features: list[str], params: dict | None = None):
    params = params or {}
    if kind == "structural_pl":
        from src.models.structural.pl_model import StructuralPL
        return StructuralPL(features, **params)
    if kind == "stats_only_pl":
        from src.models.stats_only.learned_pl import StatsOnlyPL
        return StatsOnlyPL(features, **params)
    if kind == "xgb_rank":
        from src.models.performance_ml.ranker import PerformanceRanker
        return PerformanceRanker(features, **params)
    if kind == "hgb":
        from src.models.performance_ml.ranker import PerformanceHGB
        return PerformanceHGB(features, **params)
    if kind == "legacy_pl":
        # Phase 4 configuration (legacy features + dropna) fitted with the exact analytic
        # gradient (identical objective and coefficients, see tests/test_2027_rd.py).
        from src.models.plackett_luce import PlackettLuceModel
        from src.models.structural.fast_pl import fit_pl_fast

        class _LegacyPL:
            family = "legacy_pl"
            def __init__(self, feats):
                self.model = PlackettLuceModel(feature_names=feats); self.features = feats
            def fit(self, df):
                fit_pl_fast(self.model, df); return self
            def predict(self, df):
                return self.model.predict(df)
            def coefficients(self):
                return pd.Series(self.model.beta, index=self.features)
        return _LegacyPL(features)
    raise ValueError(kind)


def _load(features: list[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(["season", "match_id", "player_id", "player_name", "team_id", "role", "brownlow_votes", "date"] + features))
    return pd.read_parquet(FEAT, columns=cols)


def run_fold(args):
    kind, features, params, fold, legacy_dropna = args
    df = _load(features)
    assert_no_overlap(fold)
    tr = df[df["season"].isin(fold.train_seasons)]
    te = df[df["season"] == fold.test_season]
    if legacy_dropna:
        tr = tr.dropna(subset=features); te = te.dropna(subset=features)
    t0 = time.time()
    model = build_model(kind, features, params)
    model.fit(tr)
    pred = model.predict(te)
    pred = pred[[c for c in KEEP if c in pred.columns]].copy()
    pred["test_season"] = fold.test_season; pred["window"] = fold.window
    mm = M.match_metrics(pred, ci=True)
    sm = M.season_metrics(pred)["by_season"][0]
    row = {"test_season": fold.test_season, "window": fold.window, "n_train_seasons": len(fold.train_seasons),
           "fit_seconds": round(time.time() - t0, 1), **mm, **{k: v for k, v in sm.items() if k != "season"}}
    extra = {}
    if hasattr(model, "coefficients"):
        extra["coefficients"] = model.coefficients().to_dict()
    if hasattr(model, "feature_importance"):
        try:
            extra["importance"] = model.feature_importance().head(40).to_dict()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(model, "tau"):
        row["tau"] = float(model.tau)
    return row, pred, extra


def run_experiment(name: str, kind: str, families: list[str], features: list[str] | None = None, params: dict | None = None,
                   windows: dict | None = None, test_seasons: list[int] | None = None, workers: int = 4, notes: str = "",
                   status: str = "candidate", legacy_dropna: bool = False, write_oof: bool = True) -> pd.DataFrame:
    features = features or family_features(families)
    windows = windows or {"expanding": None, "recent8": 8}
    test_seasons = test_seasons or TEST_SEASONS
    seasons = sorted(pd.read_parquet(FEAT, columns=["season"])["season"].unique())
    seasons = [s for s in seasons if s <= 2026]
    jobs = []
    for wname, w in windows.items():
        for fold in make_folds(seasons, test_seasons, window=w):
            jobs.append((kind, features, params or {}, fold, legacy_dropna))
    print(f"[{name}] {len(jobs)} folds, {len(features)} features, kind={kind}", flush=True)
    t0 = time.time()
    rows, preds, extras = [], [], {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for row, pred, extra in ex.map(run_fold, jobs):
                rows.append(row); preds.append(pred); extras[f"{row['window']}_{row['test_season']}"] = extra
                print(f"  [{name}] {row['window']} {row['test_season']}: correct_3={row['correct_3']:.3f} exact={row['exact_321']:.3f} ll={row['log_loss_p3']:.3f} mae={row['season_mae']:.3f} ({row['fit_seconds']}s)", flush=True)
    else:
        for job in jobs:
            row, pred, extra = run_fold(job); rows.append(row); preds.append(pred); extras[f"{row['window']}_{row['test_season']}"] = extra
            print(f"  [{name}] {row['window']} {row['test_season']}: correct_3={row['correct_3']:.3f} exact={row['exact_321']:.3f} ll={row['log_loss_p3']:.3f} ({row['fit_seconds']}s)", flush=True)
    met = pd.DataFrame(rows).sort_values(["window", "test_season"])
    MET_DIR.mkdir(parents=True, exist_ok=True); OOF_DIR.mkdir(parents=True, exist_ok=True)
    met.to_csv(MET_DIR / f"{name}.csv", index=False)
    if write_oof:
        pd.concat(preds, ignore_index=True).to_parquet(OOF_DIR / f"{name}.parquet", index=False)
    (MET_DIR / f"{name}_extras.json").write_text(json.dumps(extras, default=str))
    agg = {}
    for wname in windows:
        sub = met[met["window"] == wname]
        agg[wname] = {**M.pooled(sub, ["correct_3", "a3_in_top3", "exact_321", "log_loss_p3", "brier_p3", "ece_p3", "season_mae", "spearman", "top10_hit"]),
                      "recent5": M.pooled(sub[sub["test_season"] >= 2022], ["correct_3", "exact_321", "log_loss_p3", "season_mae", "spearman"]),
                      "n_seasons": int(len(sub))}
    registry.record(name, kind, features, families, [int(s) for s in seasons if s < min(test_seasons)], test_seasons, params or {},
                    met.to_dict("records"), agg, notes=notes, status=status, window="+".join(windows))
    print(f"[{name}] done in {time.time() - t0:.0f}s; expanding pooled: " + json.dumps({k: round(v, 4) for k, v in agg[list(windows)[0]].items() if isinstance(v, float)}), flush=True)
    return met


# ---------------------------------------------------------------- suites
def suite_reproduce() -> dict:
    """Validate metric code against the frozen 2026 evaluation."""
    out = {}
    act = pd.read_csv(ROOT / "data" / "actual" / "2026_brownlow_match_votes.csv", dtype={"player_id": "string"})
    av = act.set_index(["match_id", "player_id"])["actual_brownlow_votes"]
    def label(df):
        df = df.copy(); df["player_id"] = df["player_id"].astype(str)
        df["brownlow_votes"] = [av.get((m, p), 0) for m, p in zip(df["match_id"], df["player_id"])]
        df["season"] = 2026; return df
    prod = label(pd.read_csv(ROOT / "reports" / "2026_match_probabilities.csv"))
    obj = label(pd.read_csv(ROOT / "reports" / "2026_objective_votes.csv"))
    for nm, df in (("production_2026_frozen", prod), ("objective_2026_frozen", obj)):
        mm = M.match_metrics(df); out[nm] = {k: mm[k] for k in ("n_matches", "correct_3", "a3_in_top3", "exact_321", "log_loss_p3", "brier_p3", "ece_p3")}
    frozen = pd.read_csv(ROOT / "data" / "evaluation" / "2026" / "match_scorecard.csv").set_index("model")
    out["frozen_2026_evaluation"] = {m: {"hit_3_rate": float(frozen.loc[m, "hit_3_rate"]), "exact_321_rate": float(frozen.loc[m, "exact_321_rate"]), "mean_log_loss_p3": float(frozen.loc[m, "mean_log_loss_p3"])} for m in ("Production", "Objective", "Wheelo")}
    out["reproduced"] = bool(abs(out["production_2026_frozen"]["correct_3"] - out["frozen_2026_evaluation"]["Production"]["hit_3_rate"]) < 0.01
                             and abs(out["objective_2026_frozen"]["correct_3"] - out["frozen_2026_evaluation"]["Objective"]["hit_3_rate"]) < 0.01)
    (EXP / "reproduce_2026_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def suite_baseline(workers: int):
    run_experiment("baseline_phase4_pl_legacy", "legacy_pl", ["legacy_full"], features=LEGACY_FULL, windows={"recent8": 8, "expanding": None},
                   workers=workers, legacy_dropna=True, status="baseline",
                   notes="Phase 4 / Production Scenario A configuration: legacy FULL CORE features incl. same-season-unrevealed lagged-form dropna behaviour; reproduces docs/MODEL_BACKTEST.md for 2015-2025 and extends to 2026.")


def suite_candidates(workers: int):
    run_experiment("A_structural_pl", "structural_pl", STRUCTURAL_FAMILIES, workers=workers,
                   notes="Candidate A: Plackett-Luce on all point-in-time families (same-match + prior-match + prior-season reputation). Median imputation, no dropped matches.")
    run_experiment("C_stats_only_pl", "stats_only_pl", STATS_ONLY_FAMILIES, workers=workers,
                   notes="Candidate C: learned linear PL on strictly same-match features (Objective successor).")
    run_experiment("B_performance_xgb_rank", "xgb_rank", ML_FAMILIES, workers=max(1, workers // 2),
                   notes="Candidate B: XGBoost LambdaMART (rank:ndcg, match = query) + inner-holdout temperature -> PL marginalisation.")
    run_experiment("C_stats_only_xgb_rank", "xgb_rank", STATS_ONLY_FAMILIES, workers=max(1, workers // 2),
                   notes="Nonlinear stats-only variant of C (same strict feature set as C_stats_only_pl).")


def suite_ablation(workers: int):
    base = STRUCTURAL_FAMILIES
    run_experiment("ABL_full", "structural_pl", base, test_seasons=ABLATION_SEASONS, windows={"recent8": 8}, workers=workers, write_oof=False, status="ablation")
    for fam in base:
        keep = [f for f in base if f != fam]
        run_experiment(f"ABL_minus_{fam}", "structural_pl", keep, test_seasons=ABLATION_SEASONS, windows={"recent8": 8}, workers=workers, write_oof=False, status="ablation",
                       notes=f"leave-one-family-out: without {fam}")
    run_experiment("ABL_plus_reputation_legacy", "structural_pl", base + ["reputation_legacy"], test_seasons=ABLATION_SEASONS, windows={"recent8": 8}, workers=workers, write_oof=False, status="ablation",
                   notes="adds the SAME-SEASON-UNREVEALED legacy reputation columns: an upper bound that is NOT deployable for a live forecast")
    run_experiment("ABL_plus_advanced_2015plus", "structural_pl", base + ["advanced_stats"], test_seasons=ABLATION_SEASONS, windows={"recent8": 8}, workers=workers, write_oof=False, status="ablation",
                   notes="adds footywire advanced stats (available 2015+), recent-8 window so training never predates the family")
    run_experiment("ABL_plus_era", "structural_pl", base + ["era"], test_seasons=ABLATION_SEASONS, windows={"recent8": 8}, workers=workers, write_oof=False, status="ablation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--suite", required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"reproduce": lambda w: suite_reproduce(), "baseline": suite_baseline, "candidates": suite_candidates, "ablation": suite_ablation}[a.suite](a.workers)
    sys.exit(0)
