"""
Leakage-safe hyperparameter selection for the Performance ML ranker (B).
Small fixed grid, scored by Plackett-Luce NLL on the INNER holdout (last
training season) of three pilot folds (test seasons 2014-2016), so no test
season -- and no recent season -- is used to choose parameters. The chosen
setting is written to data/experiments/analysis/ranker_tuning.csv and applied
to the candidate run by hand (recorded in the registry hyperparameters).
Run: python -m src.validation.tune_ranker
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.performance_ml.ranker import PerformanceRanker, _pl_nll, fit_temperature
from src.validation.run_experiments import ML_FAMILIES, family_features, _load, FEAT
from src.validation.walk_forward import make_folds

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "experiments" / "analysis" / "ranker_tuning.csv"
GRID = {"max_depth": [3, 4, 6], "n_estimators": [200, 400], "learning_rate": [0.05], "min_child_weight": [5, 20]}
PILOT_TEST_SEASONS = [2014, 2015, 2016]


def run():
    feats = family_features(ML_FAMILIES)
    df = _load(feats)
    seasons = sorted(df["season"].unique())
    rows = []
    for fold in make_folds(seasons, PILOT_TEST_SEASONS, window=None):
        inner_tr = df[df["season"].isin(fold.inner_train)]; hold = df[df["season"] == fold.inner_holdout]
        codes, _ = pd.factorize(hold["match_id"].to_numpy())
        for vals in itertools.product(*GRID.values()):
            p = dict(zip(GRID, vals)); t0 = time.time()
            m = PerformanceRanker(feats, **p); m._fit_core(inner_tr)
            s = m._score(hold); tau = fit_temperature(s, hold["brownlow_votes"].to_numpy(), codes)
            u = tau * (s - s.mean()) / (s.std() + 1e-9)
            nll = _pl_nll(u, hold["brownlow_votes"].to_numpy(), codes) / codes.max()
            top = hold.assign(u=u).groupby("match_id").apply(lambda g: g.loc[g["u"].idxmax(), "brownlow_votes"] == 3, include_groups=False).mean()
            rows.append({"pilot_test_season": fold.test_season, "inner_holdout": fold.inner_holdout, **p, "holdout_pl_nll_per_match": nll, "holdout_correct_3": float(top), "tau": tau, "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    agg = res.groupby(list(GRID)).agg(mean_nll=("holdout_pl_nll_per_match", "mean"), mean_correct_3=("holdout_correct_3", "mean"), n=("pilot_test_season", "size")).reset_index().sort_values("mean_nll")
    agg["selected"] = False; agg.iloc[0, agg.columns.get_loc("selected")] = True
    res.to_csv(OUT, index=False); agg.to_csv(OUT.with_name("ranker_tuning_summary.csv"), index=False)
    print(agg.to_string()); return agg


if __name__ == "__main__":
    run()
