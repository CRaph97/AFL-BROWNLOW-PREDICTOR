"""
2027 candidate B -- PERFORMANCE ML model.

Nonlinear learning-to-rank on the within-match structure: an XGBoost
LambdaMART ranker (objective rank:ndcg) with each match as one query group
and relevance = actual votes (3/2/1/0). Its raw score is treated as a
Plackett-Luce utility and turned into coherent P3/P2/P1/P0 by the exact
marginalisation used everywhere else in the project, after a single
TEMPERATURE parameter is fitted on an inner chronological holdout (the last
training season) by maximising the Plackett-Luce likelihood -- the ranker
never sees that season while its temperature is chosen, and the test season
is never touched. The ranker is then refit on all training seasons with the
same hyperparameters.

Why a temperature: Phase 4 found a GBM regression score used directly as a
utility was badly calibrated (ECE 0.032). A ranking score has no natural
scale; one learned scalar fixes the scale without touching the ranking.

A HistGradientBoosting regression variant with the same temperature step is
kept as a dependency-free fallback (PerformanceHGB).

XGBoost dependency: already installed in this environment (2.x); the earlier
Phase 4 libomp issue no longer reproduces. lightgbm is not installed and is
not needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.plackett_luce import _attach_pl_probabilities

DEFAULT_XGB = {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
               "min_child_weight": 5, "reg_lambda": 5.0, "objective": "rank:ndcg", "lambdarank_pair_method": "topk",
               "lambdarank_num_pair_per_sample": 8, "random_state": 20270101, "n_jobs": 2, "tree_method": "hist"}


def _pl_nll(u: np.ndarray, votes: np.ndarray, match_codes: np.ndarray) -> float:
    """Exploded-logit negative log likelihood of the observed 3-2-1 given utilities."""
    df = pd.DataFrame({"u": u, "v": votes, "m": match_codes})
    tot = 0.0
    for _, g in df.groupby("m", sort=False):
        uu = g["u"].to_numpy(); vv = g["v"].to_numpy()
        if (vv == 3).sum() != 1 or (vv == 2).sum() != 1 or (vv == 1).sum() != 1:
            continue
        mx = uu.max(); w = np.exp(uu - mx)
        s = w.sum(); w3 = w[vv == 3][0]; w2 = w[vv == 2][0]; w1 = w[vv == 1][0]
        tot -= np.log(w3 / s) + np.log(w2 / max(s - w3, 1e-300)) + np.log(w1 / max(s - w3 - w2, 1e-300))
    return tot


def fit_temperature(score: np.ndarray, votes: np.ndarray, match_codes: np.ndarray) -> float:
    s = (score - score.mean()) / (score.std() + 1e-9)
    res = minimize_scalar(lambda t: _pl_nll(t * s, votes, match_codes), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


class _TemperedUtilityModel:
    family = "performance_ml"

    def __init__(self, features: list[str], **params):
        self.features = list(dict.fromkeys(features))
        from src.validation.denylist import assert_not_denied
        assert_not_denied(self.features, context=type(self).__name__)
        self.params = params
        self.tau = 1.0
        self._mu, self._sd = 0.0, 1.0

    def _new(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _fit_core(self, df: pd.DataFrame):  # pragma: no cover - abstract
        raise NotImplementedError

    def _score(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, df: pd.DataFrame, season_col: str = "season") -> "_TemperedUtilityModel":
        seasons = sorted(df[season_col].unique())
        if len(seasons) >= 3:
            inner_train = df[df[season_col] < seasons[-1]]
            hold = df[df[season_col] == seasons[-1]]
            self._fit_core(inner_train)
            s = self._score(hold)
            self._mu, self._sd = float(s.mean()), float(s.std() + 1e-9)
            codes, _ = pd.factorize(hold["match_id"].to_numpy())
            self.tau = fit_temperature(s, hold["brownlow_votes"].to_numpy(), codes)
        self._fit_core(df)
        s_all = self._score(df)
        self._mu, self._sd = float(s_all.mean()), float(s_all.std() + 1e-9)
        return self

    def utility(self, df: pd.DataFrame) -> np.ndarray:
        return self.tau * (self._score(df) - self._mu) / self._sd

    def predict(self, df: pd.DataFrame, match_col: str = "match_id") -> pd.DataFrame:
        out = df.copy()
        out["_u"] = self.utility(df)
        parts = [_attach_pl_probabilities(g.drop(columns=["_u"]), g["_u"].to_numpy()) for _, g in out.groupby(match_col, sort=False)]
        return pd.concat(parts, axis=0)


class PerformanceRanker(_TemperedUtilityModel):
    family = "performance_ml_xgb_rank"

    def _fit_core(self, df: pd.DataFrame):
        import xgboost as xgb
        p = {**DEFAULT_XGB, **self.params}
        d = df.sort_values("match_id", kind="stable")
        X = d[self.features].to_numpy(dtype=float)
        y = d["brownlow_votes"].to_numpy(dtype=float)
        qid, _ = pd.factorize(d["match_id"].to_numpy())
        self.model = xgb.XGBRanker(**p)
        self.model.fit(X, y, qid=qid)

    def _score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].to_numpy(dtype=float))

    def feature_importance(self) -> pd.Series:
        booster = self.model.get_booster()
        gain = booster.get_score(importance_type="total_gain")
        return pd.Series({self.features[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)


class PerformanceHGB(_TemperedUtilityModel):
    family = "performance_ml_hgb"

    def _fit_core(self, df: pd.DataFrame):
        p = {"max_depth": 4, "max_iter": 300, "learning_rate": 0.05, "l2_regularization": 1.0, "random_state": 20270101, **self.params}
        self.model = HistGradientBoostingRegressor(**p)
        self.model.fit(df[self.features].to_numpy(dtype=float), df["brownlow_votes"].to_numpy(dtype=float))

    def _score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.features].to_numpy(dtype=float))
