"""
Phase 4, Model 2: a gradient-boosted nonlinear model.

Implementation note: XGBoost was attempted first but its prebuilt macOS wheel
requires a newer libomp than this machine's Homebrew install provides
(a native-library ABI mismatch, not a Python-level issue) and fixing it was
judged not worth the time given `HistGradientBoostingClassifier` is a
genuinely comparable histogram-based gradient boosting implementation (the
same family of algorithm LightGBM popularised), already available via
scikit-learn (already a project dependency), with zero extra native
dependencies. This is a pragmatic substitution, not a compromise on the
model family the brief asked for -- documented here rather than silently
made.

Two formulations are implemented, per the brief's explicit instruction to
compare formulations rather than assume one:

  1. `GBMUtilityModel`: trains a GBM regressor to predict `brownlow_votes`
     directly (0-3, treated as an ordered numeric target), uses its raw
     predicted score as a Plackett-Luce utility (same marginalisation as
     Model 1, so probabilities are coherent within-match BY CONSTRUCTION),
     giving the model nonlinear/interaction power while keeping the
     structurally-correct ranking mechanism.
  2. `GBMMulticlassModel`: trains a GBM multiclass classifier directly on the
     4-class {0,1,2,3} target (the "naive" formulation, ignoring the
     within-match structure at training time), with the same post-hoc
     within-match renormalisation as Model 0. Kept as a comparison point to
     test whether the structural Plackett-Luce coupling actually matters
     once a flexible nonlinear model is used (Phase 4 question B/G2).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .plackett_luce import _attach_pl_probabilities


class GBMUtilityModel:
    def __init__(self, feature_names: list, **hgb_kwargs):
        self.feature_names = feature_names
        self.model = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=0, **hgb_kwargs)

    def fit(self, df: pd.DataFrame, vote_col: str = "brownlow_votes") -> "GBMUtilityModel":
        X = df[self.feature_names].to_numpy(dtype=float)
        y = df[vote_col].to_numpy(dtype=float)
        self.model.fit(X, y)
        return self

    def _utility(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].to_numpy(dtype=float)
        return self.model.predict(X)

    def predict(self, df: pd.DataFrame, match_col: str = "match_id") -> pd.DataFrame:
        out = df.copy()
        out["_utility"] = self._utility(df)
        parts = [
            _attach_pl_probabilities(g.drop(columns=["_utility"]), g["_utility"].to_numpy())
            for _, g in out.groupby(match_col, sort=False)
        ]
        return pd.concat(parts, axis=0)


class GBMMulticlassModel:
    def __init__(self, feature_names: list, **hgb_kwargs):
        self.feature_names = feature_names
        self.model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=0, **hgb_kwargs)

    def fit(self, df: pd.DataFrame, vote_col: str = "brownlow_votes") -> "GBMMulticlassModel":
        X = df[self.feature_names].to_numpy(dtype=float)
        y = df[vote_col].to_numpy(dtype=int)
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame, match_col: str = "match_id") -> pd.DataFrame:
        X = df[self.feature_names].to_numpy(dtype=float)
        raw_proba = self.model.predict_proba(X)
        out = df.copy()
        for cls_idx, cls in enumerate(self.model.classes_):
            out[f"raw_p{cls}"] = raw_proba[:, cls_idx]
        for c in (0, 1, 2, 3):
            if f"raw_p{c}" not in out.columns:
                out[f"raw_p{c}"] = 0.0
        for c in (3, 2, 1):
            match_sum = out.groupby(match_col)[f"raw_p{c}"].transform("sum").replace(0, np.nan)
            out[f"p{c}"] = out[f"raw_p{c}"] / match_sum
        out["p0"] = np.clip(1 - out["p3"] - out["p2"] - out["p1"], 0, None)
        out["expected_votes"] = 3 * out["p3"] + 2 * out["p2"] + 1 * out["p1"]
        return out
