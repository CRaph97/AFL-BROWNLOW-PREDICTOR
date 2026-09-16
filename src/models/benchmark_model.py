"""
Phase 4, Model 0: the transparent benchmark every other model must beat.

A standard multinomial logistic regression predicting P(votes=0/1/2/3) per
player-match INDEPENDENTLY (it does not know that votes are allocated
within-match, unlike Model 1's Plackett-Luce formulation). Its raw
probabilities are then renormalised WITHIN EACH MATCH so that P(3) sums to 1,
P(2) sums to 1, and P(1) sums to 1 across players -- a simple, principled
post-hoc coherence fix, per the Phase 3/4 instruction ("If a modelling method
does not naturally satisfy this, implement a principled normalization").

This is deliberately the simplest model in the project: no interactions, no
regularisation search, no nonlinear terms -- exactly the "minimum standard"
role the brief assigns it.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class BenchmarkModel:
    def __init__(self, feature_names: list):
        self.feature_names = feature_names
        self.clf = LogisticRegression(max_iter=2000)

    def fit(self, df: pd.DataFrame, vote_col: str = "brownlow_votes") -> "BenchmarkModel":
        X = df[self.feature_names].to_numpy(dtype=float)
        y = df[vote_col].to_numpy(dtype=int)
        self.clf.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame, match_col: str = "match_id") -> pd.DataFrame:
        X = df[self.feature_names].to_numpy(dtype=float)
        raw_proba = self.clf.predict_proba(X)  # columns follow self.clf.classes_
        out = df.copy()
        for cls_idx, cls in enumerate(self.clf.classes_):
            out[f"raw_p{cls}"] = raw_proba[:, cls_idx]
        for c in (0, 1, 2, 3):
            if f"raw_p{c}" not in out.columns:
                out[f"raw_p{c}"] = 0.0

        # within-match renormalisation for p3/p2/p1 individually (each must sum to 1 across
        # players in the match); p0 is whatever remains
        for c in (3, 2, 1):
            match_sum = out.groupby(match_col)[f"raw_p{c}"].transform("sum").replace(0, np.nan)
            out[f"p{c}"] = out[f"raw_p{c}"] / match_sum
        out["p0"] = np.clip(1 - out["p3"] - out["p2"] - out["p1"], 0, None)
        out["expected_votes"] = 3 * out["p3"] + 2 * out["p2"] + 1 * out["p1"]
        return out
