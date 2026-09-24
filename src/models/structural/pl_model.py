"""
2027 candidate A -- STRUCTURAL Brownlow model.

A Plackett-Luce (exploded-logit) ordered 3-2-1 model, the validated Phase 4 /
Production architecture (src/models/plackett_luce.py), wrapped for the 2027
feature store: train-only median imputation so that a player's first game
(no lagged form) no longer drops a whole match from training or prediction
(the documented Phase 5 6-match gap), plus explicit feature-family provenance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.plackett_luce import PlackettLuceModel
from src.models.structural.fast_pl import fit_pl_fast


class StructuralPL:
    family = "structural_pl"

    def __init__(self, features: list[str], l2: float = 1.0):
        self.features = list(dict.fromkeys(features))
        from src.validation.denylist import assert_not_denied
        assert_not_denied(self.features, context=type(self).__name__)
        self.l2 = l2
        self._median: pd.Series | None = None
        self.model = PlackettLuceModel(feature_names=self.features)

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        if self._median is None:
            self._median = X.median(numeric_only=True).fillna(0.0)
        return df.assign(**{c: X[c].fillna(self._median[c]) for c in self.features})

    def fit(self, df: pd.DataFrame) -> "StructuralPL":
        self._median = None
        d = self._impute(df)
        fit_pl_fast(self.model, d, l2=self.l2)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self._impute(df)
        out = self.model.predict(d)
        return out

    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.beta, index=self.features)
