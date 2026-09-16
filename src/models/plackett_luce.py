"""
Phase 4, Model 1: a rank-ordered (Plackett-Luce / "exploded logit") choice
model for within-match Brownlow vote allocation.

Why this model exists (per docs/MODELLING_PLAN.md §2 Model B): Brownlow votes
are a within-match ranking problem -- exactly one 3, one 2, one 1 per match,
drawn from the same pool of competing players. A Plackett-Luce model treats
this literally: each player i has a latent utility u_i = beta . x_i (a linear
combination of features), and the observed 3-2-1 allocation is treated as a
sequential "pick the best remaining player" process:
    P(3-vote = i)                 = exp(u_i) / sum_j exp(u_j)
    P(2-vote = k | 3-vote = i)    = exp(u_k) / sum_{j != i} exp(u_j)
    P(1-vote = m | 3-vote=i, 2-vote=k) = exp(u_m) / sum_{j != i,k} exp(u_j)
The training log-likelihood is the sum of the logs of the three observed
picks, summed over all matches. This is exactly the "exploded logit"
formulation named in the brief.

At prediction time we need MARGINAL probabilities P(3=i), P(2=i), P(1=i) for
every player i (not just the likelihood of the observed sequence), which
requires summing over all orderings of the other players for the 2nd and 3rd
pick. For a match of size n (~35-46 players) this is computed directly via
the closed-form marginalisation below (see `predict_match_probabilities`),
which is O(n^2) per match -- fast enough for every match in the dataset.

These probabilities are coherent BY CONSTRUCTION (sum of P(3) over all
players in a match == 1, same for P(2) and P(1)) -- no post-hoc
normalisation is needed, unlike Model 2's classifier-based formulation. This
directly satisfies the Phase 3/4 "within-match probability coherence"
requirement.

=== PHASE 4 CORRECTNESS BUG AND FIX (2026-09-17) ===
The first version of this model fit raw, unstandardised features directly.
CORE's raw stats stay in modest ranges (disposals 0-40, goals 0-8) and
happened not to trigger a failure, but the ADVANCED feature set adds
`metres_gained` (0-600+) alongside everything else. That scale mismatch, an
unconstrained L-BFGS optimiser, and a NAIVE (non-log-space) computation of
`match_sum - w3 - w2` combined to produce catastrophic cancellation: when one
player's utility dominates a match by many orders of magnitude, subtracting
its weight from the match total can go slightly NEGATIVE in floating point,
and `log()` of a negative number is NaN. This silently corrupted the
optimiser and produced garbage final probabilities (observed: ADVANCED
correct-3% around 10-20%, exact-3-2-1% at exactly 0% every season, log loss
~2.5 -- all symptoms of a broken fit, not a real "ADVANCED loses" finding).

Two independent fixes are applied, deliberately not just one:
1. **Standardisation** (fit on TRAIN ONLY, applied identically to test) --
   removes the raw-scale disparity that made the optimiser's job needlessly
   hard in the first place. Zero/near-zero-variance features get scale=1
   (never divide by zero). Binary 0/1 indicator columns (win/loss/draw,
   close-game/blowout flags, role dummies) are detected automatically and
   left UNSCALED so their coefficients keep the direct "effect of switching
   from 0 to 1" interpretation -- only centering+scaling continuous features.
2. **Log-space likelihood** -- `log(match_sum - w3)` and
   `log(match_sum - w3 - w2)` are now computed via `log1p(-exp(...))` in log
   space (a per-match log-sum-exp), which never forms the literal difference
   of two large floats and so cannot go negative from rounding error. Fitted
   coefficients and predicted probabilities are asserted finite (and
   probabilities asserted within [0,1]) -- this now FAILS LOUDLY with a clear
   error instead of silently returning garbage.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_LOG1P_CLIP = -1e-12  # keep the argument of exp() strictly negative before log1p(-exp(.))


def _attach_pl_probabilities(df_match: pd.DataFrame, u: np.ndarray) -> pd.DataFrame:
    """Shared exact Plackett-Luce marginalisation: given per-player utilities `u` for one
    match, returns df_match with p3/p2/p1/p0/expected_votes columns attached, coherent
    within the match by construction (sum of p3 == 1, sum of p2 == 1, sum of p1 == 1).
    Used by both PlackettLuceModel (linear utility) and GBMUtilityModel (tree-based
    utility) in gbm_model.py.

    Bounded to a single match (n ~ 35-46 players): the per-match max-shift keeps every
    w = exp(u) in (0, 1], so `total - w[i]` cannot suffer the catastrophic cancellation
    that motivated the log-space rewrite of the TRAINING likelihood below -- but we still
    assert finiteness/range here so a broken upstream utility (e.g. a corrupted model)
    fails loudly rather than silently shipping bad probabilities.
    """
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all():
        raise ValueError("Non-finite utility passed to _attach_pl_probabilities -- upstream model is broken.")
    u = u - u.max()
    w = np.exp(u)
    n = len(w)
    total = w.sum()

    p3 = w / total

    p2 = np.zeros(n)
    idx_all = np.arange(n)
    for i in range(n):
        others = idx_all != i
        denom = total - w[i]
        if denom > 0:
            p2[others] += p3[i] * (w[others] / denom)

    p1 = np.zeros(n)
    for i in range(n):
        denom1 = total - w[i]
        if denom1 <= 0:
            continue
        k_mask = idx_all != i
        k_idx = idx_all[k_mask]
        p_3i_2k = p3[i] * (w[k_idx] / denom1)
        denom2 = denom1 - w[k_idx]
        valid = denom2 > 0
        for k_local, k in enumerate(k_idx[valid]):
            others = k_mask & (idx_all != k)
            p1[others] += p_3i_2k[valid][k_local] * (w[others] / denom2[valid][k_local])

    for name, arr in (("p3", p3), ("p2", p2), ("p1", p1)):
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite {name} computed in Plackett-Luce marginalisation.")
        if (arr < -1e-9).any() or (arr > 1 + 1e-6).any():
            raise ValueError(f"{name} outside [0,1] in Plackett-Luce marginalisation: "
                              f"min={arr.min()}, max={arr.max()}")

    out = df_match.copy()
    out["p3"], out["p2"], out["p1"] = np.clip(p3, 0, 1), np.clip(p2, 0, 1), np.clip(p1, 0, 1)
    out["p0"] = np.clip(1 - out["p3"] - out["p2"] - out["p1"], 0, None)
    out["expected_votes"] = 3 * out["p3"] + 2 * out["p2"] + 1 * out["p1"]
    return out


def _segment_logsumexp(u: np.ndarray, match_codes: np.ndarray, n_matches: int) -> np.ndarray:
    """log(sum_j exp(u_j)) computed PER MATCH (segment), numerically stable via a
    per-match max-shift. Returns an (n_matches,) array. This is the standard
    log-sum-exp trick applied group-wise via a scatter-max + bincount, avoiding both
    the O(matches) Python loop the original implementation used AND the overflow/
    cancellation risk of forming raw exp() sums at training scale."""
    match_max = np.full(n_matches, -np.inf)
    np.maximum.at(match_max, match_codes, u)
    shifted = u - match_max[match_codes]
    sum_shifted = np.bincount(match_codes, weights=np.exp(shifted), minlength=n_matches)
    return match_max + np.log(sum_shifted)


def _neg_log_likelihood(beta: np.ndarray, X: np.ndarray, match_codes: np.ndarray, n_matches: int,
                         idx_3: np.ndarray, idx_2: np.ndarray, idx_1: np.ndarray, l2: float) -> float:
    """Fully vectorised, LOG-SPACE Plackett-Luce partial log-likelihood -- no per-match
    Python loop, and no raw exp()-then-subtract step that could go negative in floating
    point (the Phase 4 correctness bug -- see module docstring).

    Exploits the fact that (per Phase 2's validated target integrity, see
    docs/TARGET_VALIDATION.md) every match has EXACTLY one 3-vote, one 2-vote and one
    1-vote row, so their positions (idx_3/idx_2/idx_1, one entry per match, precomputed
    once outside the optimiser loop) are known in advance:
        NLL = sum_m [ log(denom_3,m) - u_{3,m} ]
            + sum_m [ log(denom_2,m) - u_{2,m} ]   where denom_2 = denom_3 - w_3
            + sum_m [ log(denom_1,m) - u_{1,m} ]   where denom_1 = denom_2 - w_2
    log(denom_2) and log(denom_1) are computed via log1p(-exp(u_k - log_denom)) --
    i.e. entirely in log space, never forming `denom - w` as a literal float
    subtraction of two potentially very different magnitudes.
    """
    u = X @ beta
    if not np.isfinite(u).all():
        return np.inf  # a non-finite utility means this beta is invalid -- tell the optimiser to back off

    log_denom_3 = _segment_logsumexp(u, match_codes, n_matches)  # log(match_sum)
    u3, u2, u1 = u[idx_3], u[idx_2], u[idx_1]

    d3 = np.clip(u3 - log_denom_3, None, _LOG1P_CLIP)
    log_denom_2 = log_denom_3 + np.log1p(-np.exp(d3))

    d2 = np.clip(u2 - log_denom_2, None, _LOG1P_CLIP)
    log_denom_1 = log_denom_2 + np.log1p(-np.exp(d2))

    nll = -np.sum(u3 - log_denom_3) - np.sum(u2 - log_denom_2) - np.sum(u1 - log_denom_1)
    if not np.isfinite(nll):
        return np.inf
    return nll + l2 * np.sum(beta ** 2)


@dataclass
class PlackettLuceModel:
    feature_names: list
    beta: np.ndarray | None = None
    # fitted preprocessing state (train-only) -- populated by fit(), reused by predict()
    _mean: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)
    _is_binary: np.ndarray | None = field(default=None, repr=False)

    def _fit_scaler(self, X: np.ndarray) -> None:
        """Determine standardisation parameters from TRAINING data only. Binary 0/1
        indicator columns are detected and left unscaled (mean 0, scale 1, i.e. a
        no-op) so their coefficients keep a direct 0->1 interpretation. Zero/near-zero
        variance columns get scale=1 to avoid division by zero (their centred value is
        then just a constant, contributing nothing informative -- correct behaviour for
        a feature with no variation in the training data)."""
        is_binary = np.array([
            np.isin(np.unique(X[:, j][~np.isnan(X[:, j])]), [0.0, 1.0]).all()
            for j in range(X.shape[1])
        ])
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std_safe = np.where(std < 1e-8, 1.0, std)
        mean[is_binary] = 0.0
        std_safe[is_binary] = 1.0
        self._mean, self._scale, self._is_binary = mean, std_safe, is_binary

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("PlackettLuceModel._transform called before fit() -- no scaler fitted.")
        return (X - self._mean) / self._scale

    def fit(self, df: pd.DataFrame, match_col: str = "match_id", vote_col: str = "brownlow_votes",
             l2: float = 1.0) -> "PlackettLuceModel":
        df = df.sort_values(match_col).reset_index(drop=True)

        # A match-level ranking model needs the WHOLE match's ranking structure intact.
        # Per-row feature dropna (e.g. a lagged-form feature missing for an early-career
        # player) can silently remove a match's own 3/2/1-vote getter without removing the
        # match itself, breaking the one-3/one-2/one-1 invariant Phase 2 validated on the
        # full table. Rather than crash or silently fit on a corrupted match, drop any
        # INCOMPLETE match here -- entirely, all its rows -- and report how many.
        vote_counts = df.groupby(match_col)[vote_col].agg(
            n3=lambda s: (s == 3).sum(), n2=lambda s: (s == 2).sum(), n1=lambda s: (s == 1).sum()
        )
        complete_matches = vote_counts[(vote_counts.n3 == 1) & (vote_counts.n2 == 1) & (vote_counts.n1 == 1)].index
        n_dropped = df[match_col].nunique() - len(complete_matches)
        if n_dropped > 0:
            print(f"  PlackettLuceModel.fit: dropping {n_dropped} match(es) with an incomplete "
                  f"3/2/1 vote structure after feature filtering (out of {df[match_col].nunique()})")
            df = df[df[match_col].isin(complete_matches)].reset_index(drop=True)

        X_raw = df[self.feature_names].to_numpy(dtype=float)
        self._fit_scaler(X_raw)
        X = self._transform(X_raw)
        votes = df[vote_col].to_numpy(dtype=float)

        # integer-code each match once; precompute the row index of the 3/2/1-vote getter
        # in every match once -- these never change across optimiser iterations, only beta
        # (and therefore u=X@beta) does, so recomputing this per NLL call (as the original
        # per-match-loop implementation effectively did) was pure waste.
        match_codes, _ = pd.factorize(df[match_col].to_numpy(), sort=False)
        n_matches = match_codes.max() + 1
        idx_3 = np.full(n_matches, -1, dtype=int)
        idx_2 = np.full(n_matches, -1, dtype=int)
        idx_1 = np.full(n_matches, -1, dtype=int)
        idx_3[match_codes[votes == 3]] = np.where(votes == 3)[0]
        idx_2[match_codes[votes == 2]] = np.where(votes == 2)[0]
        idx_1[match_codes[votes == 1]] = np.where(votes == 1)[0]
        assert (idx_3 >= 0).all() and (idx_2 >= 0).all() and (idx_1 >= 0).all(), \
            "internal error: incomplete match slipped through the filter above"

        beta0 = np.zeros(X.shape[1])
        result = minimize(
            _neg_log_likelihood, beta0, args=(X, match_codes, n_matches, idx_3, idx_2, idx_1, l2),
            method="L-BFGS-B", options={"maxiter": 300},
        )
        if not np.isfinite(result.x).all():
            raise FloatingPointError(
                f"PlackettLuceModel.fit produced non-finite coefficients (features={self.feature_names}). "
                f"This is the failure mode the Phase 4 standardisation/log-space fix was designed to prevent "
                f"-- failing loudly rather than returning a broken model."
            )
        self.beta = result.x
        return self

    def predict_match_probabilities(self, df_match: pd.DataFrame) -> pd.DataFrame:
        """df_match: rows for ONE match. Returns a copy with p3/p2/p1/p0/expected_votes columns,
        computed via exact Plackett-Luce marginalisation (coherent within the match by
        construction). Applies the TRAIN-FITTED scaler -- never refit on test data."""
        X_raw = df_match[self.feature_names].to_numpy(dtype=float)
        X = self._transform(X_raw)
        u = X @ self.beta
        return _attach_pl_probabilities(df_match, u)

    def predict(self, df: pd.DataFrame, match_col: str = "match_id") -> pd.DataFrame:
        parts = [self.predict_match_probabilities(g) for _, g in df.groupby(match_col, sort=False)]
        return pd.concat(parts, axis=0)
