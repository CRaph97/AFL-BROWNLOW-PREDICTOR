"""
2027 R&D metrics. Every function takes a predictions frame with columns
  season, match_id, player_id, brownlow_votes (actual 0-3), p3, p2, p1, p0, expected_votes
Match metrics are computed per match then averaged; season metrics on
season totals. Denominators are always returned. Bootstrap CIs resample
matches (match level) or players (season level) with a fixed seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20270101


def per_match_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, g in preds.groupby("match_id", sort=False):
        v = g["brownlow_votes"].to_numpy()
        if (v == 3).sum() != 1 or (v == 2).sum() != 1 or (v == 1).sum() != 1:
            continue
        p3 = g["p3"].to_numpy(); p2 = g["p2"].to_numpy(); p1 = g["p1"].to_numpy(); ev = g["expected_votes"].to_numpy()
        pid = g["player_id"].to_numpy()
        a3, a2, a1 = pid[v == 3][0], pid[v == 2][0], pid[v == 1][0]
        order = np.argsort(-p3, kind="stable")
        rank_of_a3 = int(np.where(pid[order] == a3)[0][0]) + 1
        pred3 = pid[order[0]]
        # sequential 3-2-1 assignment from marginals
        rem = np.ones(len(pid), bool); rem[order[0]] = False
        i2 = int(np.argmax(np.where(rem, p2, -1))); rem[i2] = False
        i1 = int(np.argmax(np.where(rem, p1, -1)))
        pred2, pred1 = pid[i2], pid[i1]
        top3_ev = set(pid[np.argsort(-ev, kind="stable")[:3]])
        y3 = (pid == a3).astype(float)
        rows.append({
            "season": int(g["season"].iloc[0]), "match_id": mid, "n_players": len(pid),
            "correct_3": pred3 == a3, "a3_in_top2": rank_of_a3 <= 2, "a3_in_top3": rank_of_a3 <= 3, "a3_rank": rank_of_a3,
            "correct_2": pred2 == a2, "correct_1": pred1 == a1,
            "exact_321": (pred3 == a3) and (pred2 == a2) and (pred1 == a1),
            "unordered_top3": {pred3, pred2, pred1} == {a3, a2, a1},
            "top3_ev_precision": len(top3_ev & {a3, a2, a1}) / 3,
            "log_loss_p3": float(-np.log(max(p3[pid == a3][0], 1e-9))),
            "brier_p3": float(np.mean((p3 - y3) ** 2)),
            "p3_of_actual_3": float(p3[pid == a3][0]),
            "ev_mae": float(np.mean(np.abs(ev - v))),
            "predicted_3": pred3, "actual_3": a3, "actual_2": a2, "actual_1": a1,
        })
    return pd.DataFrame(rows)


def match_metrics(preds: pd.DataFrame, ci: bool = False, n_boot: int = 300) -> dict:
    pm = per_match_table(preds)
    if pm.empty:
        return {"n_matches": 0}
    cols = ["correct_3", "a3_in_top2", "a3_in_top3", "exact_321", "unordered_top3", "top3_ev_precision", "log_loss_p3", "brier_p3", "ev_mae"]
    out = {"n_matches": int(len(pm))}
    for c in cols:
        out[c] = float(pm[c].astype(float).mean())
    # per-player P3 calibration
    y = (preds["brownlow_votes"] == 3).astype(float); p = preds["p3"].clip(0, 1)
    out["ece_p3"] = ece(p, y)
    out["n_player_rows"] = int(len(preds))
    if ci:
        rng = np.random.default_rng(SEED)
        boots = {c: [] for c in ("correct_3", "log_loss_p3", "exact_321")}
        arr = pm[list(boots)].astype(float).to_numpy()
        n = len(pm)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            m = arr[idx].mean(axis=0)
            for j, c in enumerate(boots):
                boots[c].append(m[j])
        for c, b in boots.items():
            out[f"{c}_ci_lo"], out[f"{c}_ci_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return out


def ece(p: pd.Series, y: pd.Series, n_bins: int = 10) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def season_metrics(preds: pd.DataFrame) -> dict:
    """Season totals from expected votes vs actual totals, computed on the
    same player universe (every player with a prediction row)."""
    t = preds.groupby(["season", "player_id"]).agg(pred=("expected_votes", "sum"), actual=("brownlow_votes", "sum")).reset_index()
    rows = []
    for s, g in t.groupby("season"):
        err = g["pred"] - g["actual"]
        g = g.assign(pred_rank=g["pred"].rank(ascending=False, method="min"), act_rank=g["actual"].rank(ascending=False, method="min"))
        winner = g.sort_values(["actual", "pred"], ascending=False).iloc[0]
        def hit(k):
            a = set(g[g["act_rank"] <= k]["player_id"]); p = set(g.sort_values(["pred", "actual"], ascending=False).head(len(a))["player_id"])
            return len(a & p) / len(a), len(a)
        t3, n3 = hit(3); t5, n5 = hit(5); t10, n10 = hit(10)
        top30 = g[g["act_rank"] <= 30]
        rows.append({"season": int(s), "n_players": int(len(g)), "season_mae": float(err.abs().mean()), "season_rmse": float(np.sqrt((err ** 2).mean())),
                     "season_bias": float(err.mean()), "spearman": float(spearmanr(g["pred"], g["actual"]).correlation),
                     "rank_mae_top30": float((top30["pred_rank"] - top30["act_rank"]).abs().mean()),
                     "winner_pred_rank": int(winner["pred_rank"]), "winner_correct": bool(winner["pred_rank"] == 1),
                     "top3_hit": t3, "top3_n": n3, "top5_hit": t5, "top5_n": n5, "top10_hit": t10, "top10_n": n10})
    return {"by_season": rows}


def pooled(by_season: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c in by_season:
            out[f"{c}_mean"] = float(by_season[c].mean()); out[f"{c}_std"] = float(by_season[c].std(ddof=0))
    return out
