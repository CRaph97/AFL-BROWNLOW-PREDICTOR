"""
Final pre-freeze audit: same-period B vs ensemble, proof that ensemble
weights and Platt calibrators are fit only on earlier seasons, and raw vs
calibrated metrics by season. Writes data/experiments/analysis/audit_*.
Run: python -m src.validation.audit_freeze
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.ensemble.stacking import fit_weights
from src.validation import metrics as M
from src.validation.analyze import load_oof, paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "data" / "experiments" / "analysis"
OOF = ROOT / "data" / "experiments" / "oof"


def platt_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Calibrator for season t fit on OOF rows of seasons < t only (never t or later)."""
    out = []
    for t in sorted(df["season"].unique()):
        prior = df[df["season"] < t]; cur = df[df["season"] == t].copy()
        if len(prior) == 0:
            continue
        lp = lambda p: np.log(np.clip(p, 1e-9, 1) / np.clip(1 - p, 1e-9, 1))
        cal = LogisticRegression(C=1e6).fit(lp(prior["p3"].to_numpy()).reshape(-1, 1), (prior["brownlow_votes"] == 3).astype(int))
        q = cal.predict_proba(lp(cur["p3"].to_numpy()).reshape(-1, 1))[:, 1]
        cur["p3"] = q / cur.assign(q=q).groupby("match_id")["q"].transform("sum").to_numpy()
        cur["calibrator_fit_on_seasons"] = f"{int(prior['season'].min())}-{int(prior['season'].max())}"
        out.append(cur)
    return pd.concat(out)


def by_season_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("season"):
        mm = M.match_metrics(g); sm = M.season_metrics(g)["by_season"][0]
        rows.append({"model": label, "season": int(s), "n_matches": mm["n_matches"], "correct_3": mm["correct_3"], "exact_321": mm["exact_321"], "log_loss_p3": mm["log_loss_p3"],
                     "brier_p3": mm["brier_p3"], "ece_p3": mm["ece_p3"], "season_mae": sm["season_mae"], "spearman": sm["spearman"]})
    return pd.DataFrame(rows)


def run() -> dict:
    ens = pd.read_parquet(OOF / "ENS_learned_ABC.parquet"); ens["player_id"] = ens["player_id"].astype(str)
    b = load_oof("B_performance_xgb_rank_tuned"); a = load_oof("A_structural_pl"); c = load_oof("C_stats_only_pl")
    seasons = sorted(ens["season"].unique())
    # same rows: ensemble only exists where all three components exist
    keys = set(zip(ens["match_id"], ens["player_id"]))
    def same(df):
        m = pd.Series(list(zip(df["match_id"], df["player_id"]))).isin(keys).to_numpy(); return df[m].reset_index(drop=True)
    b_s, a_s = same(b[b["season"].isin(seasons)]), same(a[a["season"].isin(seasons)])
    tab = pd.concat([by_season_metrics(b_s, "Performance ML (B)"), by_season_metrics(ens, "Learned ensemble (A+B+C)"), by_season_metrics(a_s, "Structural (A)")])
    tab.to_csv(AN / "audit_same_period_by_season.csv", index=False)
    pooled = tab.groupby("model")[["n_matches", "correct_3", "exact_321", "log_loss_p3", "brier_p3", "ece_p3", "season_mae", "spearman"]].agg(
        {"n_matches": "sum", **{c: "mean" for c in ["correct_3", "exact_321", "log_loss_p3", "brier_p3", "ece_p3", "season_mae", "spearman"]}}).reset_index()
    pooled["n_seasons"] = len(seasons); pooled.to_csv(AN / "audit_same_period_pooled.csv", index=False)
    pv = tab.pivot(index="season", columns="model", values=["correct_3", "log_loss_p3", "exact_321", "season_mae"])
    wins = {m: {"ens_better": int((pv[m]["Learned ensemble (A+B+C)"] < pv[m]["Performance ML (B)"]).sum() if m in ("log_loss_p3", "season_mae") else (pv[m]["Learned ensemble (A+B+C)"] > pv[m]["Performance ML (B)"]).sum()),
                "ties": int((pv[m]["Learned ensemble (A+B+C)"] == pv[m]["Performance ML (B)"]).sum())} for m in ("correct_3", "log_loss_p3", "exact_321", "season_mae")}
    boots = [paired_bootstrap(ens, b_s, m) for m in ("correct_3", "log_loss_p3", "exact_321", "brier_p3")]
    pd.DataFrame(boots).assign(a="ensemble", b="B").to_csv(AN / "audit_paired_bootstrap_ens_vs_B.csv", index=False)
    # proof: stored weights == refit on seasons < t only; and they change if t were (wrongly) included
    w = pd.read_csv(AN / "ensemble_weights.csv")
    oof = {"A_structural_pl": a, "B_performance_xgb_rank_tuned": b, "C_stats_only_pl": c}
    proof = []
    for t in (seasons[0], seasons[len(seasons) // 2], seasons[-1]):
        prior = [s for s in sorted(a["season"].unique()) if s < t]
        w_prior, names = fit_weights(oof, prior)
        stored = w[w["test_season"] == t].iloc[0]
        w_leak, _ = fit_weights(oof, prior + [t])
        proof.append({"test_season": int(t), "n_prior_seasons": len(prior), "stored_n_train_seasons": int(stored["n_train_seasons"]),
                      "max_abs_diff_stored_vs_refit_prior_only": float(max(abs(stored[f"w_{n}"] - x) for n, x in zip(names, w_prior))),
                      "max_abs_diff_if_test_season_included": float(max(abs(stored[f"w_{n}"] - x) for n, x in zip(names, w_leak)))})
    pd.DataFrame(proof).to_csv(AN / "audit_ensemble_weight_proof.csv", index=False)
    # calibration: raw vs walk-forward Platt, by season and pooled, for B and the ensemble
    cal_rows = []
    for label, df in (("Performance ML (B)", b), ("Learned ensemble (A+B+C)", ens)):
        raw = by_season_metrics(df[df["season"] >= 2013], f"{label} raw")
        pl = platt_walk_forward(df); cal = by_season_metrics(pl, f"{label} platt")
        cal_rows += [raw, cal]
        fit_spans = pl.drop_duplicates("season")[["season", "calibrator_fit_on_seasons"]]
        fit_spans.assign(model=label).to_csv(AN / f"audit_platt_fit_spans_{'B' if 'B' in label and 'ensemble' not in label else 'ens'}.csv", index=False)
    calib = pd.concat(cal_rows); calib.to_csv(AN / "audit_calibration_by_season.csv", index=False)
    calib_pooled = calib.groupby("model")[["correct_3", "log_loss_p3", "brier_p3", "ece_p3"]].mean().reset_index(); calib_pooled.to_csv(AN / "audit_calibration_pooled.csv", index=False)
    summary = {"seasons": [int(s) for s in seasons], "same_period_pooled": pooled.round(4).to_dict("records"), "ens_vs_B_wins": wins, "paired_bootstrap": boots,
               "weight_proof": proof, "calibration_pooled": calib_pooled.round(4).to_dict("records")}
    (AN / "audit_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps(s, indent=1, default=str))
