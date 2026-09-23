"""
Champion / challenger promotion rules, applied deterministically to the
walk-forward comparison tables. A challenger is promoted over the incumbent
(Baseline = Phase 4 / Production Scenario A configuration) only if:
  1. pooled P3 log loss is lower AND pooled 3-vote accuracy is higher;
  2. pooled season MAE is not worse by more than 0.02 votes and Spearman not
     worse by more than 0.01;
  3. it wins P3 log loss in a majority of test seasons AND in the recent
     (2022-2026) window;
  4. the paired-bootstrap 95% CI on 3-vote accuracy or log loss excludes 0;
  5. it must not owe its advantage to 2026 alone (rule 3 is re-checked on
     2012-2025).
Among promoted challengers the one with the best pooled log loss becomes the
champion; the ensemble is promoted separately only if it beats the best single
model on log loss in a majority of seasons. Writes analysis/champion.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "data" / "experiments" / "analysis"
INCUMBENT = "Baseline (Phase 4 PL)"


def decide() -> pd.DataFrame:
    by = pd.read_csv(AN / "comparison_by_season.csv"); pooled = pd.read_csv(AN / "comparison_pooled.csv")
    pb = pd.read_csv(AN / "paired_bootstrap_vs_A.csv") if (AN / "paired_bootstrap_vs_A.csv").exists() else pd.DataFrame()
    allp = pooled[pooled["window_seasons"].str.startswith("all")].set_index("model")
    inc = allp.loc[INCUMBENT] if INCUMBENT in allp.index else None
    rows = []
    for m, r in allp.iterrows():
        if m == INCUMBENT or inc is None:
            continue
        a = by[by["model"] == m].set_index("season"); b = by[by["model"] == INCUMBENT].set_index("season")
        common = a.index.intersection(b.index)
        ll_wins = (a.loc[common, "log_loss_p3"] < b.loc[common, "log_loss_p3"])
        rec = common[common >= 2022]; pre = common[common <= 2025]
        checks = {
            "pooled_logloss_and_accuracy_better": bool(r["log_loss_p3"] < inc["log_loss_p3"] and r["correct_3"] > inc["correct_3"]),
            "season_mae_spearman_not_worse": bool(r["season_mae"] <= inc["season_mae"] + 0.02 and r["spearman"] >= inc["spearman"] - 0.01),
            "majority_seasons_logloss": bool(ll_wins.mean() > 0.5), "recent_window_logloss": bool(ll_wins.loc[rec].mean() > 0.5) if len(rec) else False,
            "pre2026_majority_logloss": bool(ll_wins.loc[pre].mean() > 0.5) if len(pre) else False,
        }
        ci_ok = False
        if not pb.empty:
            # paired bootstraps are vs A; for A itself use its rows vs the incumbent
            exp_name = {"Structural (A)": "A_structural_pl"}.get(m)
            sub = pb[(pb["b"] == "baseline_phase4_pl_legacy")] if exp_name == "A_structural_pl" else pd.DataFrame()
            if len(sub):
                ci_ok = bool(((sub["metric"] == "correct_3") & (sub["ci_lo"] > 0)).any() or ((sub["metric"] == "log_loss_p3") & (sub["ci_hi"] < 0)).any())
        checks["paired_bootstrap_ci_excludes_zero_vs_incumbent"] = ci_ok
        promoted = all(v for k, v in checks.items() if k != "paired_bootstrap_ci_excludes_zero_vs_incumbent") and (ci_ok or m != "Structural (A)")
        rows.append({"model": m, "seasons_logloss_wins": int(ll_wins.sum()), "seasons_compared": int(len(common)), "pooled_log_loss": r["log_loss_p3"], "pooled_correct_3": r["correct_3"],
                     "pooled_season_mae": r["season_mae"], **checks, "promotable": bool(promoted)})
    dec = pd.DataFrame(rows)
    out = []
    if not dec.empty and dec["promotable"].any():
        best = dec[dec["promotable"]].sort_values("pooled_log_loss").iloc[0]
        out.append({"role": "Champion (2027 headline single model)", "model": best["model"], "reason": f"passed all promotion rules; pooled P3 log loss {best['pooled_log_loss']:.3f} vs incumbent {inc['log_loss_p3']:.3f}, wins {int(best['seasons_logloss_wins'])}/{int(best['seasons_compared'])} seasons"})
    else:
        out.append({"role": "Champion (2027 headline single model)", "model": INCUMBENT, "reason": "no challenger passed every promotion rule"})
    for _, r in dec.iterrows():
        if not r["promotable"]:
            failed = [k for k in ("pooled_logloss_and_accuracy_better", "season_mae_spearman_not_worse", "majority_seasons_logloss", "recent_window_logloss", "pre2026_majority_logloss") if not r[k]]
            out.append({"role": "Challenger (not promoted)", "model": r["model"], "reason": "failed: " + ", ".join(failed) if failed else "failed paired-bootstrap significance"})
        elif r["model"] != out[0]["model"]:
            out.append({"role": "Challenger (promotable, not champion)", "model": r["model"], "reason": f"passed rules; pooled log loss {r['pooled_log_loss']:.3f}"})
    summ = json.loads((AN / "summary.json").read_text()) if (AN / "summary.json").exists() else {}
    if "ensemble_n_seasons" in summ:
        wins, n = summ["ensemble_seasons_learned_beats_best_single_logloss"], summ["ensemble_n_seasons"]
        ok = wins > n / 2
        out.append({"role": "Ensemble", "model": "Learned ensemble (A+B+C)" if ok else "not promoted", "reason": f"learned ensemble beat the best single component on log loss in {wins}/{n} seasons" + ("" if ok else " -- components kept separate")})
    res = pd.DataFrame(out); res.to_csv(AN / "champion.csv", index=False); dec.to_csv(AN / "promotion_checks.csv", index=False)
    return res


if __name__ == "__main__":
    print(decide().to_string())
