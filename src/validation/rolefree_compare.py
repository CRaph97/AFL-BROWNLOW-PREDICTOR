"""
Pre-freeze closure: current Structural A vs role-free A on identical folds
and the common-match / common-player universe, by season and pooled, with
paired bootstraps and the pre-declared promotion rules (A as incumbent).
Writes analysis/rolefree_* and analysis/a_decision.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.validation.analyze import load_oof, comparison_tables, paired_bootstrap, restrict_to_common_matches

ROOT = Path(__file__).resolve().parents[2]
AN = ROOT / "data" / "experiments" / "analysis"


def run() -> dict:
    out = {}
    for window in ("expanding", "recent8"):
        oofs = {"A_structural_pl": load_oof("A_structural_pl", window), "A_structural_pl_rolefree": load_oof("A_structural_pl_rolefree", window)}
        by, pooled = comparison_tables(oofs)
        by["model"] = by["experiment"].map({"A_structural_pl": "A (current, with role)", "A_structural_pl_rolefree": "A role-free"})
        pooled["model"] = pooled["model"].replace({"A_structural_pl": "A (current, with role)", "Structural (A)": "A (current, with role)", "A_structural_pl_rolefree": "A role-free"})
        by.to_csv(AN / f"rolefree_by_season_{window}.csv", index=False); pooled.to_csv(AN / f"rolefree_pooled_{window}.csv", index=False)
        c = restrict_to_common_matches(oofs)
        boots = [paired_bootstrap(c["A_structural_pl_rolefree"], c["A_structural_pl"], m) for m in ("correct_3", "log_loss_p3", "exact_321", "brier_p3")]
        pd.DataFrame(boots).to_csv(AN / f"rolefree_paired_bootstrap_{window}.csv", index=False)
        pv = by.pivot(index="season", columns="model", values=["correct_3", "log_loss_p3", "exact_321", "season_mae", "spearman", "brier_p3", "ece_p3"])
        rf, cur = "A role-free", "A (current, with role)"
        wins = {m: int((pv[m][rf] > pv[m][cur]).sum()) if m in ("correct_3", "exact_321", "spearman") else int((pv[m][rf] < pv[m][cur]).sum()) for m in ("correct_3", "log_loss_p3", "exact_321", "season_mae", "spearman", "brier_p3", "ece_p3")}
        seasons = sorted(pv.index)
        allp = pooled[pooled["window_seasons"].str.startswith("all")].set_index("model")
        r, i = allp.loc[rf], allp.loc[cur]
        ll = pv["log_loss_p3"]
        checks = {
            "pooled_logloss_and_accuracy_better": bool(r["log_loss_p3"] < i["log_loss_p3"] and r["correct_3"] > i["correct_3"]),
            "season_mae_spearman_not_worse": bool(r["season_mae"] <= i["season_mae"] + 0.02 and r["spearman"] >= i["spearman"] - 0.01),
            "majority_seasons_logloss": bool((ll[rf] < ll[cur]).mean() > 0.5),
            "recent_window_logloss": bool((ll[rf] < ll[cur])[[s for s in seasons if s >= 2022]].mean() > 0.5),
            "pre2026_majority_logloss": bool((ll[rf] < ll[cur])[[s for s in seasons if s <= 2025]].mean() > 0.5),
            "paired_bootstrap_ci_excludes_zero": bool(any((b["metric"] == "correct_3" and b["ci_lo"] > 0) or (b["metric"] == "log_loss_p3" and b["ci_hi"] < 0) for b in boots)),
        }
        out[window] = {"n_seasons": len(seasons), "wins_rolefree_vs_current": wins, "pooled": {"current": {k: float(i[k]) for k in ("correct_3", "exact_321", "log_loss_p3", "brier_p3", "ece_p3", "season_mae", "spearman")},
                       "rolefree": {k: float(r[k]) for k in ("correct_3", "exact_321", "log_loss_p3", "brier_p3", "ece_p3", "season_mae", "spearman")}},
                       "paired_bootstrap": boots, "promotion_checks": checks, "promotable": all(checks.values())}
    decision = {"decision": "A role-free" if out["expanding"]["promotable"] else "A (current, with role) retained",
                "reason": ("role-free A passed every promotion rule on the expanding window" if out["expanding"]["promotable"]
                           else "role-free A failed: " + ", ".join(k for k, v in out["expanding"]["promotion_checks"].items() if not v)),
                "windows": out}
    (AN / "a_decision.json").write_text(json.dumps(decision, indent=1, default=str))
    return decision


if __name__ == "__main__":
    d = run(); print(json.dumps({k: v for k, v in d.items() if k != "windows"}, indent=1)); print(json.dumps(d["windows"]["expanding"], indent=1, default=str)[:2500])
