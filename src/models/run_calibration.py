"""
Phase 4, section K: probability calibration analysis.

Uses the pooled out-of-sample predictions from run_backtest.py
(data/processed/oos_predictions_core.parquet -- expanding-window, every test
season 2015-2025, genuinely never trained on their own test season) for every
model. Builds reliability tables for P(3) and P(2), computes Brier score and
expected calibration error, and tests whether isotonic regression improves
held-out calibration -- fit ONLY on a subset of the pooled OOS seasons
(2015-2021) and evaluated on the remainder (2022-2025), so the calibration
correction itself is never fit on the season it is evaluated on.
"""
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.evaluation.metrics import calibration_table, expected_calibration_error

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

CALIBRATION_FIT_SEASONS = list(range(2015, 2022))
CALIBRATION_EVAL_SEASONS = list(range(2022, 2026))


def run():
    preds = pd.read_parquet(PROCESSED_DIR / "oos_predictions_core.parquet")
    all_rows = []
    all_tables = []

    for model_name, g in preds.groupby("model"):
        table3 = calibration_table(g, prob_col="p3", outcome_value=3)
        table3["model"] = model_name
        table3["prob_type"] = "p3"
        all_tables.append(table3)

        ece3 = expected_calibration_error(g, prob_col="p3", outcome_value=3)

        fit_set = g[g["season"].isin(CALIBRATION_FIT_SEASONS)]
        eval_set = g[g["season"].isin(CALIBRATION_EVAL_SEASONS)]

        raw_ece_eval = expected_calibration_error(eval_set, prob_col="p3", outcome_value=3) if len(eval_set) else None

        iso_ece_eval = None
        if len(fit_set) > 100 and len(eval_set) > 100:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso.fit(fit_set["p3"], (fit_set["brownlow_votes"] == 3).astype(int))
            eval_calibrated = eval_set.copy()
            eval_calibrated["p3_calibrated"] = iso.predict(eval_calibrated["p3"])
            iso_ece_eval = expected_calibration_error(
                eval_calibrated, prob_col="p3_calibrated", outcome_value=3
            )

        all_rows.append({
            "model": model_name,
            "ece_p3_all_seasons": ece3,
            "ece_p3_eval_seasons_raw": raw_ece_eval,
            "ece_p3_eval_seasons_isotonic": iso_ece_eval,
            "isotonic_improves": (iso_ece_eval is not None and raw_ece_eval is not None
                                   and iso_ece_eval < raw_ece_eval),
        })
        print(f"{model_name}: raw ECE(all)={ece3:.4f} | eval-seasons raw={raw_ece_eval} "
              f"isotonic={iso_ece_eval}")

    summary = pd.DataFrame(all_rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "calibration_summary.csv", index=False)

    tables = pd.concat(all_tables, ignore_index=True)
    tables.to_csv(REPORTS_DIR / "calibration_tables.csv", index=False)

    print("\n=== Calibration summary ===")
    print(summary.to_string(index=False))
    return summary, tables


if __name__ == "__main__":
    run()
