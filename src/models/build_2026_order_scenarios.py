"""Orchestrates reports/2026_order_scenarios.csv from the two models' already-
persisted raw simulation draws (production: run_2026_montecarlo.py's 100,000
sims; objective: run_2026_objective_montecarlo.py's 20,000 sims). Pure
aggregation -- computes nothing new about either model."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.order_scenarios import build_order_scenarios_csv

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def load(totals_path: Path, index_path: Path):
    totals = np.load(totals_path)
    players = pd.read_csv(index_path)
    return totals, players


def run() -> pd.DataFrame:
    prod = load(PROCESSED_DIR / "mc_totals_2026.npy", REPORTS_DIR / "2026_mc_player_index.csv")
    obj = load(PROCESSED_DIR / "mc_totals_objective_2026.npy", REPORTS_DIR / "2026_objective_mc_player_index.csv")

    out = build_order_scenarios_csv({"Production": prod, "Objective": obj})
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS_DIR / "2026_order_scenarios.csv", index=False)
    print(f"Wrote reports/2026_order_scenarios.csv ({len(out)} rows)")
    return out


if __name__ == "__main__":
    run()
