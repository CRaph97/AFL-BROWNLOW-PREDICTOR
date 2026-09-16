"""
Phase 3, section L/Q10: correlation matrix among a curated set of candidate
features, to identify collinearity/redundancy before Phase 4 feature selection.
Restricted to a representative subset (not all 148 analytical columns) because
a 148x148 matrix is not human-interpretable -- this is exactly the
"uncontrolled feature explosion" the brief warns against; the full pairwise
matrix among every engineered variant is implicitly available by rerunning
this script with a different column list if needed later.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

FEATURES = [
    "disposals", "kicks", "handballs", "effective_disposals", "disposal_efficiency_pct",
    "contested_possessions", "uncontested_possessions", "clearances", "centre_clearances",
    "stoppage_clearances", "tackles", "goals", "behinds", "goal_assists", "score_involvements",
    "marks", "contested_marks", "inside_50s", "metres_gained", "hitouts", "one_percenters",
    "rebound_50s", "frees_for", "frees_against",
    "disposals_match_z", "disposals_team_share", "disposals_gap_best_team",
    "contested_possessions_match_z", "clearances_match_z", "goals_match_z",
    "possession_impact_index", "contest_index", "scoring_index", "territory_index", "defensive_index",
    "margin", "absolute_margin", "afl_fantasy_points", "supercoach_points",
    "brownlow_votes",
]


def build() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "analytical_features_v1.parquet")
    df = df[df["season"].between(2015, 2025)]  # window where every listed feature is available
    cols = [c for c in FEATURES if c in df.columns]
    corr = df[cols].corr(method="spearman")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")
    print(f"Wrote {corr.shape[0]}x{corr.shape[1]} correlation matrix -> reports/correlation_matrix.csv")
    return corr


def find_high_collinearity(corr: pd.DataFrame, threshold: float = 0.85):
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] == "brownlow_votes" or cols[j] == "brownlow_votes":
                continue
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((cols[i], cols[j], round(r, 3)))
    return sorted(pairs, key=lambda p: -abs(p[2]))


if __name__ == "__main__":
    c = build()
    high = find_high_collinearity(c)
    print(f"\n{len(high)} feature pairs with |Spearman r| >= 0.85:")
    for a, b, r in high:
        print(f"  {a} <-> {b}: {r}")
