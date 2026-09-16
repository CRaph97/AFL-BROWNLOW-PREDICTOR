"""
Phase 2C: per-season, per-feature availability coverage matrix.

Distinguishes STRUCTURALLY UNAVAILABLE (the whole season has 0% non-null --
the stat did not exist yet) from MISSING OBSERVATION (some rows null within a
season where the stat otherwise exists -- a real data gap worth investigating).

Writes a machine-readable table to reports/coverage_matrix.csv.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

CORE_FEATURES = [
    "kicks", "marks", "handballs", "disposals", "goals", "behinds", "hitouts",
    "tackles", "rebound_50s", "inside_50s", "clearances", "clangers",
    "frees_for", "frees_against", "contested_possessions", "uncontested_possessions",
    "contested_marks", "marks_inside_50", "one_percenters", "bounces", "goal_assists",
    "time_on_ground_pct",
]

ADVANCED_FEATURES = [
    "effective_disposals", "disposal_efficiency_pct", "centre_clearances",
    "stoppage_clearances", "score_involvements", "metres_gained", "turnovers",
    "intercepts", "tackles_inside_50",
]


def build() -> pd.DataFrame:
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    adv_path = PROCESSED_DIR / "player_match_advanced_2010_2025.parquet"
    adv = pd.read_parquet(adv_path) if adv_path.exists() else None

    rows = []
    for season, grp in core.groupby("season"):
        for feat in CORE_FEATURES:
            pct = grp[feat].notna().mean()
            rows.append({"season": season, "feature": feat, "source": "afltables_core", "pct_available": round(pct, 4)})
        if adv is not None:
            adv_grp = adv[adv["season"] == season]
            for feat in ADVANCED_FEATURES:
                if feat in adv_grp.columns:
                    pct = adv_grp[feat].notna().mean()
                else:
                    pct = 0.0
                rows.append({"season": season, "feature": feat, "source": "footywire_advanced", "pct_available": round(pct, 4)})

    matrix = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(REPORTS_DIR / "coverage_matrix.csv", index=False)

    # Pivoted, human-scannable wide view too
    wide = matrix.pivot_table(index="feature", columns="season", values="pct_available")
    wide.to_csv(REPORTS_DIR / "coverage_matrix_wide.csv")

    print(f"Wrote reports/coverage_matrix.csv ({len(matrix):,} rows) and reports/coverage_matrix_wide.csv")
    return matrix


def summarise_transitions(matrix: pd.DataFrame) -> pd.DataFrame:
    """For each feature, find the first season where coverage crosses 95% and stays there,
    to distinguish 'structurally unavailable early' from 'available throughout'."""
    out = []
    for feat, grp in matrix.groupby("feature"):
        grp = grp.sort_values("season")
        available_seasons = grp[grp["pct_available"] >= 0.95]["season"]
        first_reliable = available_seasons.min() if len(available_seasons) else None
        max_pct = grp["pct_available"].max()
        out.append({"feature": feat, "first_reliable_season": first_reliable, "max_pct_available": round(max_pct, 3)})
    summary = pd.DataFrame(out).sort_values("first_reliable_season")
    summary.to_csv(REPORTS_DIR / "coverage_first_reliable_season.csv", index=False)
    return summary


if __name__ == "__main__":
    m = build()
    s = summarise_transitions(m)
    print(s.to_string(index=False))
