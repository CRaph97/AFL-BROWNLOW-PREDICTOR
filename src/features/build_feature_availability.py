"""Computes, for every column in the Phase 3 analytical dataset, the first and
last season with >=95% non-null coverage, and classifies the family. Backs
docs/FEATURE_REGISTRY.md and reports/feature_availability.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

FAMILY_MAP = {
    "kicks": "raw_volume", "marks": "raw_volume", "handballs": "raw_volume", "disposals": "raw_volume",
    "goals": "raw_scoring", "behinds": "raw_scoring", "goal_assists": "raw_scoring",
    "hitouts": "raw_ruck", "tackles": "raw_contest", "rebound_50s": "raw_territory",
    "inside_50s": "raw_territory", "clearances": "raw_clearance", "clangers": "raw_efficiency",
    "frees_for": "raw_efficiency", "frees_against": "raw_efficiency",
    "contested_possessions": "raw_contest", "uncontested_possessions": "raw_volume",
    "contested_marks": "raw_contest", "marks_inside_50": "raw_scoring", "one_percenters": "raw_defence",
    "bounces": "raw_volume", "time_on_ground_pct": "raw_workload",
    "effective_disposals": "advanced_efficiency", "disposal_efficiency_pct": "advanced_efficiency",
    "centre_clearances": "advanced_clearance", "stoppage_clearances": "advanced_clearance",
    "score_involvements": "advanced_scoring", "metres_gained": "advanced_territory",
    "turnovers": "advanced_efficiency", "intercepts": "advanced_defence", "tackles_inside_50": "advanced_contest",
    "afl_fantasy_points": "advanced_composite_external", "supercoach_points": "advanced_composite_external",
    "is_win": "match_outcome", "is_loss": "match_outcome", "is_draw": "match_outcome",
    "margin": "match_outcome", "absolute_margin": "match_outcome",
    "is_close_game": "match_outcome", "is_blowout": "match_outcome",
    "role": "role", "is_proxy_position": "role",
    "possession_impact_index": "composite", "contest_index": "composite", "clearance_index": "composite",
    "scoring_index": "composite", "territory_index": "composite", "defensive_index": "composite",
}


def _family(col: str) -> str:
    if col in FAMILY_MAP:
        return FAMILY_MAP[col]
    for suffix, fam in [
        ("_match_rank", "match_relative"), ("_match_pct", "match_relative"), ("_match_z", "match_relative"),
        ("_team_rank", "team_relative"), ("_team_share", "team_relative"),
        ("_gap_best_team", "teammate_competition"), ("_gap_2nd_team", "teammate_competition"),
        ("_opp_best_gap", "opponent_relative"), ("n_teammates_", "teammate_competition"),
        ("team_disposal_", "teammate_competition"), ("_n_components", "composite_metadata"),
    ]:
        if suffix in col:
            return fam
    return "other"


def build() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "analytical_features_v1.parquet")
    rows = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and col != "role":
            continue
        if col == "brownlow_votes":
            continue
        by_season = df.groupby("season")[col].apply(lambda s: s.notna().mean())
        reliable = by_season[by_season >= 0.95]
        rows.append({
            "feature": col,
            "family": _family(col),
            "first_reliable_season": int(reliable.index.min()) if len(reliable) else None,
            "last_reliable_season": int(reliable.index.max()) if len(reliable) else None,
            "pct_seasons_reliable": round(len(reliable) / by_season.notna().sum(), 3) if by_season.notna().sum() else 0,
            "overall_pct_available": round(df[col].notna().mean(), 4),
        })
    result = pd.DataFrame(rows).sort_values(["family", "feature"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "feature_availability.csv", index=False)
    print(f"Wrote {len(result)} features -> reports/feature_availability.csv")
    return result


if __name__ == "__main__":
    build()
