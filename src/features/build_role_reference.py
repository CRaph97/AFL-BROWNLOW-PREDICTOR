"""
Phase 3, section H: role/position data.

Ground truth: the AFL's own officially-listed player position (KEY_FORWARD,
MEDIUM_FORWARD, MIDFIELDER, MIDFIELDER_FORWARD, RUCK, KEY_DEFENDER,
MEDIUM_DEFENDER) is available for 2021-2025 via the torp/torpdata ecosystem
already validated in Phase 2 (`docs/EVENT_DATA_2021_AUDIT.md`). This is a
STATIC, once-per-season listed position -- not a dynamic per-match role -- per
the Phase 1 audit's expectation that only static position data would be
publicly available.

No equivalent official position field exists in our afltables/footywire-derived
CORE/ADVANCED tables for any year, and none exists publicly before ~2021 at
all (see docs/DATA_SOURCE_AUDIT.md). This module therefore does two things:

1. Builds a (season, team, player) -> official_position reference table for
   2021-2025 from the downloaded torp player_details files (join by normalised
   surname + team + season, the same approach and the same caution about
   surname collisions as the ADVANCED footywire join in Phase 2).
2. Leaves the pre-2021 gap as a gap -- see build_role_proxy.py for the
   statistically-derived approximation used to extend coverage backward, and
   docs/ROLE_ANALYSIS.md for how well that proxy agrees with this ground
   truth in the overlap years.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "torpdata_pilot"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"

AVAILABLE_SEASONS = [2021, 2022, 2023, 2024, 2025]


def normalise_surname(series: pd.Series) -> pd.Series:
    return series.str.split().str[-1].str.lower().str.replace(r"[^a-z]", "", regex=True)


def build() -> pd.DataFrame:
    team_map = dict(pd.read_csv(CONFIG_DIR / "team_mapping.csv")[["source_name", "canonical_team_id"]].values)

    frames = []
    for season in AVAILABLE_SEASONS:
        path = RAW_DIR / f"player_details_{season}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)[["season", "team", "surname", "position"]].copy()
        df["team_id"] = df["team"].map(team_map)
        df["surname_key"] = normalise_surname(df["surname"])
        frames.append(df[["season", "team_id", "surname_key", "position"]])

    ref = pd.concat(frames, ignore_index=True)

    # Log (not silently drop) surname collisions within the same season+team -- a genuine limitation
    # of a surname-only join, same caveat as the ADVANCED footywire join in Phase 2.
    dup = ref.duplicated(subset=["season", "team_id", "surname_key"], keep=False)
    if dup.any():
        REPORTS_DIR = ROOT / "reports"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ref[dup].to_csv(REPORTS_DIR / "role_reference_surname_collisions.csv", index=False)
        ref = ref[~dup]  # drop ambiguous rows entirely rather than guess
        print(f"WARNING: dropped {dup.sum()} rows with an ambiguous (season, team, surname) role lookup -- "
              f"logged to reports/role_reference_surname_collisions.csv")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(PROCESSED_DIR / "role_reference_2021_2025.parquet", index=False)
    print(f"Wrote {len(ref):,} rows -> data/processed/role_reference_2021_2025.parquet")
    print(ref["position"].value_counts())
    return ref


if __name__ == "__main__":
    build()
