"""
Phase 3: assemble the versioned analytical feature table from the canonical
CORE/ADVANCED datasets plus every build_*.py feature module in this package.

Does NOT modify data/processed/player_match_core_1984_2025.parquet or
player_match_advanced_2010_2025.parquet (the Phase 2 canonical, target-bearing
tables) -- this is a separate, additive output:

  data/processed/analytical_features_v1.parquet

Grain: one row per player-match, 1984-2025 (home-and-away only), carrying the
target (brownlow_votes) plus every engineered feature. Role features are only
populated from 1999 onward (see build_role_proxy.py); ADVANCED-sourced
components are only populated from their confirmed start years
(docs/DATA_COVERAGE.md). Nothing is zero- or mean-imputed -- missingness is
preserved and documented.
"""
from pathlib import Path

import pandas as pd

from . import build_composite_indices, build_context_features, build_relative_features, build_teammate_features

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

VERSION = "v1"


def build() -> pd.DataFrame:
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    advanced = pd.read_parquet(PROCESSED_DIR / "player_match_advanced_2010_2025.parquet")
    role = pd.read_parquet(PROCESSED_DIR / "player_season_role.parquet")

    advanced_only_cols = [c for c in advanced.columns if c not in core.columns]
    advanced_extra = advanced[["match_id", "player_id"] + advanced_only_cols]

    base = core.merge(advanced_extra, on=["match_id", "player_id"], how="left")
    base = base.merge(role, on=["season", "player_id"], how="left")

    relative = build_relative_features.build(core)
    context = build_context_features.build(core)
    teammate = build_teammate_features.build(core)
    composite = build_composite_indices.build(
        base, extra_components=build_composite_indices.ADVANCED_EXTRA_COMPONENTS
    )

    analytical = pd.concat([base, relative, context, teammate, composite], axis=1)

    out_path = PROCESSED_DIR / f"analytical_features_{VERSION}.parquet"
    analytical.to_parquet(out_path, index=False)
    print(f"Wrote {len(analytical):,} rows x {analytical.shape[1]} columns -> {out_path.relative_to(ROOT)}")
    return analytical


if __name__ == "__main__":
    build()
