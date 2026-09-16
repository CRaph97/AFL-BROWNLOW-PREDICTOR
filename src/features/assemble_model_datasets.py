"""
Phase 4, section A5: assemble the three separate, NOT-merged model-ready
datasets (CORE, ADVANCED, EXPERIMENTAL), incorporating the Phase 4 leakage
fixes (lagged role, lagged form features) on top of the Phase 3 safe features
(match/team-relative, context, teammate, composite).

Role is now `role_lagged` (from build_role_lagged.py, strictly prior-games-only
for 1999-2020, real label for 2021-2025, "UNKNOWN" where insufficient history)
-- NOT the Phase 3 `player_season_role.parquet`, which is retained on disk
only as a record of the original (leaky) Phase 3 exploratory work.
"""
from pathlib import Path

import pandas as pd

from . import build_composite_indices, build_context_features, build_relative_features, build_teammate_features
from .build_lagged_form_features import build as build_lagged_form

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

CORE_START, CORE_END = 2003, 2025
ADVANCED_START, ADVANCED_END = 2015, 2025
EXPERIMENTAL_START, EXPERIMENTAL_END = 2021, 2025


def _base_with_safe_features(core: pd.DataFrame) -> pd.DataFrame:
    relative = build_relative_features.build(core)
    context = build_context_features.build(core)
    teammate = build_teammate_features.build(core)
    composite = build_composite_indices.build(core)
    return pd.concat([core, relative, context, teammate, composite], axis=1)


def build_core():
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    role_lagged = pd.read_parquet(PROCESSED_DIR / "player_match_role_lagged.parquet")[
        ["match_id", "player_id", "role", "role_source"]
    ]
    lagged_form = build_lagged_form(core)

    base = _base_with_safe_features(core)
    base = base.merge(role_lagged, on=["match_id", "player_id"], how="left")
    base = base.merge(
        lagged_form.drop(columns=["match_id", "player_id"]), left_index=True, right_index=True, how="left"
    )

    out = base[base["season"].between(CORE_START, CORE_END)].copy()
    out.to_parquet(PROCESSED_DIR / "model_core.parquet", index=False)
    print(f"model_core.parquet: {len(out):,} rows, {out.shape[1]} cols, seasons {CORE_START}-{CORE_END}")
    return out


def build_advanced():
    core_full = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    advanced = pd.read_parquet(PROCESSED_DIR / "player_match_advanced_2010_2025.parquet")
    role_lagged = pd.read_parquet(PROCESSED_DIR / "player_match_role_lagged.parquet")[
        ["match_id", "player_id", "role", "role_source"]
    ]
    lagged_form = build_lagged_form(core_full)

    base = _base_with_safe_features(core_full)
    adv_only_cols = [c for c in advanced.columns if c not in core_full.columns]
    base = base.merge(advanced[["match_id", "player_id"] + adv_only_cols], on=["match_id", "player_id"], how="left")
    base = base.merge(role_lagged, on=["match_id", "player_id"], how="left")
    base = base.merge(
        lagged_form.drop(columns=["match_id", "player_id"]), left_index=True, right_index=True, how="left"
    )

    out = base[base["season"].between(ADVANCED_START, ADVANCED_END)].copy()
    out.to_parquet(PROCESSED_DIR / "model_advanced.parquet", index=False)
    print(f"model_advanced.parquet: {len(out):,} rows, {out.shape[1]} cols, seasons {ADVANCED_START}-{ADVANCED_END}")
    return out


if __name__ == "__main__":
    build_core()
    build_advanced()
