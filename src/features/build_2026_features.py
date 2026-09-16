"""
Phase 5: apply the exact same Phase 3/4 feature-engineering functions used for
CORE/ADVANCED to the 2026-extended tables built by build_2026_extension.py,
producing 2026-inclusive model-ready tables:
    data/processed/model_core_2026.parquet      (2003-2026, mirrors model_core.parquet)
    data/processed/model_advanced_2026.parquet  (2015-2026, mirrors model_advanced.parquet)
    data/processed/player_match_role_lagged_2026.parquet

Every feature function used here (build_relative_features, build_context_features,
build_teammate_features, build_composite_indices, build_lagged_form_features) is a
pure function of box-score stats using ONLY `.shift(1)`-style lagged/expanding
windows or within-match/within-team groupings -- none of them read brownlow_votes
for anything except lagged_form's own strictly-prior-games "reputation" candidate
column, which by construction cannot see 2026's (nonexistent) votes. Running them
on the extended (1984-2026 / 2010-2026) table therefore produces 2026-row features
using only pre-2026 and same-match information -- no leakage, consistent with the
Phase 4 methodology.

Role for 2026 rows: the real-label roster reference (role_reference_2021_2025.parquet,
sourced from torpdata) does not extend to 2026, so EVERY 2026 row's role comes from
the lagged proxy classifier (Tier B in build_role_lagged.py's hierarchy), never the
real label. This is a documented limitation, not an error -- see docs/2026_DATA_VALIDATION.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_composite_indices, build_context_features, build_relative_features, build_teammate_features
from src.features.build_lagged_form_features import build as build_lagged_form
from src.features.build_role_lagged import build_lagged_inputs, train_classifier, MIN_PRIOR_GAMES
from src.features.build_role_proxy import season_player_profile

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

CORE_START, CORE_END_2026 = 2003, 2026
ADVANCED_START, ADVANCED_END_2026 = 2015, 2026


def _base_with_safe_features(core: pd.DataFrame) -> pd.DataFrame:
    relative = build_relative_features.build(core)
    context = build_context_features.build(core)
    teammate = build_teammate_features.build(core)
    composite = build_composite_indices.build(core)
    return pd.concat([core, relative, context, teammate, composite], axis=1)


def build_role_lagged_2026(core_full: pd.DataFrame) -> pd.DataFrame:
    role_ref = pd.read_parquet(PROCESSED_DIR / "role_reference_2021_2025.parquet")

    core_names = core_full[["season", "team_id", "player_id", "player_name"]].drop_duplicates()
    core_names["surname_key"] = core_names["player_name"].str.split().str[-1].str.lower().str.replace(r"[^a-z]", "", regex=True)
    role_ref_by_player = core_names.merge(role_ref, on=["season", "team_id", "surname_key"], how="inner")
    dup = role_ref_by_player.duplicated(subset=["season", "player_id"], keep=False)
    role_ref_by_player = role_ref_by_player[~dup][["season", "player_id", "position"]]

    profile_full_season = season_player_profile(core_full[core_full["season"] <= 2025])
    clf, z_cols, acc = train_classifier(profile_full_season, role_ref_by_player)
    print(f"role_lagged_2026 classifier held-out accuracy (merged labels): {acc:.3f}")

    lagged = build_lagged_inputs(core_full)
    eligible = lagged["n_prior_games"] >= MIN_PRIOR_GAMES
    valid_input = eligible & lagged[z_cols].notna().all(axis=1)

    lagged["role_lagged_proxy"] = "UNKNOWN"
    lagged.loc[valid_input, "role_lagged_proxy"] = clf.predict(lagged.loc[valid_input, z_cols].to_numpy())

    out = lagged[["season", "match_id", "player_id", "n_prior_games", "role_lagged_proxy"]].merge(
        role_ref_by_player, on=["season", "player_id"], how="left"
    )
    out["role"] = out["position"].fillna(out["role_lagged_proxy"])
    out["role_source"] = np.where(out["position"].notna(), "real_label",
                          np.where(out["role_lagged_proxy"] == "UNKNOWN", "unknown_insufficient_history", "lagged_proxy"))
    out = out[["season", "match_id", "player_id", "role", "role_source"]]

    n2026 = out[out["season"] == 2026]
    print(f"2026 role_source breakdown:\n{n2026['role_source'].value_counts()}")
    out.to_parquet(PROCESSED_DIR / "player_match_role_lagged_2026.parquet", index=False)
    return out


def build_core_2026_model(role_lagged: pd.DataFrame):
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2026.parquet")
    role_lagged = role_lagged[["match_id", "player_id", "role", "role_source"]]
    lagged_form = build_lagged_form(core)

    base = _base_with_safe_features(core)
    base = base.merge(role_lagged, on=["match_id", "player_id"], how="left")
    base = base.merge(
        lagged_form.drop(columns=["match_id", "player_id"]), left_index=True, right_index=True, how="left"
    )

    out = base[base["season"].between(CORE_START, CORE_END_2026)].copy()
    out.to_parquet(PROCESSED_DIR / "model_core_2026.parquet", index=False)
    print(f"model_core_2026.parquet: {len(out):,} rows, {out.shape[1]} cols, seasons {CORE_START}-{CORE_END_2026}")
    return out


def build_advanced_2026_model(role_lagged: pd.DataFrame):
    core_full = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2026.parquet")
    advanced = pd.read_parquet(PROCESSED_DIR / "player_match_advanced_2010_2026.parquet")
    lagged_form = build_lagged_form(core_full)

    base = _base_with_safe_features(core_full)
    adv_only_cols = [c for c in advanced.columns if c not in core_full.columns]
    base = base.merge(advanced[["match_id", "player_id"] + adv_only_cols], on=["match_id", "player_id"], how="left")
    base = base.merge(role_lagged[["match_id", "player_id", "role", "role_source"]], on=["match_id", "player_id"], how="left")
    base = base.merge(
        lagged_form.drop(columns=["match_id", "player_id"]), left_index=True, right_index=True, how="left"
    )

    out = base[base["season"].between(ADVANCED_START, ADVANCED_END_2026)].copy()
    out.to_parquet(PROCESSED_DIR / "model_advanced_2026.parquet", index=False)
    print(f"model_advanced_2026.parquet: {len(out):,} rows, {out.shape[1]} cols, seasons {ADVANCED_START}-{ADVANCED_END_2026}")
    return out


def main():
    core_full = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2026.parquet")
    role_lagged = build_role_lagged_2026(core_full)
    build_core_2026_model(role_lagged)
    build_advanced_2026_model(role_lagged)


if __name__ == "__main__":
    main()
