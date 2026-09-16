"""
Phase 3, section H (continued): a statistically-derived role proxy for seasons
outside the 2021-2025 window where the AFL's real listed position is available
(see build_role_reference.py).

Method (deliberately simple and inspectable, not a black box):
1. Aggregate each player's per-game stat averages within a season (players
   with < 5 games in a season are excluded -- too little signal to classify).
2. Z-score every input stat WITHIN its season, so the proxy is era-relative
   (a "high" clearance rate means "high relative to that season's players",
   not an absolute threshold that would drift as the game changes over time).
3. Train a shallow decision tree (max_depth=4, so it stays human-readable) on
   the 2021-2025 seasons where we have real official positions as labels.
4. Evaluate with grouped cross-validation (grouped by player, so the same
   player's multiple seasons never leak between train and test folds) and
   report a confusion matrix -- this number is what should drive how much
   trust to place in the pre-2021 proxy labels, not an assumption.
5. Apply the fitted tree to every other season (1984-2020, 2026) to produce a
   proxy label. These are clearly flagged `is_proxy_position = True` wherever
   used, vs. `False` for the 2021-2025 real labels.

Input stats used (all available across the full 1984-2025 CORE window from
1999 onward -- see docs/DATA_COVERAGE.md; seasons before 1999 lack several of
these and are excluded from role classification entirely, not guessed at):
hitouts, marks_inside_50, goals, rebound_50s, one_percenters, inside_50s,
clearances, contested_possessions, disposals, uncontested_possessions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

STAT_INPUTS = [
    "hitouts", "marks_inside_50", "goals", "rebound_50s", "one_percenters",
    "inside_50s", "clearances", "contested_possessions", "disposals", "uncontested_possessions",
]
MIN_GAMES = 5
ROLE_DATA_START_SEASON = 1999  # first season all STAT_INPUTS are structurally available -- see docs/DATA_COVERAGE.md


def season_player_profile(core: pd.DataFrame) -> pd.DataFrame:
    df = core[core["season"] >= ROLE_DATA_START_SEASON]
    agg = df.groupby(["season", "player_id"]).agg(
        games=("match_id", "nunique"),
        **{f"{s}_pg": (s, "mean") for s in STAT_INPUTS},
    ).reset_index()
    agg = agg[agg["games"] >= MIN_GAMES].copy()

    # z-score within season
    for s in STAT_INPUTS:
        col = f"{s}_pg"
        mean = agg.groupby("season")[col].transform("mean")
        std = agg.groupby("season")[col].transform("std").replace(0, np.nan)
        agg[f"{s}_z"] = (agg[col] - mean) / std
    return agg


def fit_and_evaluate(profile: pd.DataFrame, role_ref_by_player: pd.DataFrame):
    z_cols = [f"{s}_z" for s in STAT_INPUTS]
    labelled = profile.merge(role_ref_by_player, on=["season", "player_id"], how="inner").dropna(subset=z_cols)

    X = labelled[z_cols].to_numpy()
    y = labelled["position"].to_numpy()
    groups = labelled["player_id"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    preds = np.empty_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])

    accuracy = (preds == y).mean()
    from sklearn.metrics import confusion_matrix, classification_report
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y, preds, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    report = classification_report(y, preds, labels=labels_sorted, zero_division=0)

    # final tree fit on ALL labelled data, used for applying to unlabelled seasons
    final_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)
    final_clf.fit(X, y)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(REPORTS_DIR / "role_proxy_confusion_matrix.csv")
    with open(REPORTS_DIR / "role_proxy_classification_report.txt", "w") as f:
        f.write(f"Cross-validated (grouped by player) overall accuracy: {accuracy:.3f}\n\n")
        f.write(report)
        f.write("\n\nDecision tree structure (max_depth=4):\n")
        f.write(export_text(final_clf, feature_names=z_cols))

    print(f"Grouped CV accuracy: {accuracy:.3f}")
    print(cm_df)
    return final_clf, z_cols, accuracy


def apply_proxy(profile: pd.DataFrame, clf, z_cols: list) -> pd.DataFrame:
    out = profile[["season", "player_id"]].copy()
    valid = profile[z_cols].notna().all(axis=1)
    out["role_proxy"] = pd.NA
    out.loc[valid, "role_proxy"] = clf.predict(profile.loc[valid, z_cols].to_numpy())
    return out


def build():
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    role_ref = pd.read_parquet(PROCESSED_DIR / "role_reference_2021_2025.parquet")

    # role_ref is keyed by (season, team_id, surname_key); map to player_id via the CORE table's own
    # (season, team_id, surname) so both sides use exactly the same surname-normalisation logic.
    core_names = core[["season", "team_id", "player_id", "player_name"]].drop_duplicates()
    core_names["surname_key"] = core_names["player_name"].str.split().str[-1].str.lower().str.replace(r"[^a-z]", "", regex=True)
    role_ref_by_player = core_names.merge(role_ref, on=["season", "team_id", "surname_key"], how="inner")
    dup = role_ref_by_player.duplicated(subset=["season", "player_id"], keep=False)
    role_ref_by_player = role_ref_by_player[~dup][["season", "player_id", "position"]]

    profile = season_player_profile(core)
    clf, z_cols, accuracy = fit_and_evaluate(profile, role_ref_by_player)
    proxy = apply_proxy(profile, clf, z_cols)

    # final table: real label where available (2021-2025), proxy label otherwise
    final = proxy.merge(role_ref_by_player, on=["season", "player_id"], how="left")
    final["is_proxy_position"] = final["position"].isna()
    final["role"] = final["position"].fillna(final["role_proxy"])
    final = final[["season", "player_id", "role", "is_proxy_position"]]

    final.to_parquet(PROCESSED_DIR / "player_season_role.parquet", index=False)
    print(f"\nWrote {len(final):,} player-season role rows -> data/processed/player_season_role.parquet")
    print(f"Real (non-proxy) rows: {(~final['is_proxy_position']).sum():,}; proxy rows: {final['is_proxy_position'].sum():,}")
    return final, accuracy


if __name__ == "__main__":
    build()
