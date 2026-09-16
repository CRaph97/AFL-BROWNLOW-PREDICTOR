"""
Phase 4, section A1/A2: fixes the role-leakage issue identified in
docs/LEAKAGE_AUDIT.md #5 (Phase 3's role feature used full-season averages,
which for a match in round 5 would use information from rounds 6-23 -- not
available at prediction time).

Hierarchy applied (per the Phase 4 brief):
  A. 2021-2025: the AFL's officially-listed position (torp/torpdata) is used
     as-is. DOCUMENTED ASSUMPTION: a club's listed position for a player is a
     roster designation set independently of that specific match's outcome,
     not derived from in-season performance data, so it is treated as
     legitimately known before every match that player plays in that season.
     This is a real assumption, not a proven fact (positions can in principle
     be updated mid-season) -- stated plainly rather than silently relied on.
  B. 1999-2020: role is inferred from a LAGGED, PRIOR-GAMES-ONLY rolling
     average of the same 10 box-score stats used in Phase 3's proxy
     (build_role_proxy.py), excluding the current match entirely. Requires
     >= 3 prior games in that season; below that, or in each player's first
     3 games of a season, the role is "UNKNOWN" rather than guessed from
     nothing or filled from future games.
  C. The classifier itself is RETRAINED (not reused from Phase 3) with the
     unreliable MIDFIELDER_FORWARD training label merged into a broader
     "HYBRID_MID_FWD" category, per the Phase 4 instruction not to fabricate
     distinctions that cannot be reliably recovered. This changes what the
     classifier CAN predict, not just how its output is post-processed.

Z-scoring for the lagged (prior-games) inputs uses each season's FULL-SEASON
population mean/std as the normalisation reference (not a point-in-time
cross-sectional recompute at every round, which would be materially more
expensive) -- an explicit, documented approximation: an early-season running
average is compared against the season's overall spread. This does not leak
any INDIVIDUAL player's future games into their OWN feature value (the
leakage this fix targets), it only uses the general population's full-season
spread as a scaling reference, which is a much weaker and, in this project's
judgement, acceptable form of look-ahead for a normalisation constant.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

STAT_INPUTS = [
    "hitouts", "marks_inside_50", "goals", "rebound_50s", "one_percenters",
    "inside_50s", "clearances", "contested_possessions", "disposals", "uncontested_possessions",
]
MIN_PRIOR_GAMES = 3
ROLE_DATA_START_SEASON = 1999
MERGE_LABEL = {"MIDFIELDER_FORWARD": "HYBRID_MID_FWD"}


def build_lagged_inputs(core: pd.DataFrame) -> pd.DataFrame:
    """One row per player-match (season >= 1999): prior-games-only rolling mean
    of each STAT_INPUTS stat (excluding the current match), the count of prior
    games available, and season-population z-scores of that rolling mean."""
    df = core[core["season"] >= ROLE_DATA_START_SEASON].sort_values(["season", "player_id", "date"]).copy()

    g = df.groupby(["season", "player_id"])
    for s in STAT_INPUTS:
        # expanding mean of all PRIOR games: shift(1) before expanding excludes the current row
        df[f"{s}_prior_mean"] = g[s].transform(lambda x: x.shift(1).expanding().mean())
    df["n_prior_games"] = g.cumcount()  # 0 for a player's first game of the season, etc.

    # season-level population mean/std of the RAW per-game stat (full season, used only as a
    # normalisation reference -- see module docstring)
    for s in STAT_INPUTS:
        pop_mean = df.groupby("season")[s].transform("mean")
        pop_std = df.groupby("season")[s].transform("std").replace(0, np.nan)
        df[f"{s}_z"] = (df[f"{s}_prior_mean"] - pop_mean) / pop_std

    return df


def train_classifier(profile_full_season: pd.DataFrame, role_ref_by_player: pd.DataFrame):
    """Retrain the role classifier (Phase 4 version) with MIDFIELDER_FORWARD merged
    into HYBRID_MID_FWD, using the SAME full-season z-scored features as Phase 3
    (this training step itself is not per-match leakage -- it is a one-time fit
    used to learn the general statistical signature of each role, analogous to
    fitting any model on historical data)."""
    z_cols = [f"{s}_z" for s in STAT_INPUTS]
    labelled = profile_full_season.merge(role_ref_by_player, on=["season", "player_id"], how="inner").dropna(subset=z_cols)
    labelled["position_merged"] = labelled["position"].replace(MERGE_LABEL)

    X = labelled[z_cols].to_numpy()
    y_orig = labelled["position"].to_numpy()
    y_merged = labelled["position_merged"].to_numpy()
    groups = labelled["player_id"].to_numpy()

    def cv_accuracy(y):
        gkf = GroupKFold(n_splits=5)
        preds = np.empty_like(y)
        for train_idx, test_idx in gkf.split(X, y, groups):
            clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)
            clf.fit(X[train_idx], y[train_idx])
            preds[test_idx] = clf.predict(X[test_idx])
        return (preds == y).mean(), preds

    acc_orig, preds_orig = cv_accuracy(y_orig)
    acc_merged, preds_merged = cv_accuracy(y_merged)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "role_merge_impact.txt", "w") as f:
        f.write(f"n_MIDFIELDER_FORWARD affected rows (training/eval set): {(y_orig == 'MIDFIELDER_FORWARD').sum()} "
                f"of {len(y_orig)} ({(y_orig == 'MIDFIELDER_FORWARD').mean():.2%})\n\n")
        f.write(f"7-class (original) grouped CV accuracy:        {acc_orig:.4f}\n")
        f.write(f"6-class (MIDFIELDER_FORWARD merged) grouped CV accuracy: {acc_merged:.4f}\n\n")
        f.write("Per-class report, 7-class original:\n")
        f.write(classification_report(y_orig, preds_orig, zero_division=0))
        f.write("\nPer-class report, 6-class merged:\n")
        f.write(classification_report(y_merged, preds_merged, zero_division=0))

    print(f"n_MIDFIELDER_FORWARD affected: {(y_orig == 'MIDFIELDER_FORWARD').sum()} / {len(y_orig)}")
    print(f"7-class accuracy: {acc_orig:.4f} | 6-class (merged) accuracy: {acc_merged:.4f}")

    final_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)
    final_clf.fit(X, y_merged)
    with open(REPORTS_DIR / "role_lagged_tree.txt", "w") as f:
        f.write(export_text(final_clf, feature_names=z_cols))
    return final_clf, z_cols, acc_merged


def build():
    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    role_ref = pd.read_parquet(PROCESSED_DIR / "role_reference_2021_2025.parquet")

    core_names = core[["season", "team_id", "player_id", "player_name"]].drop_duplicates()
    core_names["surname_key"] = core_names["player_name"].str.split().str[-1].str.lower().str.replace(r"[^a-z]", "", regex=True)
    role_ref_by_player = core_names.merge(role_ref, on=["season", "team_id", "surname_key"], how="inner")
    dup = role_ref_by_player.duplicated(subset=["season", "player_id"], keep=False)
    role_ref_by_player = role_ref_by_player[~dup][["season", "player_id", "position"]]

    # full-season profile (for TRAINING the classifier only, per the module docstring)
    from src.features.build_role_proxy import season_player_profile
    profile_full_season = season_player_profile(core)
    clf, z_cols, acc = train_classifier(profile_full_season, role_ref_by_player)

    # lagged, prior-games-only inputs (for APPLYING the classifier to every match, 1999-2020)
    lagged = build_lagged_inputs(core)
    eligible = lagged["n_prior_games"] >= MIN_PRIOR_GAMES
    z_cols_lagged = z_cols  # same column names, now populated from prior-games data
    valid_input = eligible & lagged[z_cols_lagged].notna().all(axis=1)

    lagged["role_lagged_proxy"] = "UNKNOWN"
    lagged.loc[valid_input, "role_lagged_proxy"] = clf.predict(lagged.loc[valid_input, z_cols_lagged].to_numpy())

    # attach real label for 2021-2025 (Tier A), overriding the proxy for those rows
    out = lagged[["season", "match_id", "player_id", "n_prior_games", "role_lagged_proxy"]].merge(
        role_ref_by_player, on=["season", "player_id"], how="left"
    )
    out["role"] = out["position"].fillna(out["role_lagged_proxy"])
    out["role_source"] = np.where(out["position"].notna(), "real_label",
                          np.where(out["role_lagged_proxy"] == "UNKNOWN", "unknown_insufficient_history", "lagged_proxy"))
    out = out[["season", "match_id", "player_id", "role", "role_source"]]

    n_unknown = (out["role_source"] == "unknown_insufficient_history").sum()
    print(f"\nRole source breakdown:\n{out['role_source'].value_counts()}")
    print(f"({n_unknown:,} rows / {len(out):,} = {n_unknown/len(out):.1%} UNKNOWN due to insufficient prior-game history)")

    out.to_parquet(PROCESSED_DIR / "player_match_role_lagged.parquet", index=False)
    print(f"\nWrote {len(out):,} rows -> data/processed/player_match_role_lagged.parquet")
    return out, acc


if __name__ == "__main__":
    build()
