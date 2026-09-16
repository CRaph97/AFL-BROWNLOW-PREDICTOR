"""
Phase 5, sections 3/4: train the four 2026 forecasting scenarios plus the
reputation on/off comparison, and produce per-match/per-player probability
predictions for the entire real 2026 home-and-away season.

Architecture choice for every scenario: Plackett-Luce (Model 1), the model
that won on every tracked metric in Phase 4 (docs/MODEL_BACKTEST.md). Only the
TRAINING WINDOW and FEATURE SET vary across scenarios -- no new architecture is
introduced for the 2026 production forecast, per the instruction to avoid
inventing untested machinery this late.

Scenario A -- HISTORICAL BEHAVIOUR: CORE features, recent-8-season window
  (2018-2025), the single best-validated Phase 4 configuration. Represents
  "2026 votes like recent past seasons."
Scenario B -- RECENT ERA: CORE features, a narrower window (chosen from
  run_window_comparison_2026.py's result -- see docs/2026_MODELLING_METHODOLOGY.md
  for which window was selected and why).
Scenario C -- 2026 STATS-ASSISTED: ADVANCED features (CORE + footywire's
  extended stats: score_involvements, metres_gained, effective_disposals,
  disposal_efficiency_pct, centre/stoppage clearances, intercepts (proxy),
  turnovers, tackles_inside_50), recent-8-season window on the ADVANCED table
  (2018-2025). This is the closest buildable approximation to "umpire-visible
  objective stats have more influence" -- see docs/2026_STRUCTURAL_BREAK.md for
  exactly which of the 17 confirmed umpire stats this can and cannot cover.
Scenario A_rep -- Scenario A's features plus the reputation family
  (brownlow_votes_prev5_mean, brownlow_votes_season_to_date_mean), for the
  with/without reputation comparison (section 4) -- NOT part of the default
  ensemble.

All four are trained ONLY on seasons <= 2025 and used to predict every 2026
home-and-away match -- no 2026 data is used in training (there is no 2026
target to train on regardless).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.plackett_luce import PlackettLuceModel
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TRAIN_WINDOW_A = 8   # Phase 4 validated best (docs/MODEL_BACKTEST.md)
TRAIN_WINDOW_B = 3   # default recent-era choice; overridden below if window_comparison_2026 says otherwise
TRAIN_WINDOW_C = 8

CORE_FEATURES = (
    feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]
    + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]
    + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]
    + feature_sets.FAMILIES["lagged_form"] + feature_sets.FAMILIES["win_margin_interaction"]
)
ADVANCED_FEATURES = CORE_FEATURES + feature_sets.ADVANCED_FAMILY["advanced_stats"]
REPUTATION_FEATURES = CORE_FEATURES + feature_sets.FAMILIES["reputation"]


def _select_window_b() -> int:
    """Pick Scenario B's window from run_window_comparison_2026.py's summary if available:
    the narrowest window whose mean_correct_3 is within 0.5pp of the best (a materially
    "more recent" choice is preferred over the single best-by-a-hair result, since Scenario
    B's purpose is specifically to show the recent-era-only alternative)."""
    path = REPORTS_DIR / "window_comparison_2026_summary.csv"
    if not path.exists():
        print(f"WARNING: {path} not found -- using default TRAIN_WINDOW_B={TRAIN_WINDOW_B}")
        return TRAIN_WINDOW_B
    summary = pd.read_csv(path)
    numeric_windows = summary[summary["window"].isin(["recent3", "recent5", "recent8"])].copy()
    if numeric_windows.empty:
        return TRAIN_WINDOW_B
    best_c3 = numeric_windows["mean_correct_3"].max()
    close_enough = numeric_windows[numeric_windows["mean_correct_3"] >= best_c3 - 0.005]
    close_enough["n"] = close_enough["window"].str.replace("recent", "").astype(int)
    chosen = int(close_enough["n"].min())
    print(f"Scenario B window selected from window_comparison_2026: recent{chosen} "
          f"(best mean_correct_3={best_c3:.4f}, chosen within 0.5pp)")
    return chosen


def _fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, features: list, scenario: str) -> pd.DataFrame:
    if len(test) == 0:
        raise ValueError(f"Scenario {scenario}: empty 2026 test frame after feature-completeness filtering -- "
                          f"cannot predict. Check the relevant feature family for a season-boundary NaN-propagation "
                          f"issue (see _freeze_reputation_for_2026 for a real example of this failure mode).")
    model = PlackettLuceModel(feature_names=features).fit(train)
    preds = model.predict(test)
    preds = preds.copy()
    preds["scenario"] = scenario
    # standardised (within-match z-score) utility, needed for cross-scenario blending later --
    # PL utility is only meaningful up to an additive-per-match constant and an arbitrary overall
    # scale, so blending raw utilities across differently-scaled models would be unsound.
    X_raw = preds[features].to_numpy(dtype=float)
    u = model._transform(X_raw) @ model.beta
    preds["utility_raw"] = u
    g = preds.groupby("match_id")["utility_raw"]
    match_mean = g.transform("mean")
    match_std = g.transform("std").replace(0, np.nan)
    preds["utility_z"] = ((preds["utility_raw"] - match_mean) / match_std).fillna(0.0)
    keep = ["scenario", "season", "round", "match_id", "player_id", "player_name", "team_id",
            "brownlow_votes", "p3", "p2", "p1", "p0", "expected_votes", "utility_raw", "utility_z"]
    return preds[[c for c in keep if c in preds.columns]]


def _carry_forward_season_to_date_2026(core: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Phase 5 audit fix (2026-09-17): generalises `_freeze_reputation_for_2026`'s carry-forward
    logic to every `lagged_form` season-to-date feature. A player's FIRST 2026 match has no
    2026-season history yet, so `<stat>_season_to_date_mean` (an expanding mean that resets each
    season) is genuinely NaN by construction for every player in that match -- for 6 of 207 2026
    matches (5 of the season's 10 Round-1 games, played by only 10 of the 18 teams under 2026's
    split-opening-round format, plus the Round-2 match between the two teams that had the Round-1
    bye) EVERY player hits this simultaneously, so a per-row dropna silently voided the entire
    match rather than just thinning its roster -- the bug this audit was asked to fix.

    Fix (fallback priority 1 of the audit brief -- "prior-season lagged form where available"):
    for season==2026 rows only, carry forward each player's LAST REAL (pre-2026) value via a
    per-player, time-sorted forward-fill. This is the exact same, already-validated technique
    used for the reputation family (`_freeze_reputation_for_2026`) applied to a different column
    set. It changes the 2026-only interpretation from "this player's mean so far in 2026"
    (unknowable before a game is played) to "this player's mean as of the end of their last
    completed AFL season" -- a defensible, disclosed proxy, not a fabricated value. It can never
    use a future 2026 match: ffill only propagates a season<=2025 value forward in time, and
    pre-2026 rows are left byte-for-byte unchanged (the fill is written back only where
    season == 2026).

    For a true debutant (zero prior AFL games of any kind -- 11 of 276 rows across the 6 affected
    matches), no prior value exists to carry forward, so the column stays NaN and that individual
    PLAYER row (not the whole match) is excluded downstream by the existing per-row dropna, exactly
    matching the pre-existing, documented policy for insufficient-history players
    (docs/2026_DATA_VALIDATION.md section 7) -- fallback priority 4 ("reduced feature model for
    matches with insufficient history"), applied here at the single-player granularity it was
    always intended for."""
    cols = [c for c in cols if c in core.columns]
    if not cols:
        return core
    core = core.sort_values(["player_id", "date"]).copy()
    mask_2026 = core["season"] == 2026
    before_na = core.loc[mask_2026, cols].isna().any(axis=1)
    filled = core.groupby("player_id")[cols].ffill()
    for c in cols:
        core.loc[mask_2026, c] = filled.loc[mask_2026, c]
    after_na = core.loc[mask_2026, cols].isna().any(axis=1)
    flag = pd.Series(False, index=core.index)
    flag.loc[mask_2026] = (before_na & ~after_na).to_numpy()
    core["season_to_date_carried_forward_2026"] = flag
    n_carried = int(flag.sum())
    n_still_missing = int((mask_2026 & core[cols].isna().any(axis=1)).sum())
    print(f"season_to_date carry-forward (2026): {n_carried} player-match rows carried forward from "
          f"their last completed season; {n_still_missing} true-debutant rows still missing "
          f"(no prior AFL history at all) and will be excluded per-row as before.")
    return core


def _freeze_reputation_for_2026(core: pd.DataFrame) -> pd.DataFrame:
    """CRITICAL 2026-specific fix: brownlow_votes_prev5_mean/prev10_mean/prev3_mean and
    brownlow_votes_season_to_date_mean are built (build_lagged_form_features.py) from rolling/
    expanding windows over the player's OWN brownlow_votes history. Since 2026 votes are entirely
    unrevealed (set to NaN in build_2026_extension.py), any window that comes to include even one
    2026 game inherits NaN and STAYS NaN for the rest of the season -- this is not a computation
    bug, it correctly reflects that "how many votes has this player received so far this season"
    is genuinely unknowable in real time (Brownlow votes are revealed once, after the whole season,
    never round-by-round). Verified: without this fix, EVERY 2026 row has NaN in these columns by
    round 2, and 0 rows survive a dropna filter.

    Fix: for 2026 rows only, freeze each player's reputation features at their last known REAL
    (pre-2026) value via a per-player forward-fill, applied so it can only ever propagate a
    genuine historical number into 2026 -- pre-2026 rows are left byte-for-byte unchanged (the
    forward-fill result is written back only where season == 2026), so Phase 4's validated
    training-time values are untouched. This changes the 2026-specific interpretation of
    `..._season_to_date_mean` from "votes so far this season" (impossible to know) to "voting rate
    as of the end of the player's last completed season" -- documented in
    docs/2026_MODELLING_METHODOLOGY.md."""
    cols = [c for c in feature_sets.FAMILIES["reputation"] if c in core.columns]
    if not cols:
        return core
    core = core.sort_values(["player_id", "date"]).copy()
    filled = core.groupby("player_id")[cols].ffill()
    mask_2026 = core["season"] == 2026
    for c in cols:
        core.loc[mask_2026, c] = filled.loc[mask_2026, c]
    return core


def run():
    season_to_date_cols = [c for c in feature_sets.FAMILIES["lagged_form"] if c.endswith("_season_to_date_mean")]

    core = pd.read_parquet(PROCESSED_DIR / "model_core_2026.parquet")
    core = feature_sets.prepare_features(core, CORE_FEATURES, dropna=False)
    core = _carry_forward_season_to_date_2026(core, season_to_date_cols)
    core = _freeze_reputation_for_2026(core)
    core_train_pool = core[core["season"] <= 2025].dropna(subset=[c for c in CORE_FEATURES if c in core.columns])
    core_2026 = core[core["season"] == 2026].copy()
    # 2026 prediction rows must not be dropped for missing lagged-form/role history the way
    # training rows are -- see docs/2026_DATA_VALIDATION.md for how NaNs are handled at inference.
    n_2026_before = len(core_2026)
    core_2026_complete = core_2026.dropna(subset=[c for c in CORE_FEATURES if c in core_2026.columns])
    print(f"2026 CORE prediction rows: {n_2026_before:,} total, {len(core_2026_complete):,} with complete features "
          f"({n_2026_before - len(core_2026_complete)} dropped for missing features, e.g. debutants with <3 games "
          f"of history for lagged-form/role features)")

    adv = pd.read_parquet(PROCESSED_DIR / "model_advanced_2026.parquet")
    adv = feature_sets.prepare_features(adv, ADVANCED_FEATURES, dropna=False)
    adv = _carry_forward_season_to_date_2026(adv, season_to_date_cols)
    adv_train_pool = adv[adv["season"] <= 2025].dropna(subset=[c for c in ADVANCED_FEATURES if c in adv.columns])
    adv_2026 = adv[adv["season"] == 2026].copy()
    adv_2026_complete = adv_2026.dropna(subset=[c for c in ADVANCED_FEATURES if c in adv_2026.columns])
    print(f"2026 ADVANCED prediction rows: {len(adv_2026):,} total, {len(adv_2026_complete):,} with complete features")

    all_seasons = sorted(core_train_pool["season"].unique())

    window_b = _select_window_b()

    train_A = core_train_pool[core_train_pool["season"].isin(all_seasons[-TRAIN_WINDOW_A:])]
    train_B = core_train_pool[core_train_pool["season"].isin(all_seasons[-window_b:])]
    adv_seasons = sorted(adv_train_pool["season"].unique())
    train_C = adv_train_pool[adv_train_pool["season"].isin(adv_seasons[-TRAIN_WINDOW_C:])]

    print(f"Scenario A: training on seasons {sorted(train_A['season'].unique())}")
    preds_A = _fit_and_predict(train_A, core_2026_complete, CORE_FEATURES, "A_historical")

    print(f"Scenario B: training on seasons {sorted(train_B['season'].unique())}")
    preds_B = _fit_and_predict(train_B, core_2026_complete, CORE_FEATURES, "B_recent_era")

    print(f"Scenario C: training on ADVANCED seasons {sorted(train_C['season'].unique())}")
    preds_C = _fit_and_predict(train_C, adv_2026_complete, ADVANCED_FEATURES, "C_stats_assisted")

    train_A_rep = core_train_pool[core_train_pool["season"].isin(all_seasons[-TRAIN_WINDOW_A:])].dropna(
        subset=[c for c in REPUTATION_FEATURES if c in core_train_pool.columns])
    core_2026_rep_complete = core_2026.dropna(subset=[c for c in REPUTATION_FEATURES if c in core_2026.columns])
    print(f"Scenario A_rep: training on seasons {sorted(train_A_rep['season'].unique())}, "
          f"{len(core_2026_rep_complete):,} 2026 rows with reputation history")
    preds_A_rep = _fit_and_predict(train_A_rep, core_2026_rep_complete, REPUTATION_FEATURES, "A_with_reputation")

    all_preds = pd.concat([preds_A, preds_B, preds_C, preds_A_rep], ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_preds.to_parquet(PROCESSED_DIR / "scenario_predictions_2026.parquet", index=False)
    print(f"\nWrote {len(all_preds):,} scenario prediction rows -> "
          f"data/processed/scenario_predictions_2026.parquet")
    for s, g in all_preds.groupby("scenario"):
        print(f"  {s}: {g['match_id'].nunique()} matches, {len(g):,} player-match rows")
    return all_preds


if __name__ == "__main__":
    run()
