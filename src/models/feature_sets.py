"""
Phase 4: feature family definitions and derived-feature construction shared
across every experiment script. Centralised here so the ablation study
(docs/FEATURE_ABLATION.md) and the main backtest use IDENTICAL definitions.
"""
import numpy as np
import pandas as pd

RAW_STATS = [
    "disposals", "kicks", "handballs", "marks", "goals", "behinds", "hitouts", "tackles",
    "rebound_50s", "inside_50s", "clearances", "clangers", "frees_for", "frees_against",
    "contested_possessions", "uncontested_possessions", "contested_marks", "marks_inside_50",
    "one_percenters", "bounces", "goal_assists", "time_on_ground_pct",
]
RELATIVE_STATS_Z = [f"{s}_match_z" for s in
                     ["disposals", "contested_possessions", "clearances", "tackles",
                      "goals", "inside_50s", "contested_marks", "marks"]]
CONTEXT = ["is_win", "margin", "absolute_margin", "is_close_game", "is_blowout"]
TEAMMATE = ["n_teammates_disposals_ge_25", "n_teammates_disposals_ge_30",
            "n_teammates_goals_ge_2", "n_teammates_goals_ge_3", "team_disposal_concentration"]
LAGGED_FORM = [f"{s}_prev{w}_mean" for s in ["disposals", "contested_possessions", "clearances", "goals"]
               for w in (3, 5, 10)] + [f"{s}_season_to_date_mean" for s in
                                        ["disposals", "contested_possessions", "clearances", "goals"]]
REPUTATION = ["brownlow_votes_prev5_mean", "brownlow_votes_season_to_date_mean"]
NONLINEAR = ["disposals_over25", "disposals_over30", "goals_over3", "goals_over5"]
ADVANCED_STATS = ["effective_disposals", "disposal_efficiency_pct", "centre_clearances",
                  "stoppage_clearances", "score_involvements", "metres_gained", "turnovers",
                  "intercepts", "tackles_inside_50"]
ROLE_CATEGORIES = ["MIDFIELDER", "KEY_FORWARD", "MEDIUM_FORWARD", "RUCK", "KEY_DEFENDER",
                   "MEDIUM_DEFENDER", "HYBRID_MID_FWD"]  # UNKNOWN is the implicit reference category
ROLE_DUMMIES = [f"role_{r}" for r in ROLE_CATEGORIES]
WIN_MARGIN_INTERACTION = ["win_x_margin"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["disposals_over25"] = (df["disposals"] - 25).clip(lower=0)
    df["disposals_over30"] = (df["disposals"] - 30).clip(lower=0)
    df["goals_over3"] = (df["goals"] - 3).clip(lower=0)
    df["goals_over5"] = (df["goals"] - 5).clip(lower=0)
    df["win_x_margin"] = df["is_win"] * df["absolute_margin"]
    for r in ROLE_CATEGORIES:
        df[f"role_{r}"] = (df["role"] == r).astype(float)
    return df


def fillna_zero_for_counts(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Teammate-count features are legitimately 0 (not missing) when a team simply has
    no teammates above a threshold -- unlike the rest of this project's stats, a 0 here
    IS structurally meaningful, so filling is appropriate only for this specific family."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


FAMILIES = {
    "raw": RAW_STATS,
    "match_relative": RELATIVE_STATS_Z,
    "context": CONTEXT,
    "teammate": TEAMMATE,
    "role": ROLE_DUMMIES,
    "nonlinear": NONLINEAR,
    "lagged_form": LAGGED_FORM,
    "reputation": REPUTATION,
    "win_margin_interaction": WIN_MARGIN_INTERACTION,
}
ADVANCED_FAMILY = {"advanced_stats": ADVANCED_STATS}


def prepare_features(df: pd.DataFrame, feature_cols: list, dropna: bool = True) -> pd.DataFrame:
    df = add_derived_features(df)
    df = fillna_zero_for_counts(df, TEAMMATE)
    keep = list(dict.fromkeys(feature_cols))  # de-dup, preserve order
    if dropna:
        df = df.dropna(subset=[c for c in keep if c in df.columns])
    return df
