"""
2027 R&D: point-in-time player x match feature store.

Builds data/features/player_match_features.parquet (2003-2026) from the
already-validated Phase 3/4/5 model tables (model_core_2026 + the ADVANCED
extension) and attaches the 2026 actual votes as labels (data/actual/), then
adds the new 2027 feature families. Every feature carries an explicit
TIMING class in FEATURE_REGISTRY:

  same_match        -- a function of this match's own final box score / result
                       (what the umpires themselves see when voting; safe)
  prior_matches     -- strictly earlier matches' box-score stats (safe)
  prior_seasons     -- strictly earlier SEASONS' Brownlow votes (safe: revealed)
  same_season_unrevealed -- legacy reputation columns that average Brownlow
                       votes from earlier rounds of the SAME season. Those
                       votes are not revealed until count night, so at
                       forecast time they do not exist. Kept in the table for
                       backward compatibility, EXCLUDED from every 2027
                       candidate feature set (see docs/2027_MODEL_R&D_PLAN.md).

No feature reads brownlow_votes of the current match or of any later match.
Run:  python -m src.features.point_in_time
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import feature_sets as fs

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
ACTUAL = ROOT / "data" / "actual"
OUT_DIR = ROOT / "data" / "features"
OUT_PATH = OUT_DIR / "player_match_features.parquet"
REGISTRY_PATH = OUT_DIR / "feature_registry.json"
MANIFEST_PATH = OUT_DIR / "build_manifest.json"

BUILD_VERSION = "1.0.0"

# ---- stat lists ------------------------------------------------------------
BASE_REL_STATS = ["disposals", "contested_possessions", "clearances", "tackles", "goals", "inside_50s", "contested_marks", "marks"]
EXT_REL_STATS = ["kicks", "handballs", "hitouts", "rebound_50s", "marks_inside_50", "one_percenters", "goal_assists",
                 "uncontested_possessions", "clangers", "behinds", "frees_for", "frees_against", "time_on_ground_pct"]
ADV_STATS = ["effective_disposals", "disposal_efficiency_pct", "centre_clearances", "stoppage_clearances", "score_involvements",
             "metres_gained", "turnovers", "intercepts", "tackles_inside_50"]
IMPACT_COMPONENTS = ["disposals", "contested_possessions", "clearances", "tackles", "goals", "inside_50s", "contested_marks", "marks"]
ROLE_CATEGORIES = ["MIDFIELDER", "KEY_FORWARD", "MEDIUM_FORWARD", "RUCK", "KEY_DEFENDER", "MEDIUM_DEFENDER", "MIDFIELDER_FORWARD"]
BASELINE_STATS = ["disposals", "contested_possessions", "clearances", "goals"]


def _z(g: pd.Series) -> pd.Series:
    s = g.std(ddof=0)
    return (g - g.mean()) / (s if s > 1e-9 else 1.0)


def _match_pct(g: pd.Series) -> pd.Series:
    return g.rank(pct=True, method="average")


def _team_share(g: pd.Series) -> pd.Series:
    t = g.sum()
    return g / t if t > 0 else g * 0.0


def _gap_to_best_excl_self(df: pd.DataFrame, col: str, group_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """own value minus best / 2nd-best OTHER value within the group."""
    grp = df.groupby(group_cols)[col]
    top1 = grp.transform("max")
    # second max: nlargest(2) per group; take the 2nd; via sorted values
    def second(g):
        v = np.sort(g.to_numpy())[::-1]
        return pd.Series(v[1] if len(v) > 1 else v[0], index=g.index)
    top2 = grp.transform(second)
    own = df[col]
    best_other = np.where(own >= top1, top2, top1)
    # 2nd-best other: if own is the max -> 3rd value; if own is 2nd -> ... approximate with 2nd value excluding self
    def third(g):
        v = np.sort(g.to_numpy())[::-1]
        return pd.Series(v[2] if len(v) > 2 else v[-1], index=g.index)
    top3 = grp.transform(third)
    second_other = np.where(own >= top1, top3, np.where(own >= top2, top3, top2))
    return own - best_other, own - second_other


# ============================================================================
def load_base() -> pd.DataFrame:
    core = pd.read_parquet(PROCESSED / "model_core_2026.parquet")
    adv = pd.read_parquet(PROCESSED / "model_advanced_2026.parquet", columns=["match_id", "player_id"] + ADV_STATS + ["afl_fantasy_points", "supercoach_points"])
    core["player_id"] = core["player_id"].astype(str)
    adv["player_id"] = adv["player_id"].astype(str)
    df = core.merge(adv, on=["match_id", "player_id"], how="left")
    assert len(df) == len(core), "advanced join duplicated rows"
    return df


def attach_2026_labels(df: pd.DataFrame) -> pd.DataFrame:
    mv = pd.read_csv(ACTUAL / "2026_brownlow_match_votes.csv", dtype={"player_id": "string"})
    df = df.copy()
    is26 = df["season"] == 2026
    df.loc[is26, "brownlow_votes"] = 0.0
    df["label_source"] = np.where(is26, "afl_actual_2026", "afltables")
    key_id = mv[mv["player_id"].notna()]
    m = df.loc[is26, ["match_id", "player_id"]].reset_index().merge(key_id[["match_id", "player_id", "actual_brownlow_votes"]], on=["match_id", "player_id"])
    df.loc[m["index"], "brownlow_votes"] = m["actual_brownlow_votes"].to_numpy(dtype=float)
    # unresolved-id fallback: name + team is unique within a match
    key_nm = mv[mv["player_id"].isna()]
    if len(key_nm):
        m2 = df.loc[is26, ["match_id", "player_name", "team_id"]].reset_index().merge(
            key_nm[["match_id", "player_name", "player_team", "actual_brownlow_votes"]],
            left_on=["match_id", "player_name", "team_id"], right_on=["match_id", "player_name", "player_team"])
        df.loc[m2["index"], "brownlow_votes"] = m2["actual_brownlow_votes"].to_numpy(dtype=float)
        df.loc[m2["index"], "label_source"] = "afl_actual_2026_name_team_fallback"
    n_lab = int(len(m) + (len(m2) if len(key_nm) else 0))
    assert n_lab == len(mv), f"2026 labels attached {n_lab} != {len(mv)} actual vote rows"
    return df


def add_relative_ext(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = []
    for s in EXT_REL_STATS + ADV_STATS:
        if s not in df.columns:
            continue
        gm = df.groupby("match_id")[s]
        df[f"{s}_match_z"] = gm.transform(_z)
        df[f"{s}_match_pct"] = gm.transform(_match_pct)
        df[f"{s}_team_share"] = df.groupby(["match_id", "team_id"])[s].transform(_team_share)
        cols += [f"{s}_match_z", f"{s}_match_pct", f"{s}_team_share"]
    return df, cols


def add_dominance(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Vote-competition features from a fixed, un-fitted composite of the 8
    always-available match z-scores (never a model output, so no circularity)."""
    zc = [f"{s}_match_z" for s in IMPACT_COMPONENTS]
    df["impact_z"] = df[zc].mean(axis=1)
    gm = df.groupby("match_id")["impact_z"]
    df["impact_match_rank"] = gm.rank(ascending=False, method="min")
    df["impact_match_pct"] = gm.transform(_match_pct)
    top1 = gm.transform("max")
    def second(g):
        v = np.sort(g.to_numpy())[::-1]; return pd.Series(v[1] if len(v) > 1 else v[0], index=g.index)
    top2 = gm.transform(second)
    df["impact_gap_to_match_best"] = np.where(df["impact_z"] >= top1, df["impact_z"] - top2, df["impact_z"] - top1)
    df["match_standout_margin"] = top1 - top2
    df["is_match_top_impact"] = (df["impact_z"] >= top1).astype(float)
    g1, g2 = _gap_to_best_excl_self(df, "impact_z", ["match_id", "team_id"])
    df["impact_gap_best_teammate"] = g1
    df["impact_gap_2nd_teammate"] = g2
    gt = df.groupby(["match_id", "team_id"])["impact_z"]
    df["n_strong_teammates"] = gt.transform(lambda g: (g >= 1.0).sum()) - (df["impact_z"] >= 1.0).astype(int)
    df["n_strong_in_match"] = gm.transform(lambda g: (g >= 1.0).sum())
    pos = df["impact_z"].clip(lower=0)
    tot = pos.groupby([df["match_id"], df["team_id"]]).transform("sum")
    share = np.where(tot > 0, pos / tot.replace(0, np.nan), 0.0)
    df["_share_sq"] = np.square(share)
    df["teammate_impact_concentration"] = df.groupby(["match_id", "team_id"])["_share_sq"].transform("sum")
    df = df.drop(columns="_share_sq")
    cols = ["impact_z", "impact_match_rank", "impact_match_pct", "impact_gap_to_match_best", "match_standout_margin",
            "is_match_top_impact", "impact_gap_best_teammate", "impact_gap_2nd_teammate", "n_strong_teammates",
            "n_strong_in_match", "teammate_impact_concentration"]
    return df, cols


def add_role_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    dummies = []
    for r in ROLE_CATEGORIES:
        df[f"role_{r}"] = (df["role"] == r).astype(float); dummies.append(f"role_{r}")
    df["role_is_defender"] = df["role"].isin(["KEY_DEFENDER", "MEDIUM_DEFENDER"]).astype(float)
    df["role_is_forward"] = df["role"].isin(["KEY_FORWARD", "MEDIUM_FORWARD", "MIDFIELDER_FORWARD"]).astype(float)
    inter = {
        "def_x_rebound50_z": df["role_is_defender"] * df["rebound_50s_match_z"],
        "def_x_one_percenters_z": df["role_is_defender"] * df["one_percenters_match_z"],
        "def_x_marks_z": df["role_is_defender"] * df["marks_match_z"],
        "def_x_disposals_z": df["role_is_defender"] * df["disposals_match_z"],
        "ruck_x_hitouts_z": df["role_RUCK"] * df["hitouts_match_z"],
        "ruck_x_clearances_z": df["role_RUCK"] * df["clearances_match_z"],
        "fwd_x_goals_z": df["role_is_forward"] * df["goals_match_z"],
        "mid_x_clearances_z": df["role_MIDFIELDER"] * df["clearances_match_z"],
        "mid_x_contested_z": df["role_MIDFIELDER"] * df["contested_possessions_match_z"],
    }
    for k, v in inter.items():
        df[k] = v
    return df, dummies + ["role_is_defender", "role_is_forward"], list(inter)


def add_baseline_relative(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = []
    for s in BASELINE_STATS:
        df[f"{s}_vs_prev10"] = df[s] - df[f"{s}_prev10_mean"]
        df[f"{s}_vs_season_to_date"] = df[s] - df[f"{s}_season_to_date_mean"]
        cols += [f"{s}_vs_prev10", f"{s}_vs_season_to_date"]
    df["has_prior_form"] = df["disposals_prevN_count"].fillna(0).gt(0).astype(float)
    df["career_games_prior"] = df.groupby("player_id").cumcount()  # rows are date-sorted below
    cols += ["has_prior_form", "career_games_prior"]
    return df, cols


def add_team_strength(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Season-to-date and prior-season win % for team and opponent, strictly
    before this match (computed from match results already in the table)."""
    tm = df.drop_duplicates(["match_id", "team_id"])[["season", "date", "match_id", "team_id", "is_win", "is_draw"]].copy()
    tm["pts"] = tm["is_win"].fillna(0) + 0.5 * tm["is_draw"].fillna(0)
    tm = tm.sort_values(["team_id", "season", "date", "match_id"])
    g = tm.groupby(["team_id", "season"])
    tm["games_before"] = g.cumcount()
    tm["pts_before"] = g["pts"].cumsum() - tm["pts"]
    tm["team_std_win_pct"] = np.where(tm["games_before"] > 0, tm["pts_before"] / tm["games_before"].replace(0, np.nan), np.nan)
    season_final = tm.groupby(["team_id", "season"])["pts"].mean().rename("prev_season_win_pct").reset_index()
    season_final["season"] = season_final["season"] + 1
    tm = tm.merge(season_final, on=["team_id", "season"], how="left")
    keep = tm[["match_id", "team_id", "team_std_win_pct", "prev_season_win_pct", "games_before"]]
    df = df.merge(keep, on=["match_id", "team_id"], how="left")
    opp = keep.rename(columns={"team_id": "opponent_id", "team_std_win_pct": "opp_std_win_pct", "prev_season_win_pct": "opp_prev_season_win_pct"}).drop(columns="games_before")
    df = df.merge(opp, on=["match_id", "opponent_id"], how="left")
    df["team_strength_diff_std"] = df["team_std_win_pct"] - df["opp_std_win_pct"]
    df["team_strength_diff_prev"] = df["prev_season_win_pct"] - df["opp_prev_season_win_pct"]
    df["is_home"] = (df["home_away"].astype(str).str.lower().str.startswith("h")).astype(float)
    cols = ["team_std_win_pct", "opp_std_win_pct", "prev_season_win_pct", "opp_prev_season_win_pct",
            "team_strength_diff_std", "team_strength_diff_prev", "is_home"]
    return df, cols


def add_reputation_pit(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reputation from PRIOR SEASONS' revealed votes only."""
    ps = df.groupby(["player_id", "season"])["brownlow_votes"].agg(votes="sum", games="size").reset_index().sort_values(["player_id", "season"])
    ps["cum_votes"] = ps.groupby("player_id")["votes"].cumsum() - ps["votes"]
    ps["cum_games"] = ps.groupby("player_id")["games"].cumsum() - ps["games"]
    ps["prior_seasons_votes_per_game"] = np.where(ps["cum_games"] > 0, ps["cum_votes"] / ps["cum_games"].replace(0, np.nan), 0.0)
    ps["prior_seasons_total_votes"] = ps["cum_votes"]
    last = ps[["player_id", "season", "votes", "games"]].copy(); last["season"] = last["season"] + 1
    last = last.rename(columns={"votes": "last_season_votes", "games": "last_season_games"})
    ps = ps.merge(last, on=["player_id", "season"], how="left")
    ps["last_season_votes"] = ps["last_season_votes"].fillna(0.0)
    ps["last_season_votes_per_game"] = np.where(ps["last_season_games"].fillna(0) > 0, ps["last_season_votes"] / ps["last_season_games"].replace(0, np.nan), 0.0)
    ps["has_prior_season"] = (ps["cum_games"] > 0).astype(float)
    cols = ["prior_seasons_votes_per_game", "prior_seasons_total_votes", "last_season_votes", "last_season_votes_per_game", "has_prior_season"]
    df = df.merge(ps[["player_id", "season"] + cols], on=["player_id", "season"], how="left")
    return df, cols


def build() -> dict:
    df = load_base()
    df = attach_2026_labels(df)
    df = df.sort_values(["date", "match_id", "team_id", "player_id"]).reset_index(drop=True)
    df = fs.add_derived_features(df)  # nonlinear hinges, win_x_margin, legacy role dummies
    df, rel_ext = add_relative_ext(df)
    df, dom = add_dominance(df)
    df, role_cols, role_inter = add_role_features(df)
    df, base_rel = add_baseline_relative(df)
    df, team_str = add_team_strength(df)
    df, rep = add_reputation_pit(df)
    df["season_idx"] = df["season"] - 2003
    df["round_num"] = pd.to_numeric(df["round"], errors="coerce")

    registry = {
        "raw": {"timing": "same_match", "columns": fs.RAW_STATS},
        "match_relative": {"timing": "same_match", "columns": fs.RELATIVE_STATS_Z},
        "match_relative_ext": {"timing": "same_match", "columns": rel_ext},
        "team_relative": {"timing": "same_match", "columns": [f"{s}_{k}" for s in BASE_REL_STATS for k in ("team_rank", "team_share", "gap_best_team", "gap_2nd_team")]},
        "context": {"timing": "same_match", "columns": fs.CONTEXT + fs.WIN_MARGIN_INTERACTION},
        "teammate": {"timing": "same_match", "columns": fs.TEAMMATE},
        "dominance": {"timing": "same_match", "columns": dom},
        "nonlinear": {"timing": "same_match", "columns": fs.NONLINEAR},
        "advanced_stats": {"timing": "same_match", "columns": ADV_STATS, "first_season": 2015},
        "role": {"timing": "prior_matches", "columns": role_cols},
        "role_interactions": {"timing": "prior_matches", "columns": role_inter},
        "lagged_form": {"timing": "prior_matches", "columns": fs.LAGGED_FORM},
        "baseline_relative": {"timing": "prior_matches", "columns": base_rel},
        "team_strength": {"timing": "prior_matches", "columns": team_str},
        "reputation_pit": {"timing": "prior_seasons", "columns": rep},
        "reputation_legacy": {"timing": "same_season_unrevealed", "columns": fs.REPUTATION + ["brownlow_votes_prev3_mean", "brownlow_votes_prev10_mean"]},
        "era": {"timing": "same_match", "columns": ["season_idx", "round_num"]},
    }
    for fam, meta in registry.items():
        missing = [c for c in meta["columns"] if c not in df.columns]
        assert not missing, f"{fam}: missing {missing}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=1))
    manifest = {
        "build_version": BUILD_VERSION, "built_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)), "seasons": [int(df["season"].min()), int(df["season"].max())],
        "matches": int(df["match_id"].nunique()), "n_feature_columns": int(sum(len(m["columns"]) for m in registry.values())),
        "label_sources": df["label_source"].value_counts().to_dict(),
        "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in
                         [PROCESSED / "model_core_2026.parquet", PROCESSED / "model_advanced_2026.parquet", ACTUAL / "2026_brownlow_match_votes.csv"]},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    m = build()
    print(json.dumps(m, indent=1))
    sys.exit(0)
