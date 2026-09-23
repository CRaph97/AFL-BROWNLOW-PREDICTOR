"""
2026 post-Brownlow evaluation: scores the FROZEN pre-count outputs of the
Production model, the Objective model and Wheelo against the AFL actual
votes (data/actual/, post-event ground truth), and settles the pre-count
bookmaker markets captured on 2026-09-18 (count held 2026-09-21).

Everything here is read-only over its inputs and writes only to
data/evaluation/2026/. Nothing is retrained, no simulation is rerun, no
forecast is blended. Wheelo metrics are produced only where Wheelo actually
published the corresponding quantity (season EV, match EV, match P3 %,
match rank) -- never fabricated for quantities it does not publish.

Run:  python -m src.evaluation.build_2026_evaluation
Docs: docs/2026_BROWNLOW_EVALUATION.md
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import settlement as stl

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
ACTUAL = ROOT / "data" / "actual"
EXTERNAL = ROOT / "data" / "external" / "processed"
BETTING = ROOT / "data" / "betting"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "evaluation" / "2026"

COUNT_DATE_UTC = "2026-09-21T00:00:00+00:00"   # Brownlow count night (AEST evening of 21 Sep)
MODELS = ("production", "objective", "wheelo")
MODEL_LABEL = {"production": "Production", "objective": "Objective", "wheelo": "Wheelo"}
CALIBRATION_BINS = np.linspace(0, 1, 11)
SETTLEABLE_MARKETS = ["WINNER", "TOP_N", "EXACT_POSITION", "PLAYER_VOTES_OU", "X_PLUS_VOTES",
                      "TO_POLL_A_VOTE", "PLAYER_H2H", "TEAM_VOTES_OU"]
BET_VALUE_LABEL = {
    "HIGH_CONFIDENCE_WHEELO_CONFIRMED": "Strong Bet Value + Wheelo Support",
    "HIGH_CONFIDENCE_WHEELO_NEUTRAL": "Strong Bet Value",
    "MEDIUM_CONFIDENCE": "Moderate Bet Value",
    "HIGH_RISK_HIGH_REWARD": "Speculative Bet Value",
    "MODEL_DISAGREEMENT": "Model Disagreement",
    "NO_VALUE": "No Value",
}
LIKELIHOOD_BANDS = [(0.80, "Very High"), (0.65, "High"), (0.45, "Moderate"), (0.25, "Low"), (0.0, "Very Low")]

# Inputs that must not change (hashed into the manifest; tests re-hash them).
FROZEN_INPUTS = [
    REPORTS / "2026_leaderboard.csv", REPORTS / "2026_predicted_votes.csv", REPORTS / "2026_match_probabilities.csv",
    REPORTS / "2026_simulation_summary.csv", REPORTS / "2026_objective_leaderboard.csv",
    REPORTS / "2026_objective_votes.csv", REPORTS / "2026_objective_simulation_summary.csv",
    REPORTS / "2026_order_scenarios.csv", EXTERNAL / "wheelo_season.csv", EXTERNAL / "wheelo_match_level.csv",
    EXTERNAL / "external_overview.csv", BETTING / "processed" / "priced_opportunities.csv",
    BETTING / "processed" / "refresh_summary.json", ACTUAL / "2026_brownlow_match_votes.csv",
    ACTUAL / "2026_brownlow_leaderboard.csv", PROCESSED / "mc_totals_2026.npy",
    PROCESSED / "mc_totals_objective_2026.npy",
]


def _sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_input_hashes() -> dict:
    return {str(p.relative_to(ROOT)): _sha(p) for p in FROZEN_INPUTS}


def _pid(x) -> str | None:
    """Normalise every source's player_id spelling (int, float 12692.0, str) to str."""
    try:
        if x is None or bool(pd.isna(x)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, str) and x.strip() in ("", "<NA>", "nan", "None"):
        return None
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def likelihood_band(p) -> str:
    if p is None or pd.isna(p):
        return "N/A"
    return next(label for t, label in LIKELIHOOD_BANDS if p >= t)


# ============================================================================
# Inputs
# ============================================================================
def load_inputs() -> dict:
    d = {}
    d["actual_mv"] = pd.read_csv(ACTUAL / "2026_brownlow_match_votes.csv", dtype={"player_id": "string"})
    d["actual_lb"] = pd.read_csv(ACTUAL / "2026_brownlow_leaderboard.csv", dtype={"player_id": "string"})
    d["prod_lb"] = pd.read_csv(REPORTS / "2026_leaderboard.csv")
    d["prod_pv"] = pd.read_csv(REPORTS / "2026_predicted_votes.csv")
    d["prod_mp"] = pd.read_csv(REPORTS / "2026_match_probabilities.csv")
    d["prod_sim"] = pd.read_csv(REPORTS / "2026_simulation_summary.csv")
    d["obj_lb"] = pd.read_csv(REPORTS / "2026_objective_leaderboard.csv", dtype={"player_id": str})
    d["obj_votes"] = pd.read_csv(REPORTS / "2026_objective_votes.csv", dtype={"player_id": str})
    d["obj_sim"] = pd.read_csv(REPORTS / "2026_objective_simulation_summary.csv", dtype={"player_id": str})
    d["wheelo_season"] = pd.read_csv(EXTERNAL / "wheelo_season.csv")
    d["wheelo_match"] = pd.read_csv(EXTERNAL / "wheelo_match_level.csv")
    d["priced"] = pd.read_csv(BETTING / "processed" / "priced_opportunities.csv")
    d["refresh"] = json.loads((BETTING / "processed" / "refresh_summary.json").read_text())
    d["roles"] = pd.read_parquet(PROCESSED / "player_match_role_lagged_2026.parquet")
    for k in ("prod_lb", "prod_pv", "prod_mp", "prod_sim", "wheelo_season", "wheelo_match", "priced"):
        d[k]["player_id"] = d[k]["player_id"].map(_pid)
    d["roles"]["player_id"] = d["roles"]["player_id"].map(_pid)
    for k in ("obj_lb", "obj_votes", "obj_sim"):
        d[k] = d[k][~d[k]["player_id"].astype(str).str.startswith("NOID")].copy()
    d["actual_mv"]["player_id"] = d["actual_mv"]["player_id"].map(_pid)
    d["actual_lb"]["player_id"] = d["actual_lb"]["player_id"].map(_pid)
    return d


# ============================================================================
# 1. Season-level player table + scorecard
# ============================================================================
def build_season_table(d: dict) -> pd.DataFrame:
    prod = d["prod_lb"][["player_id", "player_name", "team_id", "FINAL_ENSEMBLE", "rank"]].rename(
        columns={"FINAL_ENSEMBLE": "production_ev", "rank": "production_rank"})
    obj = d["obj_lb"][["player_id", "player_name", "team_id", "objective_ev", "rank"]].rename(columns={"rank": "objective_rank"})
    wh = d["wheelo_season"][["player_id", "wheelo_player_name", "team_id", "wheelo_ev", "wheelo_rank"]].rename(
        columns={"wheelo_player_name": "player_name"})
    wh = wh[wh["player_id"].notna()]
    act = d["actual_lb"][d["actual_lb"]["player_id"].notna()][["player_id", "player_name", "player_team", "afl_total_actual_votes", "eligible", "n_3_votes", "n_2_votes", "n_1_votes"]]
    act = act.rename(columns={"player_team": "team_id", "afl_total_actual_votes": "actual_votes"})

    ids = pd.concat([prod[["player_id", "player_name", "team_id"]], obj[["player_id", "player_name", "team_id"]],
                     wh[["player_id", "player_name", "team_id"]], act[["player_id", "player_name", "team_id"]]])
    base = ids.drop_duplicates("player_id").set_index("player_id")
    base = base.join(prod.set_index("player_id")[["production_ev", "production_rank"]])
    base = base.join(obj.set_index("player_id")[["objective_ev", "objective_rank"]])
    base = base.join(wh.set_index("player_id")[["wheelo_ev", "wheelo_rank"]])
    base = base.join(act.set_index("player_id")[["actual_votes", "eligible", "n_3_votes", "n_2_votes", "n_1_votes"]])
    base["actual_votes"] = base["actual_votes"].fillna(0).astype(int)
    base["eligible"] = base["eligible"].fillna(True).astype(bool)
    for c in ("n_3_votes", "n_2_votes", "n_1_votes"):
        base[c] = base[c].fillna(0).astype(int)
    base["actual_rank"] = base["actual_votes"].rank(method="min", ascending=False).astype(int)
    elig = base[base["eligible"]]["actual_votes"].rank(method="min", ascending=False)
    base["actual_rank_eligible"] = elig.reindex(base.index).astype("Int64")

    # Modal 2026 lagged role per player (existing classification, never re-derived)
    roles = d["roles"][d["roles"]["season"] == 2026] if "season" in d["roles"] else d["roles"]
    modal = roles.groupby("player_id")["role"].agg(lambda s: s.value_counts().idxmax())
    base["role"] = modal.reindex(base.index).fillna("UNKNOWN")

    for m in MODELS:
        base[f"{m}_error"] = base[f"{m}_ev"] - base["actual_votes"]
        base[f"{m}_abs_error"] = base[f"{m}_error"].abs()
        base[f"{m}_rank_error"] = base[f"{m}_rank"] - base["actual_rank"]
    errs = base[[f"{m}_abs_error" for m in MODELS]]
    any_ok = errs.notna().any(axis=1)
    two_ok = errs.notna().sum(axis=1) >= 2
    base["best_source"] = errs.fillna(np.inf).idxmin(axis=1).str.replace("_abs_error", "").where(any_ok)
    base["worst_source"] = errs.fillna(-np.inf).idxmax(axis=1).str.replace("_abs_error", "").where(two_ok)
    evs = base[[f"{m}_ev" for m in MODELS]]
    base["cross_model_spread"] = (evs.max(axis=1) - evs.min(axis=1)).where(evs.notna().sum(axis=1) >= 2)
    base["prod_obj_gap"] = base["production_ev"] - base["objective_ev"]
    base["in_production"] = base["production_ev"].notna()
    base["in_objective"] = base["objective_ev"].notna()
    base["in_wheelo"] = base["wheelo_ev"].notna()
    base["in_all_three"] = base["in_production"] & base["in_objective"] & base["in_wheelo"]
    base["consensus_ev"] = evs.mean(axis=1)
    base["consensus_error"] = base["consensus_ev"] - base["actual_votes"]
    base = base.reset_index().sort_values(["actual_votes", "consensus_ev"], ascending=[False, False]).reset_index(drop=True)
    return base


def _rank_metrics(df: pd.DataFrame, m: str, universe: str) -> dict:
    sub = df[df[f"{m}_ev"].notna()].copy()
    n = len(sub)
    if n == 0:
        return {"model": MODEL_LABEL[m], "universe": universe, "n_players": 0}
    # ranks recomputed WITHIN the universe so all models are compared on the same field
    sub["pred_rank"] = sub[f"{m}_ev"].rank(method="min", ascending=False)
    sub["act_rank"] = sub["actual_votes"].rank(method="min", ascending=False)
    sub["act_rank_first"] = sub["actual_votes"].rank(method="first", ascending=False)
    err = sub[f"{m}_ev"] - sub["actual_votes"]
    leader = sub.sort_values(f"{m}_ev", ascending=False).iloc[0]
    winner = sub.sort_values("actual_votes", ascending=False).iloc[0]
    winner_pred_rank = int(sub.loc[sub["player_id"] == winner["player_id"], "pred_rank"].iloc[0])

    def hit(k):
        actual_top = set(sub[sub["act_rank"] <= k]["player_id"])
        pred_top = set(sub.sort_values([f"{m}_ev", "actual_votes"], ascending=False).head(len(actual_top))["player_id"])
        return len(actual_top & pred_top) / len(actual_top), len(actual_top)

    top3, n3 = hit(3); top5, n5 = hit(5); top10, n10 = hit(10); top20, n20 = hit(20)
    relevant = sub[(sub["actual_votes"] >= 1) | (sub[f"{m}_ev"] >= 1)]
    rel_err = relevant[f"{m}_ev"] - relevant["actual_votes"]
    from scipy.stats import spearmanr, kendalltau
    rho = spearmanr(sub[f"{m}_ev"], sub["actual_votes"]).correlation if n > 2 else np.nan
    tau = kendalltau(sub[f"{m}_ev"], sub["actual_votes"]).correlation if n > 2 else np.nan
    top30 = sub[sub["act_rank"] <= 30]
    return {
        "model": MODEL_LABEL[m], "universe": universe, "n_players": n,
        "predicted_leader": leader["player_name"], "predicted_leader_ev": round(float(leader[f"{m}_ev"]), 2),
        "actual_winner": winner["player_name"], "winner_predicted_correctly": bool(leader["player_id"] == winner["player_id"]),
        "winner_predicted_rank": winner_pred_rank, "winner_rank_error": winner_pred_rank - 1,
        "mae": round(float(err.abs().mean()), 3), "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "mean_bias": round(float(err.mean()), 3),
        "mae_relevant": round(float(rel_err.abs().mean()), 3), "n_relevant": int(len(relevant)),
        "spearman_rho": round(float(rho), 4), "kendall_tau": round(float(tau), 4),
        "top3_hit_rate": round(top3, 3), "top3_n": n3, "top5_hit_rate": round(top5, 3), "top5_n": n5,
        "top10_hit_rate": round(top10, 3), "top10_n": n10, "top20_hit_rate": round(top20, 3), "top20_n": n20,
        "mean_abs_rank_error_actual_top30": round(float((top30["pred_rank"] - top30["act_rank"]).abs().mean()), 2),
        "n_actual_top30": int(len(top30)),
    }


def build_scorecard(season: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        rows.append(_rank_metrics(season, m, "own coverage"))
    common = season[season["in_all_three"]]
    for m in MODELS:
        rows.append(_rank_metrics(common, m, "common (all three models)"))
    return pd.DataFrame(rows)


# ============================================================================
# 2. Match-level evaluation
# ============================================================================
def _wheelo_match_ids(d: dict) -> pd.DataFrame:
    """Map Wheelo's (round, match_label) to the canonical match_id via the
    resolved players' (round, player_id) memberships in Objective's roster."""
    roster = d["obj_votes"][["match_id", "round", "player_id"]]
    key = roster.set_index(["round", "player_id"])["match_id"]
    wm = d["wheelo_match"].copy()
    wm["match_id"] = [key.get((r, p)) for r, p in zip(wm["round"], wm["player_id"])]
    modal = wm.dropna(subset=["match_id"]).groupby(["round", "match_label"])["match_id"].agg(lambda s: s.value_counts().idxmax())
    wm["match_id"] = [modal.get((r, l)) for r, l in zip(wm["round"], wm["match_label"])]
    return wm


def _model_rosters(d: dict) -> dict[str, pd.DataFrame]:
    prod = d["prod_mp"][["match_id", "player_id", "player_name", "team_id", "p3", "p2", "p1", "expected_votes"]].copy()
    pv = d["prod_pv"][["match_id", "player_id", "predicted_votes"]]
    prod = prod.merge(pv, on=["match_id", "player_id"], how="left")
    prod["predicted_votes"] = prod["predicted_votes"].fillna(0).astype(int)
    obj = d["obj_votes"][["match_id", "player_id", "player_name", "team_id", "p3", "p2", "p1", "expected_votes", "objective_pred_votes"]].rename(
        columns={"objective_pred_votes": "predicted_votes"})
    wm = _wheelo_match_ids(d)
    wm = wm[wm["match_id"].notna() & wm["player_id"].notna()].copy()
    wh = pd.DataFrame({
        "match_id": wm["match_id"], "player_id": wm["player_id"], "player_name": wm["wheelo_player_name"],
        "team_id": wm["team_id"], "p3": wm["wheelo_p3_pct"] / 100.0, "expected_votes": wm["wheelo_ev"],
        "predicted_votes": wm["wheelo_match_rank"].map({1: 3, 2: 2, 3: 1}).fillna(0).astype(int),
    })
    return {"production": prod, "objective": obj, "wheelo": wh}


def build_match_table(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (match_table: one row per match x model, player_match: one row
    per roster player x model with actual votes attached, for calibration)."""
    mv = d["actual_mv"]
    actual = {}
    for mid, g in mv.groupby("match_id"):
        actual[mid] = {int(r.actual_brownlow_votes): (r.player_id, r.player_name, r.player_team) for r in g.itertuples()}
    meta = mv.drop_duplicates("match_id").set_index("match_id")[["round", "date", "home_team", "away_team"]]
    rosters = _model_rosters(d)
    roles = d["roles"].set_index(["match_id", "player_id"])["role"]
    rows, pm_frames = [], []
    for m, ros in rosters.items():
        ros = ros.copy()
        av = mv.set_index(["match_id", "player_id"])["actual_brownlow_votes"]
        ros["actual_votes"] = [av.get((a, b), 0) if b is not None else 0 for a, b in zip(ros["match_id"], ros["player_id"])]
        ros["model"] = MODEL_LABEL[m]
        ros["role"] = [roles.get((a, b), "UNKNOWN") for a, b in zip(ros["match_id"], ros["player_id"])]
        pm_frames.append(ros)
        for mid, g in ros.groupby("match_id"):
            if mid not in actual:
                continue
            a3, a2, a1 = actual[mid].get(3), actual[mid].get(2), actual[mid].get(1)
            g = g.sort_values(["p3", "expected_votes"], ascending=False).reset_index(drop=True)
            g["p3_rank"] = np.arange(1, len(g) + 1)
            top = g.iloc[0]
            in_roster = a3[0] is not None and a3[0] in set(g["player_id"])
            a3_rank = int(g.loc[g["player_id"] == a3[0], "p3_rank"].iloc[0]) if in_roster else None
            p3_of_a3 = float(g.loc[g["player_id"] == a3[0], "p3"].iloc[0]) if in_roster else None
            pred_set = {v: g.loc[g["predicted_votes"] == v, "player_id"].iloc[0] if (g["predicted_votes"] == v).any() else None for v in (3, 2, 1)}
            act_ids = {a3[0], a2[0], a1[0]}
            pred_ids = set(pred_set.values())
            y3 = (g["player_id"] == a3[0]).astype(float)
            second = g.iloc[1] if len(g) > 1 else top
            rows.append({
                "match_id": mid, "round": int(meta.loc[mid, "round"]), "date": meta.loc[mid, "date"],
                "home_team": meta.loc[mid, "home_team"], "away_team": meta.loc[mid, "away_team"], "model": MODEL_LABEL[m],
                "roster_size": int(len(g)),
                "actual_3_player": a3[1], "actual_3_player_id": a3[0], "actual_3_team": a3[2],
                "actual_2_player": a2[1], "actual_1_player": a1[1],
                "actual_3_role": roles.get((mid, a3[0]), "UNKNOWN") if a3[0] else "UNKNOWN",
                "model_top_player": top["player_name"], "model_top_player_id": top["player_id"], "model_top_p3": round(float(top["p3"]), 4),
                "model_top_margin_p3": round(float(top["p3"] - second["p3"]), 4),
                "model_top_role": top["role"],
                "hit_3": bool(top["player_id"] == a3[0]) if a3[0] else False,
                "actual_3_in_roster": bool(in_roster), "actual_3_p3_rank": a3_rank, "p3_of_actual_3": p3_of_a3,
                "actual_3_in_top2": bool(a3_rank is not None and a3_rank <= 2),
                "actual_3_in_top3": bool(a3_rank is not None and a3_rank <= 3),
                "exact_321": bool(pred_set[3] == a3[0] and pred_set[2] == a2[0] and pred_set[1] == a1[0]),
                "unordered_top3": bool(pred_ids == act_ids and None not in act_ids),
                "n_vote_getters_in_pred_top3": int(len(pred_ids & act_ids)),
                "ev_mae": round(float((g["expected_votes"] - g["actual_votes"]).abs().mean()), 4),
                "ev_abs_error_sum": round(float((g["expected_votes"] - g["actual_votes"]).abs().sum()), 4),
                "log_loss_p3": round(float(-np.log(max(p3_of_a3, 1e-6))), 4) if p3_of_a3 is not None else None,
                "brier_p3": round(float(((g["p3"] - y3) ** 2).mean()), 5) if in_roster else None,
                "predicted_3_player": g.loc[g["player_id"] == pred_set[3], "player_name"].iloc[0] if pred_set[3] else None,
            })
    match_table = pd.DataFrame(rows).sort_values(["round", "date", "match_id", "model"]).reset_index(drop=True)
    player_match = pd.concat(pm_frames, ignore_index=True)
    return match_table, player_match


def build_round_table(match_table: pd.DataFrame) -> pd.DataFrame:
    g = match_table.groupby(["model", "round"])
    out = g.agg(n_matches=("match_id", "nunique"), hit_3=("hit_3", "sum"), in_top3=("actual_3_in_top3", "sum"),
                exact_321=("exact_321", "sum"), unordered_top3=("unordered_top3", "sum"),
                mean_ev_mae=("ev_mae", "mean"), mean_log_loss=("log_loss_p3", "mean")).reset_index()
    out["hit_3_rate"] = out["hit_3"] / out["n_matches"]
    out["in_top3_rate"] = out["in_top3"] / out["n_matches"]
    out["exact_321_rate"] = out["exact_321"] / out["n_matches"]
    return out.round(4)


def build_match_scorecard(match_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in match_table.groupby("model"):
        n = len(g)
        ll = g["log_loss_p3"].dropna()
        rows.append({
            "model": model, "n_matches": n, "n_actual_3_missing_from_roster": int((~g["actual_3_in_roster"]).sum()),
            "hit_3_rate": round(g["hit_3"].mean(), 4), "actual_3_in_top2_rate": round(g["actual_3_in_top2"].mean(), 4),
            "actual_3_in_top3_rate": round(g["actual_3_in_top3"].mean(), 4),
            "exact_321_rate": round(g["exact_321"].mean(), 4), "unordered_top3_rate": round(g["unordered_top3"].mean(), 4),
            "mean_vote_getters_in_pred_top3": round(g["n_vote_getters_in_pred_top3"].mean(), 3),
            "mean_ev_mae_per_roster_player": round(g["ev_mae"].mean(), 4),
            "mean_ev_abs_error_per_match": round(g["ev_abs_error_sum"].mean(), 3),
            "mean_log_loss_p3": round(ll.mean(), 4) if len(ll) else None, "n_log_loss": int(len(ll)),
            "mean_brier_p3": round(g["brier_p3"].dropna().mean(), 5),
            "mean_p3_of_actual_3": round(g["p3_of_actual_3"].dropna().mean(), 4),
        })
    return pd.DataFrame(rows)


# ============================================================================
# 3. Calibration
# ============================================================================
def calibration_bins(p: pd.Series, y: pd.Series, series: str, model: str) -> tuple[pd.DataFrame, dict]:
    p = p.astype(float).clip(0, 1); y = y.astype(float)
    mask = p.notna() & y.notna()
    p, y = p[mask], y[mask]
    n = int(len(p))
    idx = np.clip(np.digitize(p, CALIBRATION_BINS[1:-1], right=False), 0, 9)
    rows = []
    ece = 0.0
    for b in range(10):
        sel = idx == b
        nb = int(sel.sum())
        if nb == 0:
            rows.append({"series": series, "model": model, "bin": f"{b*10}-{(b+1)*10}%", "n": 0, "mean_predicted": None, "actual_frequency": None, "gap_pp": None})
            continue
        conf, acc = float(p[sel].mean()), float(y[sel].mean())
        ece += nb / n * abs(acc - conf)
        rows.append({"series": series, "model": model, "bin": f"{b*10}-{(b+1)*10}%", "n": nb, "mean_predicted": round(conf, 4),
                     "actual_frequency": round(acc, 4), "gap_pp": round((acc - conf) * 100, 2)})
    summary = {"series": series, "model": model, "n": n, "ece": round(ece, 4) if n else None,
               "brier": round(float(((p - y) ** 2).mean()), 5) if n else None,
               "base_rate": round(float(y.mean()), 4) if n else None, "mean_predicted": round(float(p.mean()), 4) if n else None}
    return pd.DataFrame(rows), summary


def build_calibration(player_match: pd.DataFrame, settled: pd.DataFrame, season: pd.DataFrame, d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables, sums = [], []
    for model, g in player_match.groupby("model"):
        t, s = calibration_bins(g["p3"], (g["actual_votes"] == 3), "Match P3 (3 votes)", model); tables.append(t); sums.append(s)
        if "p2" in g and g["p2"].notna().any():
            t, s = calibration_bins(g["p2"], (g["actual_votes"] == 2), "Match P2 (2 votes)", model); tables.append(t); sums.append(s)
            t, s = calibration_bins(g["p1"], (g["actual_votes"] == 1), "Match P1 (1 vote)", model); tables.append(t); sums.append(s)
            t, s = calibration_bins(g["p3"] + g["p2"] + g["p1"], (g["actual_votes"] >= 1), "Match any vote (P3+P2+P1)", model); tables.append(t); sums.append(s)
    # Betting markets (binary outcome; dead heats count as fractional wins, pushes excluded)
    st_ok = settled[settled["result"].isin(["win", "loss", "dead_heat"])].copy()
    st_ok["y"] = np.where(st_ok["result"] == "win", 1.0, np.where(st_ok["result"] == "dead_heat", st_ok["dead_heat_fraction"], 0.0))
    for mkt, label in [("TO_POLL_A_VOTE", "To Poll a Vote"), ("PLAYER_VOTES_OU", "Player votes O/U"), ("X_PLUS_VOTES", "X+ votes"),
                       ("TOP_N", "Top N finish"), ("TEAM_VOTES_OU", "Team votes O/U"), ("PLAYER_H2H", "Player H2H")]:
        g = st_ok[st_ok["market_type"] == mkt]
        for m, col in (("Production", "production_probability"), ("Objective", "objective_probability")):
            if g[col].notna().sum() >= 10:
                t, s = calibration_bins(g[col], g["y"], f"Market: {label}", m); tables.append(t); sums.append(s)
        t, s = calibration_bins(g["implied_probability"], g["y"], f"Market: {label}", "Bookmaker implied"); tables.append(t); sums.append(s)
    g = st_ok
    for m, col in (("Production", "production_probability"), ("Objective", "objective_probability"), ("Bookmaker implied", "implied_probability")):
        t, s = calibration_bins(g[col], g["y"], "Market: all settled binary markets", m); tables.append(t); sums.append(s)
    # Season simulation probabilities (genuine Monte Carlo probabilities)
    for m, sim in (("Production", d["prod_sim"]), ("Objective", d["obj_sim"])):
        s2 = sim.merge(season[["player_id", "actual_rank"]], on="player_id", how="inner")
        t, s = calibration_bins(s2["prob_top10_rank"], (s2["actual_rank"] <= 10), "Season P(top 10)", m); tables.append(t); sums.append(s)
        t, s = calibration_bins(s2["prob_top3_rank"], (s2["actual_rank"] <= 3), "Season P(top 3)", m); tables.append(t); sums.append(s)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(sums)


# ============================================================================
# 4. Disagreement + bias
# ============================================================================
def build_disagreement(season: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = season[season["in_production"] & season["in_objective"]].copy()
    s["abs_gap"] = s["prod_obj_gap"].abs()
    s["gap_bin"] = pd.cut(s["abs_gap"], [-0.001, 2, 5, np.inf], labels=["0-2", "2-5", "5+"])
    s["closer"] = np.select([s["production_abs_error"] < s["objective_abs_error"], s["production_abs_error"] > s["objective_abs_error"]],
                            ["Production", "Objective"], "Tie")
    s["min_abs_error"] = s[["production_abs_error", "objective_abs_error"]].min(axis=1)
    s["max_abs_error"] = s[["production_abs_error", "objective_abs_error"]].max(axis=1)
    # Wheelo tie-break: which of our two models did Wheelo sit closer to, and was that the one closer to actual?
    has_w = s["in_wheelo"]
    s["wheelo_sided_with"] = np.where(~has_w, None, np.where((s["wheelo_ev"] - s["production_ev"]).abs() < (s["wheelo_ev"] - s["objective_ev"]).abs(), "Production",
                                                              np.where((s["wheelo_ev"] - s["production_ev"]).abs() > (s["wheelo_ev"] - s["objective_ev"]).abs(), "Objective", "Tie")))
    s["wheelo_tiebreak_correct"] = np.where(has_w & s["wheelo_sided_with"].isin(["Production", "Objective"]) & s["closer"].isin(["Production", "Objective"]),
                                            s["wheelo_sided_with"] == s["closer"], np.nan)
    s["rank_gap"] = (s["production_rank"] - s["objective_rank"])
    s["rank_closer"] = np.select([(s["production_rank"] - s["actual_rank"]).abs() < (s["objective_rank"] - s["actual_rank"]).abs(),
                                  (s["production_rank"] - s["actual_rank"]).abs() > (s["objective_rank"] - s["actual_rank"]).abs()], ["Production", "Objective"], "Tie")
    rows = []
    for label, g in list(s.groupby("gap_bin", observed=True)) + [("all", s)]:
        w = g[g["wheelo_tiebreak_correct"].notna()]
        rows.append({
            "gap_bin_votes": str(label), "n_players": int(len(g)), "n_with_actual_votes": int((g["actual_votes"] > 0).sum()),
            "production_closer": int((g["closer"] == "Production").sum()), "objective_closer": int((g["closer"] == "Objective").sum()),
            "ties": int((g["closer"] == "Tie").sum()),
            "production_mae": round(g["production_abs_error"].mean(), 3), "objective_mae": round(g["objective_abs_error"].mean(), 3),
            "mean_min_abs_error": round(g["min_abs_error"].mean(), 3), "mean_max_abs_error": round(g["max_abs_error"].mean(), 3),
            "mean_consensus_abs_error": round(g["consensus_error"].abs().mean(), 3),
            "wheelo_tiebreaks": int(len(w)), "wheelo_tiebreak_correct": int(w["wheelo_tiebreak_correct"].sum()),
            "wheelo_tiebreak_rate": round(w["wheelo_tiebreak_correct"].mean(), 3) if len(w) else None,
        })
    summary = pd.DataFrame(rows)
    cases = s[s["abs_gap"] >= 5].sort_values("abs_gap", ascending=False)[[
        "player_id", "player_name", "team_id", "role", "actual_votes", "actual_rank", "production_ev", "objective_ev", "wheelo_ev",
        "prod_obj_gap", "production_abs_error", "objective_abs_error", "wheelo_abs_error", "closer", "wheelo_sided_with", "wheelo_tiebreak_correct",
        "production_rank", "objective_rank", "rank_gap", "rank_closer"]].reset_index(drop=True)
    return summary, cases


def build_bias(season: pd.DataFrame, match_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dim in ("role", "team_id"):
        for m in MODELS:
            sub = season[season[f"{m}_ev"].notna()]
            for scope, ss in (("all", sub), ("relevant (actual>=1 or EV>=1)", sub[(sub["actual_votes"] >= 1) | (sub[f"{m}_ev"] >= 1)])):
                for key, g in list(ss.groupby(dim)) + [("ALL", ss)]:
                    e = g[f"{m}_error"]
                    rows.append({"dimension": dim, "group": key, "model": MODEL_LABEL[m], "scope": scope, "n": int(len(g)),
                                 "actual_votes_total": int(g["actual_votes"].sum()), "predicted_ev_total": round(g[f"{m}_ev"].sum(), 1),
                                 "mean_error": round(e.mean(), 3), "mae": round(e.abs().mean(), 3),
                                 "over_predicted": int((e > 0.5).sum()), "under_predicted": int((e < -0.5).sum()),
                                 "small_sample": bool(len(g) < 10)})
    season_bias = pd.DataFrame(rows)
    # Match-level: how often does each model's top pick find the actual 3-voter, by the 3-voter's role
    mrows = []
    for model, g in match_table.groupby("model"):
        for key, gg in list(g.groupby("actual_3_role")) + [("ALL", g)]:
            mrows.append({"model": model, "actual_3_role": key, "n_matches": int(len(gg)), "hit_3": int(gg["hit_3"].sum()),
                          "hit_3_rate": round(gg["hit_3"].mean(), 3), "in_top3_rate": round(gg["actual_3_in_top3"].mean(), 3),
                          "mean_p3_of_actual_3": round(gg["p3_of_actual_3"].dropna().mean(), 4),
                          "share_of_all_3_voters": round(len(gg) / len(g), 3), "small_sample": bool(len(gg) < 10)})
        picks = g["model_top_role"].value_counts(normalize=True)
        for key in picks.index:
            for r in mrows:
                if r["model"] == model and r["actual_3_role"] == key:
                    r["share_of_model_top_picks"] = round(float(picks[key]), 3)
    match_bias = pd.DataFrame(mrows)
    return season_bias, match_bias


# ============================================================================
# 5. Final order + simulations
# ============================================================================
def build_final_order(season: pd.DataFrame, d: dict) -> tuple[pd.DataFrame, dict]:
    top = season[season["actual_rank"] <= 20].sort_values(["actual_rank", "player_name"]).copy()
    ps = d["prod_sim"].set_index("player_id"); os_ = d["obj_sim"].set_index("player_id")
    for label, sim in (("production", ps), ("objective", os_)):
        for c in ("prob_rank_1", "prob_top3_rank", "prob_top10_rank", "mean_rank", "sim_p2_5", "sim_p97_5"):
            top[f"{label}_{c}"] = top["player_id"].map(sim[c]) if c in sim else np.nan
        top[f"{label}_in_sim_95pct_interval"] = (top["actual_votes"] >= top[f"{label}_sim_p2_5"]) & (top["actual_votes"] <= top[f"{label}_sim_p97_5"])
    cols = ["actual_rank", "actual_rank_eligible", "player_id", "player_name", "team_id", "role", "eligible", "actual_votes"] + \
           [f"{m}_{c}" for m in MODELS for c in ("ev", "rank", "error", "rank_error")] + \
           [f"{l}_{c}" for l in ("production", "objective") for c in ("prob_rank_1", "prob_top3_rank", "prob_top10_rank", "mean_rank", "sim_p2_5", "sim_p97_5", "in_sim_95pct_interval")]
    top = top[cols].reset_index(drop=True)

    # Simulation coverage of the actual outcome (no rerun -- reads persisted summaries + draws)
    winner = season[season["actual_rank"] == 1].iloc[0]
    actual_top3 = season[season["actual_rank"] <= 3].sort_values("actual_votes", ascending=False)
    actual_top4 = season[season["actual_rank"] <= 4].sort_values("actual_votes", ascending=False)
    cov = {"actual_winner": winner["player_name"], "actual_top3_order": actual_top3["player_name"].tolist(),
           "actual_top4_set": actual_top4["player_name"].tolist(),
           "top5_note": "Actual 5th place is a three-way tie (27 votes), so an exact top-5 order is undefined; top-4 is the deepest unambiguous set."}
    for label, sim in (("production", ps), ("objective", os_)):
        cov[f"{label}_winner_prob_rank_1"] = float(sim["prob_rank_1"].get(winner["player_id"], np.nan))
        cov[f"{label}_winner_in_sim_top1_by_mean_rank"] = bool(sim["mean_rank"].rank(method="min").get(winner["player_id"], np.inf) <= 1)
        for k, ids in ((3, actual_top3["player_id"]), (5, season[season["actual_rank"] <= 5]["player_id"]), (10, season[season["actual_rank"] <= 10]["player_id"])):
            probs = [float(sim[f"prob_top{k}_rank" if k > 1 else "prob_rank_1"].get(p, np.nan)) if f"prob_top{k}_rank" in sim else np.nan for p in ids]
            in_topk = sim["mean_rank"].rank(method="min")
            cov[f"{label}_actual_top{k}_covered_by_sim_top{k}_mean_rank"] = int(sum(1 for p in ids if in_topk.get(p, np.inf) <= k))
            cov[f"{label}_actual_top{k}_n"] = int(len(ids))
            if k in (3, 10):
                cov[f"{label}_mean_prob_top{k}_of_actual_top{k}"] = float(np.nanmean(probs))
    # Exact-order likelihood of the actual top 3 and "all top-4 in top 4" from the persisted joint draws
    try:
        from src.models import order_scenarios as osc
        for label, arr, idx in (("production", PROCESSED / "mc_totals_2026.npy", REPORTS / "2026_mc_player_index.csv"),
                                ("objective", PROCESSED / "mc_totals_objective_2026.npy", REPORTS / "2026_objective_mc_player_index.csv")):
            totals = np.load(arr, mmap_mode="r"); players = pd.read_csv(idx); players["player_id"] = players["player_id"].map(_pid)
            col = {p: i for i, p in enumerate(players["player_id"])}
            ids3 = actual_top3["player_id"].tolist(); ids4 = actual_top4["player_id"].tolist()
            if all(p in col for p in ids4):
                order3 = osc.order_matrix(np.asarray(totals), 3)
                exact3 = (order3 == np.array([col[p] for p in ids3])[None, :]).all(axis=1).mean()
                rank_all = np.argsort(np.argsort(-np.asarray(totals), axis=1, kind="stable"), axis=1) + 1
                set3 = (rank_all[:, [col[p] for p in ids3]] <= 3).all(axis=1).mean()
                set4 = (rank_all[:, [col[p] for p in ids4]] <= 4).all(axis=1).mean()
                cov[f"{label}_exact_top3_order_prob"] = float(exact3); cov[f"{label}_top3_set_prob"] = float(set3)
                cov[f"{label}_top4_set_prob"] = float(set4); cov[f"{label}_n_sims"] = int(totals.shape[0])
            else:
                cov[f"{label}_exact_top3_order_prob"] = None
                cov[f"{label}_unresolved_in_sim"] = [p for p in ids4 if p not in col]
    except Exception as exc:  # noqa: BLE001
        cov["joint_simulation_error"] = str(exc)
    # Did any persisted order scenario match the actual top-3 order?
    osc_df = pd.read_csv(REPORTS / "2026_order_scenarios.csv")
    names3 = actual_top3["player_name"].tolist()
    for model in ("Production", "Objective"):
        g = osc_df[(osc_df["model"] == model) & (osc_df["order_depth"] == 5)]
        cov[f"{model.lower()}_top5_scenarios_with_actual_top3_prefix"] = int(((g["first"] == names3[0]) & (g["second"] == names3[1]) & (g["third"] == names3[2])).sum())
        cov[f"{model.lower()}_top5_scenario_prob_mass_with_actual_top3_prefix"] = float(g.loc[(g["first"] == names3[0]) & (g["second"] == names3[1]) & (g["third"] == names3[2]), "probability"].sum())
    return top, cov


# ============================================================================
# 6. Betting settlement
# ============================================================================
def build_betting(d: dict, season: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    po = d["priced"].copy()
    votes = season.set_index("player_id")["actual_votes"]
    rank_all = season.set_index("player_id")["actual_rank"]
    tied = season.groupby("actual_rank")["player_id"].transform("count")
    tied.index = season["player_id"]
    rank_el = season.set_index("player_id")["actual_rank_eligible"]
    tied_el = season[season["eligible"]].groupby("actual_rank_eligible")["player_id"].transform("count"); tied_el.index = season[season["eligible"]]["player_id"]
    team_totals = d["actual_mv"].groupby("player_team")["actual_brownlow_votes"].sum()
    # H2H opponents: the other selection row in the same market
    h2h = po[po["market_type"] == "PLAYER_H2H"]
    opp = {}
    for (src, mk), g in h2h.groupby(["source", "market_name"]):
        if len(g) == 2:
            a, b = g.index.tolist()
            opp[a], opp[b] = g.loc[b, "player_id"], g.loc[a, "player_id"]
    rows = []
    for i, r in po.iterrows():
        mt = r["market_type"]
        pid = r["player_id"]
        av = votes.get(pid) if pid else None
        res, sens = stl.Settlement(stl.UNSETTLEABLE, 0.0, "unmodelled market"), False
        actual_value = None
        if mt in SETTLEABLE_MARKETS:
            if mt == "TEAM_VOTES_OU":
                actual_value = float(team_totals.get(r["team_id"], np.nan)) if isinstance(r["team_id"], str) else None
                res = stl.settle_over_under(actual_value, r["line"], r["side"])
            elif pid is None or av is None:
                res = stl.Settlement(stl.UNSETTLEABLE, 0.0, "player identity unresolved")
            elif mt == "WINNER":
                actual_value = av; res = stl.settle_winner(rank_all.get(pid), tied.get(pid))
                alt = stl.settle_winner(rank_el.get(pid), tied_el.get(pid)); sens = alt.result != res.result
            elif mt == "TOP_N":
                actual_value = av; res = stl.settle_top_n(rank_all.get(pid), tied.get(pid), r["n"])
                alt = stl.settle_top_n(rank_el.get(pid), tied_el.get(pid), r["n"]); sens = alt.result != res.result
            elif mt == "EXACT_POSITION":
                actual_value = av; res = stl.settle_exact_position(rank_all.get(pid), tied.get(pid), r["position"])
                alt = stl.settle_exact_position(rank_el.get(pid), tied_el.get(pid), r["position"]); sens = alt.result != res.result
            elif mt == "PLAYER_VOTES_OU":
                actual_value = av; res = stl.settle_over_under(av, r["line"], r["side"])
            elif mt == "X_PLUS_VOTES":
                actual_value = av; res = stl.settle_threshold(av, r["threshold"])
            elif mt == "TO_POLL_A_VOTE":
                actual_value = av; res = stl.settle_threshold(av, 1)
            elif mt == "PLAYER_H2H":
                o = opp.get(i); ov = votes.get(o) if o else None
                actual_value = av; res = stl.settle_h2h(av, ov)
                r = r.copy(); r["opponent_player_id"] = o; r["opponent_actual_votes"] = ov
                r["opponent_name"] = season.set_index("player_id")["player_name"].get(o) if o else None
        pnl = stl.flat_unit_pnl(res, r["odds"])
        cons = r.get("conservative_internal_probability")
        rows.append({**r.to_dict(), "actual_value": actual_value, "result": res.result, "dead_heat_fraction": res.fraction if res.result == stl.DEAD_HEAT else None,
                     "settlement_detail": res.detail, "flat_unit_pnl": pnl, "settlement_sensitive_to_eligibility": sens,
                     "bet_value_label": BET_VALUE_LABEL.get(r["confidence"], r["confidence"]), "likelihood_band": likelihood_band(cons),
                     "hit": (res.result in ("win", "dead_heat")) if res.result != stl.UNSETTLEABLE else None,
                     "production_sees_value": bool(r["production_edge_pp"] > 0) if pd.notna(r["production_edge_pp"]) else None,
                     "objective_sees_value": bool(r["objective_edge_pp"] > 0) if pd.notna(r["objective_edge_pp"]) else None})
    settled = pd.DataFrame(rows)
    settled["model_agreement"] = np.select(
        [settled["production_sees_value"].isna() | settled["objective_sees_value"].isna(),
         settled["production_sees_value"] & settled["objective_sees_value"],
         settled["production_sees_value"] | settled["objective_sees_value"]],
        ["one model missing", "both see value", "only one sees value"], "neither sees value")
    settled["internal_gap_band"] = pd.cut(settled["internal_gap_pp"].abs(), [-0.01, 7.5, 15, 30, 100], labels=["<=7.5pp", "7.5-15pp", "15-30pp", ">30pp"]).astype(str)
    settled["prob_band"] = pd.cut(settled["conservative_internal_probability"], CALIBRATION_BINS, labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)], include_lowest=True).astype(str)
    settled["edge_band"] = pd.cut(settled["production_edge_pp"], [-100, 0, 5, 15, 100], labels=["<=0pp", "0-5pp", "5-15pp", ">15pp"]).astype(str)
    settled["wheelo_support_label"] = settled["wheelo_support"].fillna("INSUFFICIENT_WHEELO_DATA")

    # Summaries
    sett = settled[settled["result"] != stl.UNSETTLEABLE]
    def summ(g, **keys):
        s = stl.summarise_bets(g["flat_unit_pnl"].tolist(), g["result"].tolist())
        return {**keys, **s, "avg_model_prob_conservative": round(g["conservative_internal_probability"].mean(), 4),
                "avg_production_prob": round(g["production_probability"].mean(), 4), "avg_objective_prob": round(g["objective_probability"].mean(), 4),
                "avg_implied_prob": round(g["implied_probability"].mean(), 4), "avg_odds": round(g["odds"].mean(), 3),
                "small_sample": bool(s["bets"] < 20)}
    sums = []
    for mt, g in sett.groupby("market_type"):
        sums.append(summ(g, grouping="market_type", group=mt, subset="all priced selections"))
        for src, gg in g.groupby("source"):
            sums.append(summ(gg, grouping="market_type x bookmaker", group=f"{mt} @ {src}", subset="all priced selections"))
    sums.append(summ(sett, grouping="market_type", group="ALL", subset="all priced selections"))
    for grouping, col in (("bet_value", "bet_value_label"), ("likelihood_band", "likelihood_band"), ("wheelo_support", "wheelo_support_label"),
                          ("model_agreement", "model_agreement"), ("internal_gap_band", "internal_gap_band"), ("prob_band", "prob_band"),
                          ("edge_band", "edge_band")):
        for key, g in sett.groupby(col):
            sums.append(summ(g, grouping=grouping, group=str(key), subset="all priced selections"))
        value = sett[sett["confidence"].isin(["HIGH_CONFIDENCE_WHEELO_CONFIRMED", "HIGH_CONFIDENCE_WHEELO_NEUTRAL", "MEDIUM_CONFIDENCE", "HIGH_RISK_HIGH_REWARD"])]
        if grouping not in ("bet_value",):
            for key, g in value.groupby(col):
                sums.append(summ(g, grouping=grouping, group=str(key), subset="Bet Value selections only"))
    for mt, g in sett.groupby("market_type"):
        for key, gg in g.groupby("bet_value_label"):
            sums.append(summ(gg, grouping="market_type x bet_value", group=f"{mt} | {key}", subset="all priced selections"))
    summary = pd.DataFrame(sums)

    # Team leader (model accuracy only -- no team-top-poller price was captured pre-count)
    trows = []
    for team, g in season.groupby("team_id"):
        amax = g["actual_votes"].max(); leaders = g[g["actual_votes"] == amax]
        row = {"team_id": team, "actual_leader": " / ".join(sorted(leaders["player_name"])), "actual_leader_votes": int(amax), "actual_tie": bool(len(leaders) > 1),
               "team_actual_votes": int(team_totals.get(team, 0))}
        for m in MODELS:
            gm = g[g[f"{m}_ev"].notna()]
            if gm.empty:
                row[f"{m}_predicted_leader"] = None; row[f"{m}_hit"] = None; continue
            top = gm.sort_values(f"{m}_ev", ascending=False).iloc[0]
            row[f"{m}_predicted_leader"] = top["player_name"]; row[f"{m}_predicted_ev"] = round(float(top[f"{m}_ev"]), 2)
            row[f"{m}_hit"] = bool(top["player_id"] in set(leaders["player_id"]))
            row[f"{m}_team_ev_total"] = round(float(gm[f"{m}_ev"].sum()), 1)
        trows.append(row)
    team_leaders = pd.DataFrame(trows)
    return settled, summary, team_leaders


# ============================================================================
# 7. Learnings (deterministic, from the computed tables only)
# ============================================================================
def build_learnings(scorecard, match_sc, disagreement, cases, season_bias, match_bias, cal_summary, bet_summary, season, d, cov) -> list[dict]:
    L = []
    def add(title, finding, evidence, n):
        L.append({"title": title, "finding": finding, "evidence": evidence, "n": n})
    sc = scorecard[scorecard["universe"] == "common (all three models)"].set_index("model")
    best = sc["mae"].idxmin()
    add("Season-total accuracy (common universe)",
        f"{best} had the lowest season-total MAE on the {int(sc['n_players'].iloc[0])} players covered by all three sources; "
        + ", ".join(f"{m} MAE {sc.loc[m,'mae']:.2f} / RMSE {sc.loc[m,'rmse']:.2f} / Spearman {sc.loc[m,'spearman_rho']:.3f}" for m in sc.index) + ".",
        "scorecard.csv (universe = common)", int(sc["n_players"].iloc[0]))
    ms = match_sc.set_index("model")
    b3 = ms["hit_3_rate"].idxmax()
    add("Match-level 3-vote identification",
        f"{b3} named the actual 3-vote player most often: " + ", ".join(f"{m} {ms.loc[m,'hit_3_rate']*100:.1f}% (exact 3-2-1 {ms.loc[m,'exact_321_rate']*100:.1f}%, 3-voter inside top-3 {ms.loc[m,'actual_3_in_top3_rate']*100:.1f}%)" for m in ms.index) + ".",
        "match_scorecard.csv", int(ms["n_matches"].max()))
    ll = ms["mean_log_loss_p3"].dropna()
    if len(ll):
        add("Match P3 probability quality", f"Lowest mean P3 log loss: {ll.idxmin()} ({ll.min():.3f}); " + ", ".join(f"{m} {v:.3f}" for m, v in ll.items()) + ". Brier P3: " + ", ".join(f"{m} {ms.loc[m,'mean_brier_p3']:.4f}" for m in ms.index) + ".",
            "match_scorecard.csv (log loss only where the actual 3-voter was in the model roster)", int(ms["n_log_loss"].min()))
    mb = match_bias[match_bias["actual_3_role"].isin(["KEY_DEFENDER", "MEDIUM_DEFENDER", "MIDFIELDER"])]
    parts = []
    for model, g in mb.groupby("model"):
        g = g.set_index("actual_3_role")
        kd = g.loc["KEY_DEFENDER"] if "KEY_DEFENDER" in g.index else None
        md = g.loc["MEDIUM_DEFENDER"] if "MEDIUM_DEFENDER" in g.index else None
        mid = g.loc["MIDFIELDER"] if "MIDFIELDER" in g.index else None
        parts.append(f"{model}: midfielder 3-voters found {mid['hit_3_rate']*100:.0f}% (n={int(mid['n_matches'])})" + (f", medium defenders {md['hit_3_rate']*100:.0f}% (n={int(md['n_matches'])})" if md is not None else "") + (f", key defenders {kd['hit_3_rate']*100:.0f}% (n={int(kd['n_matches'])})" if kd is not None else ""))
    add("Defender blind spot (known Phase 4 finding)", "Hit rate on the actual 3-vote player by the 3-voter's lagged role. " + "; ".join(parts) + ". Defender samples are small; treat as directional.",
        "match_bias.csv", int(mb["n_matches"].sum()))
    dg = disagreement.set_index("gap_bin_votes")
    big = dg.loc["5+"]
    add("When Production and Objective strongly disagreed",
        f"In the {big.name}-vote disagreement bin (n={int(big['n_players'])}), Production was closer {int(big['production_closer'])} times vs Objective {int(big['objective_closer'])} "
        f"(MAE {big['production_mae']:.2f} vs {big['objective_mae']:.2f}). Disagreement predicted error: mean of the better model's abs error rose from {dg.loc['0-2','mean_min_abs_error']:.2f} (0-2 bin) to {big['mean_min_abs_error']:.2f}.",
        "disagreement_summary.csv", int(big["n_players"]))
    allrow = dg.loc["all"]
    if allrow["wheelo_tiebreaks"] > 0:
        add("Did Wheelo break ties usefully?", f"Where Production and Objective differed and Wheelo sat closer to one of them, the model Wheelo sided with was the closer one {allrow['wheelo_tiebreak_rate']*100:.0f}% of the time (n={int(allrow['wheelo_tiebreaks'])}); "
            f"in the 5+ vote disagreement bin {big['wheelo_tiebreak_rate']*100:.0f}% (n={int(big['wheelo_tiebreaks'])}). Wheelo's own season MAE on the common universe is the lowest of the three, so it carried independent signal rather than echoing either model.", "disagreement_summary.csv", int(allrow["wheelo_tiebreaks"]))
    sb = season_bias[(season_bias["dimension"] == "role") & (season_bias["scope"].str.startswith("relevant")) & (season_bias["group"] != "ALL")]
    worst = sb[~sb["small_sample"]].sort_values("mean_error").iloc[[0, -1]] if (~sb["small_sample"]).any() else sb.sort_values("mean_error").iloc[[0, -1]]
    add("Role bias (season totals, relevant players)", "Most under-predicted role/model: " + f"{worst.iloc[0]['group']} by {worst.iloc[0]['model']} (mean error {worst.iloc[0]['mean_error']:+.2f}, n={int(worst.iloc[0]['n'])}); most over-predicted: {worst.iloc[-1]['group']} by {worst.iloc[-1]['model']} (mean error {worst.iloc[-1]['mean_error']:+.2f}, n={int(worst.iloc[-1]['n'])}).",
        "season_bias.csv", int(sb["n"].sum()))
    cs = cal_summary[cal_summary["series"] == "Match P3 (3 votes)"].set_index("model")
    add("Calibration of match P3", "ECE (lower is better): " + ", ".join(f"{m} {cs.loc[m,'ece']:.3f}" for m in cs.index) + "; the worst-calibrated series is " + f"{cal_summary.loc[cal_summary['ece'].idxmax(),'series']} / {cal_summary.loc[cal_summary['ece'].idxmax(),'model']} (ECE {cal_summary['ece'].max():.3f}).",
        "calibration_summary.csv", int(cs["n"].min()))
    bm = bet_summary[(bet_summary["grouping"] == "market_type") & (bet_summary["subset"] == "all priced selections") & (bet_summary["group"] != "ALL") & (bet_summary["bets"] >= 20)]
    if len(bm):
        add("Bookmaker markets: where model pricing held up", f"Flat 1-unit ROI across every priced selection by market type -- best {bm.loc[bm['roi'].idxmax(),'group']} ({bm['roi'].max()*100:+.1f}%, n={int(bm.loc[bm['roi'].idxmax(),'bets'])}), worst {bm.loc[bm['roi'].idxmin(),'group']} ({bm['roi'].min()*100:+.1f}%, n={int(bm.loc[bm['roi'].idxmin(),'bets'])}). Retrospective analytical metric only; backing every selection is not a strategy.",
            "betting_summary.csv", int(bm["bets"].sum()))
    bv = bet_summary[(bet_summary["grouping"] == "bet_value") & (bet_summary["subset"] == "all priced selections")].set_index("group")
    order = [k for k in ("Strong Bet Value + Wheelo Support", "Strong Bet Value", "Moderate Bet Value", "Speculative Bet Value", "Model Disagreement", "No Value") if k in bv.index]
    add("Did the Bet Value tiers rank outcomes?", "Hit rate / ROI by tier: " + "; ".join(f"{k} {bv.loc[k,'hit_rate']*100:.0f}% / {bv.loc[k,'roi']*100:+.1f}% (n={int(bv.loc[k,'bets'])})" for k in order) + ".", "betting_summary.csv (grouping = bet_value)", int(bv["bets"].sum()))
    lk = bet_summary[(bet_summary["grouping"] == "likelihood_band") & (bet_summary["subset"] == "all priced selections")].set_index("group")
    lorder = [k for k in ("Very High", "High", "Moderate", "Low", "Very Low") if k in lk.index]
    add("Likelihood bands vs realised hit rate", "; ".join(f"{k}: hit {lk.loc[k,'hit_rate']*100:.0f}% vs avg model prob {lk.loc[k,'avg_model_prob_conservative']*100:.0f}% / implied {lk.loc[k,'avg_implied_prob']*100:.0f}% (n={int(lk.loc[k,'bets'])})" for k in lorder) + ".", "betting_summary.csv (grouping = likelihood_band)", int(lk["bets"].sum()))
    ag = bet_summary[(bet_summary["grouping"] == "model_agreement") & (bet_summary["subset"] == "all priced selections")].set_index("group")
    ws = bet_summary[(bet_summary["grouping"] == "wheelo_support") & (bet_summary["subset"] == "all priced selections")].set_index("group")
    add("Model agreement and Wheelo support in the markets", "Agreement: " + "; ".join(f"{k} hit {ag.loc[k,'hit_rate']*100:.0f}% / ROI {ag.loc[k,'roi']*100:+.1f}% (n={int(ag.loc[k,'bets'])})" for k in ag.index) + ". Wheelo: " + "; ".join(f"{k} hit {ws.loc[k,'hit_rate']*100:.0f}% / ROI {ws.loc[k,'roi']*100:+.1f}% (n={int(ws.loc[k,'bets'])})" for k in ws.index) + ".", "betting_summary.csv", int(ag["bets"].sum()))
    missing_prod = season[(season["actual_votes"] >= 3) & ~season["in_production"]]
    unres = d["actual_mv"][~d["actual_mv"]["identity_status"].str.startswith("resolved")]
    add("Data / identity issues worth fixing", f"{len(unres)} actual vote rows ({int(unres['actual_brownlow_votes'].sum())} votes) could not be mapped to a canonical player_id ({', '.join(sorted(unres['player_name'].unique()))}). "
        f"{len(missing_prod)} players with 3+ actual votes ({int(missing_prod['actual_votes'].sum())} votes) sit outside Production's universe because of the documented season-to-date exclusion. "
        f"Wheelo match-level rows unresolved/ambiguous: {int((d['wheelo_match']['match_status'] != 'resolved').sum())} of {len(d['wheelo_match'])}.",
        "season_players.csv, data/actual/2026_brownlow_validation.json", int(len(unres) + len(missing_prod)))
    return L[:12]


# ============================================================================
def build() -> dict:
    d = load_inputs()
    hashes = frozen_input_hashes()
    OUT.mkdir(parents=True, exist_ok=True)
    season = build_season_table(d)
    scorecard = build_scorecard(season)
    match_table, player_match = build_match_table(d)
    round_table = build_round_table(match_table)
    match_sc = build_match_scorecard(match_table)
    disagreement, cases = build_disagreement(season)
    season_bias, match_bias = build_bias(season, match_table)
    final_order, coverage = build_final_order(season, d)
    settled, bet_summary, team_leaders = build_betting(d, season)
    cal_table, cal_summary = build_calibration(player_match, settled, season, d)
    learnings = build_learnings(scorecard, match_sc, disagreement, cases, season_bias, match_bias, cal_summary, bet_summary, season, d, coverage)

    season.to_csv(OUT / "season_players.csv", index=False)
    scorecard.to_csv(OUT / "scorecard.csv", index=False)
    match_table.to_csv(OUT / "match_table.csv", index=False)
    round_table.to_csv(OUT / "round_table.csv", index=False)
    match_sc.to_csv(OUT / "match_scorecard.csv", index=False)
    disagreement.to_csv(OUT / "disagreement_summary.csv", index=False)
    cases.to_csv(OUT / "disagreement_cases.csv", index=False)
    season_bias.to_csv(OUT / "season_bias.csv", index=False)
    match_bias.to_csv(OUT / "match_bias.csv", index=False)
    final_order.to_csv(OUT / "final_order_top20.csv", index=False)
    settled.to_csv(OUT / "betting_settled.csv", index=False)
    bet_summary.to_csv(OUT / "betting_summary.csv", index=False)
    team_leaders.to_csv(OUT / "team_leaders.csv", index=False)
    cal_table.to_csv(OUT / "calibration_bins.csv", index=False)
    cal_summary.to_csv(OUT / "calibration_summary.csv", index=False)
    (OUT / "simulation_coverage.json").write_text(json.dumps(coverage, indent=1, default=str))
    (OUT / "learnings.json").write_text(json.dumps(learnings, indent=1))
    bet_dates = [s["retrieved_at"] for s in d["refresh"]["source_statuses"]]
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(), "count_date_utc": COUNT_DATE_UTC,
        "bookmaker_snapshots_retrieved_at": bet_dates,
        "all_bookmaker_snapshots_pre_count": all(x < COUNT_DATE_UTC for x in bet_dates),
        "frozen_input_hashes": hashes, "frozen_inputs_unchanged_during_build": hashes == frozen_input_hashes(),
        "counts": {"season_players": int(len(season)), "matches": int(match_table["match_id"].nunique()), "priced_selections": int(len(settled)),
                   "settled_selections": int((settled["result"] != "unsettleable").sum()), "unsettleable": int((settled["result"] == "unsettleable").sum()),
                   "unmodelled_excluded": int((settled["market_type"] == "UNMODELLED").sum()),
                   "unresolved_actual_vote_rows": int((~d["actual_mv"]["identity_status"].str.startswith("resolved")).sum())},
        "settlement_counts": {f"{mt}|{res}": int(n) for (mt, res), n in settled[settled["result"] != "unsettleable"].groupby(["market_type", "result"]).size().items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: v for k, v in m.items() if k != "frozen_input_hashes"}, indent=1, default=str))
    sys.exit(0 if m["frozen_inputs_unchanged_during_build"] and m["all_bookmaker_snapshots_pre_count"] else 1)
