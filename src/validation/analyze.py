"""
Post-hoc analysis over the walk-forward OOF predictions in
data/experiments/oof/: model comparison tables, calibration (raw vs
chronologically-fitted isotonic / Platt), disagreement-as-information,
ensemble research, Error Lab dataset, role / team / context bias, and the
2026 forensic hypothesis tests (ruck over-prediction, defender blind spot)
checked season by season. Writes data/experiments/analysis/*.
Run after the candidate suite:  python -m src.validation.analyze
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.ensemble.stacking import walk_forward_ensemble
from src.validation import metrics as M

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data" / "experiments"
OOF, MET, OUT = EXP / "oof", EXP / "metrics", EXP / "analysis"
FEAT = ROOT / "data" / "features" / "player_match_features.parquet"

CANDIDATES = {"A_structural_pl": "Structural (A)", "B_performance_xgb_rank": "Performance ML (B, default params)", "B_performance_xgb_rank_tuned": "Performance ML (B)", "C_stats_only_pl": "Stats-only (C)",
              "C_stats_only_xgb_rank": "Stats-only ML (C-ML)", "baseline_phase4_pl_legacy": "Baseline (Phase 4 PL)"}
ERROR_LAB_FEATURES = ["role", "is_win", "margin", "absolute_margin", "is_close_game", "is_blowout", "disposals", "contested_possessions", "clearances",
                      "tackles", "goals", "marks", "inside_50s", "rebound_50s", "one_percenters", "hitouts", "impact_z", "impact_match_rank",
                      "n_strong_teammates", "impact_gap_best_teammate", "prior_seasons_votes_per_game", "team_std_win_pct"]


def load_oof(name: str, window: str = "expanding") -> pd.DataFrame | None:
    p = OOF / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df[df["window"] == window] if "window" in df else df
    df["player_id"] = df["player_id"].astype(str)
    return df.reset_index(drop=True)


def restrict_to_common_matches(oofs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fair comparison: every model scored on exactly the same matches (the
    legacy dropna baseline skips matches whose vote-getters lack lagged form)."""
    common = None
    for df in oofs.values():
        pm = M.per_match_table(df)["match_id"]
        common = set(pm) if common is None else common & set(pm)
    return {n: df[df["match_id"].isin(common)].reset_index(drop=True) for n, df in oofs.items()}


def restrict_to_common_players(oofs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Season metrics on exactly the same (season, player) universe for every model."""
    common = None
    for df in oofs.values():
        keys = set(zip(df["season"], df["player_id"]))
        common = keys if common is None else common & keys
    out = {}
    for n, df in oofs.items():
        m = pd.Series(list(zip(df["season"], df["player_id"]))).isin(common).to_numpy()
        out[n] = df[m].reset_index(drop=True)
    return out


def comparison_tables(oofs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    oofs_m = restrict_to_common_matches(oofs)
    oofs_p = restrict_to_common_players(oofs)
    for name, df in oofs_m.items():
        for s, g in df.groupby("season"):
            mm = M.match_metrics(g); sm = M.season_metrics(oofs_p[name][oofs_p[name]["season"] == s])["by_season"][0]
            rows.append({"model": CANDIDATES.get(name, name), "experiment": name, "season": int(s), **{k: v for k, v in mm.items()}, **{k: v for k, v in sm.items() if k != "season"}})
    by_season = pd.DataFrame(rows)
    cols = ["correct_3", "a3_in_top2", "a3_in_top3", "exact_321", "unordered_top3", "log_loss_p3", "brier_p3", "ece_p3", "season_mae", "season_rmse", "spearman", "rank_mae_top30", "top3_hit", "top5_hit", "top10_hit", "winner_correct"]
    def agg(g):
        return pd.Series({**{c: g[c].astype(float).mean() for c in cols}, **{f"{c}_sd": g[c].astype(float).std(ddof=0) for c in ("correct_3", "log_loss_p3", "season_mae")}, "n_seasons": len(g), "n_matches": int(g["n_matches"].sum())})
    pooled = pd.concat([by_season.groupby("model").apply(agg, include_groups=False).assign(window_seasons="all (2012-2026)"),
                        by_season[by_season["season"] >= 2022].groupby("model").apply(agg, include_groups=False).assign(window_seasons="recent (2022-2026)"),
                        by_season[by_season["season"] <= 2025].groupby("model").apply(agg, include_groups=False).assign(window_seasons="pre-2026 (2012-2025)")]).reset_index()
    return by_season, pooled


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str = "correct_3", n_boot: int = 1000) -> dict:
    """Paired over matches: CI of mean(metric_a - metric_b)."""
    pa = M.per_match_table(a).set_index("match_id"); pb = M.per_match_table(b).set_index("match_id")
    common = pa.index.intersection(pb.index)
    d = (pa.loc[common, metric].astype(float) - pb.loc[common, metric].astype(float)).to_numpy()
    rng = np.random.default_rng(M.SEED)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"metric": metric, "n_matches": int(len(d)), "mean_diff": float(d.mean()), "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))}


def calibration_study(oofs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Raw vs isotonic vs Platt on P3, calibrators fit on strictly earlier seasons' OOF."""
    rows = []
    for name, df in oofs.items():
        seasons = sorted(df["season"].unique())
        for method in ("raw", "isotonic", "platt"):
            preds = []
            for t in seasons:
                prior = df[df["season"] < t]; cur = df[df["season"] == t].copy()
                if method != "raw":
                    if len(prior) == 0:
                        continue
                    y = (prior["brownlow_votes"] == 3).astype(int).to_numpy(); p = prior["p3"].to_numpy()
                    if method == "isotonic":
                        cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p, y); q = cal.predict(cur["p3"].to_numpy())
                    else:
                        lp = np.log(np.clip(p, 1e-9, 1) / np.clip(1 - p, 1e-9, 1)).reshape(-1, 1)
                        cal = LogisticRegression(C=1e6).fit(lp, y); q = cal.predict_proba(np.log(np.clip(cur["p3"], 1e-9, 1) / np.clip(1 - cur["p3"], 1e-9, 1)).to_numpy().reshape(-1, 1))[:, 1]
                    # renormalise within match so P3 still sums to 1 -> keeps ranking, changes sharpness
                    cur["p3"] = q / cur.assign(q=q).groupby("match_id")["q"].transform("sum").to_numpy()
                preds.append(cur)
            if not preds:
                continue
            pp = pd.concat(preds)
            for scope, sub in (("all", pp), ("recent (2022-2026)", pp[pp["season"] >= 2022])):
                mm = M.match_metrics(sub)
                rows.append({"model": CANDIDATES.get(name, name), "method": method, "scope": scope, "n_matches": mm["n_matches"], "ece_p3": mm["ece_p3"],
                             "brier_p3": mm["brier_p3"], "log_loss_p3": mm["log_loss_p3"], "correct_3": mm["correct_3"]})
    return pd.DataFrame(rows)


def _b_name(oofs: dict) -> str:
    return "B_performance_xgb_rank_tuned" if "B_performance_xgb_rank_tuned" in oofs else "B_performance_xgb_rank"


def disagreement_study(oofs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [n for n in ("A_structural_pl", _b_name(oofs), "C_stats_only_pl") if n in oofs]
    if len(names) < 2:
        return pd.DataFrame(), pd.DataFrame()
    base = None
    for n in names:
        f = oofs[n][["season", "match_id", "player_id", "brownlow_votes", "p3", "expected_votes"]].rename(columns={"p3": f"p3_{n}", "expected_votes": f"ev_{n}"})
        base = f if base is None else base.merge(f.drop(columns=["season", "brownlow_votes"]), on=["match_id", "player_id"])
    # match-level: top pick disagreement + max pairwise |P3| gap for the consensus top pick
    rows = []
    for (s, mid), g in base.groupby(["season", "match_id"]):
        tops = {n: g.loc[g[f"p3_{n}"].idxmax(), "player_id"] for n in names}
        a3 = g.loc[g["brownlow_votes"] == 3, "player_id"]
        if len(a3) != 1:
            continue
        a3 = a3.iloc[0]
        mean_p3 = g[[f"p3_{n}" for n in names]].mean(axis=1)
        cons = g.loc[mean_p3.idxmax(), "player_id"]
        spread = float(g[[f"p3_{n}" for n in names]].max(axis=1).max() - g.loc[mean_p3.idxmax(), [f"p3_{n}" for n in names]].min())
        rows.append({"season": int(s), "match_id": mid, "n_distinct_top_picks": len(set(tops.values())), "consensus_pick_correct": cons == a3,
                     "any_correct": a3 in tops.values(), "all_correct": all(v == a3 for v in tops.values()),
                     "p3_spread_on_consensus_pick": spread, "max_p3_consensus": float(mean_p3.max()),
                     **{f"correct_{n}": tops[n] == a3 for n in names}})
    mt = pd.DataFrame(rows)
    mt["agreement"] = np.where(mt["n_distinct_top_picks"] == 1, "all agree", np.where(mt["n_distinct_top_picks"] == 2, "2 vs 1", "all differ"))
    mt["spread_bin"] = pd.cut(mt["p3_spread_on_consensus_pick"], [-0.01, 0.1, 0.2, 0.35, 1.0], labels=["<0.10", "0.10-0.20", "0.20-0.35", ">0.35"]).astype(str)
    summ = []
    for key in ("agreement", "spread_bin"):
        for scope, sub in (("all", mt), ("recent (2022-2026)", mt[mt["season"] >= 2022])):
            for k, g in sub.groupby(key):
                summ.append({"dimension": key, "bin": k, "scope": scope, "n_matches": len(g), "consensus_correct": g["consensus_pick_correct"].mean(), "any_model_correct": g["any_correct"].mean(),
                             **{f"{CANDIDATES[n]}_correct": g[f"correct_{n}"].mean() for n in names}, "share_of_matches": len(g) / len(sub)})
    # season-level: |EV gap| between A and B vs error
    sp = base.groupby(["season", "player_id"]).agg(actual=("brownlow_votes", "sum"), **{f"ev_{n}": (f"ev_{n}", "sum") for n in names}).reset_index()
    bn = _b_name(oofs)
    if "A_structural_pl" in names and bn in names:
        sp = sp.rename(columns={f"ev_{bn}": "ev_B_performance_xgb_rank"})
        sp["gap"] = (sp["ev_A_structural_pl"] - sp["ev_B_performance_xgb_rank"]).abs()
        sp["gap_bin"] = pd.cut(sp["gap"], [-0.01, 1, 3, 6, 100], labels=["0-1", "1-3", "3-6", "6+"]).astype(str)
        sp["err_A"] = (sp["ev_A_structural_pl"] - sp["actual"]).abs(); sp["err_B"] = (sp["ev_B_performance_xgb_rank"] - sp["actual"]).abs()
        sp["err_mean"] = ((sp["ev_A_structural_pl"] + sp["ev_B_performance_xgb_rank"]) / 2 - sp["actual"]).abs()
        for k, g in sp.groupby("gap_bin"):
            summ.append({"dimension": "season_ev_gap_A_vs_B", "bin": k, "scope": "all", "n_players": len(g), "mae_A": g["err_A"].mean(), "mae_B": g["err_B"].mean(),
                         "mae_mean_of_two": g["err_mean"].mean(), "A_closer_share": (g["err_A"] < g["err_B"]).mean()})
    return mt, pd.DataFrame(summ)


def bias_study(oofs: dict[str, pd.DataFrame], feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ctx = feat[["match_id", "player_id", "role", "is_win", "absolute_margin", "is_close_game", "is_blowout", "team_id"]].copy()
    for name, df in oofs.items():
        pm = M.per_match_table(df)
        a3 = pm.merge(ctx.rename(columns={"player_id": "actual_3"}), on=["match_id", "actual_3"], how="left")
        a3["margin_bucket"] = pd.cut(a3["absolute_margin"], [-1, 12, 24, 48, 100, 999], labels=["0-12", "12-24", "24-48", "48-100", "100+"]).astype(str)
        a3["losing_team_3voter"] = a3["is_win"] == 0
        for dim in ("role", "margin_bucket", "losing_team_3voter"):
            for scope, sub in (("all", a3), ("recent (2022-2026)", a3[a3["season"] >= 2022])):
                for k, g in sub.groupby(dim):
                    rows.append({"model": CANDIDATES.get(name, name), "dimension": dim, "group": str(k), "scope": scope, "n_3voters": len(g),
                                 "hit_rate": g["correct_3"].mean(), "mean_p3_on_3voter": g["p3_of_actual_3"].mean(), "share_of_3voters": len(g) / len(sub)})
        # season-level EV bias by role
        t = df.merge(ctx[["match_id", "player_id", "role"]], on=["match_id", "player_id"], how="left")
        st = t.groupby(["season", "player_id"]).agg(pred=("expected_votes", "sum"), actual=("brownlow_votes", "sum"), role=("role", lambda s: s.value_counts().idxmax())).reset_index()
        st = st[(st["actual"] >= 1) | (st["pred"] >= 1)]
        for scope, sub in (("all", st), ("recent (2022-2026)", st[st["season"] >= 2022])):
            for k, g in sub.groupby("role"):
                rows.append({"model": CANDIDATES.get(name, name), "dimension": "season_ev_bias_by_role", "group": str(k), "scope": scope, "n_players": len(g),
                             "mean_error": (g["pred"] - g["actual"]).mean(), "mae": (g["pred"] - g["actual"]).abs().mean()})
    return pd.DataFrame(rows)


def forensic_by_season(oofs: dict[str, pd.DataFrame], feat: pd.DataFrame) -> pd.DataFrame:
    """2026 hypotheses tested year by year: ruck EV bias, defender 3-voter hit rate."""
    rows = []
    ctx = feat[["match_id", "player_id", "role"]]
    for name, df in oofs.items():
        t = df.merge(ctx, on=["match_id", "player_id"], how="left")
        st = t.groupby(["season", "player_id"]).agg(pred=("expected_votes", "sum"), actual=("brownlow_votes", "sum"), role=("role", lambda s: s.value_counts().idxmax())).reset_index()
        st = st[(st["actual"] >= 1) | (st["pred"] >= 1)]
        pm = M.per_match_table(df).merge(ctx.rename(columns={"player_id": "actual_3"}), on=["match_id", "actual_3"], how="left")
        for s in sorted(df["season"].unique()):
            r = {"model": CANDIDATES.get(name, name), "season": int(s)}
            g = st[st["season"] == s]
            for role in ("RUCK", "MIDFIELDER", "KEY_DEFENDER", "MEDIUM_DEFENDER", "KEY_FORWARD", "MEDIUM_FORWARD"):
                gg = g[g["role"] == role]
                r[f"{role}_ev_bias"] = float((gg["pred"] - gg["actual"]).mean()) if len(gg) else np.nan; r[f"{role}_n"] = int(len(gg))
            p = pm[pm["season"] == s]
            for role in ("MIDFIELDER", "KEY_DEFENDER", "MEDIUM_DEFENDER", "RUCK"):
                gg = p[p["role"] == role]
                r[f"{role}_3voter_hit"] = float(gg["correct_3"].mean()) if len(gg) else np.nan; r[f"{role}_3voters"] = int(len(gg))
            rows.append(r)
    return pd.DataFrame(rows)


def error_lab(oofs: dict[str, pd.DataFrame], feat: pd.DataFrame) -> pd.DataFrame:
    names = [n for n in ("A_structural_pl", _b_name(oofs), "C_stats_only_pl") if n in oofs]
    base = feat[["season", "match_id", "player_id", "player_name", "team_id", "brownlow_votes"] + ERROR_LAB_FEATURES].copy()
    base = base[base["season"] >= min(int(oofs[n]["season"].min()) for n in names)]
    for n in names:
        d = oofs[n][["match_id", "player_id", "p3", "p2", "p1", "expected_votes"]].copy()
        d["rank_in_match"] = d.groupby("match_id")["p3"].rank(ascending=False, method="first")
        short = {"A_structural_pl": "A", "B_performance_xgb_rank": "B", "B_performance_xgb_rank_tuned": "B", "C_stats_only_pl": "C"}[n]
        d = d.rename(columns={c: f"{c}_{short}" for c in ("p3", "p2", "p1", "expected_votes", "rank_in_match")})
        base = base.merge(d, on=["match_id", "player_id"], how="left")
    return base


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    oofs = {n: load_oof(n) for n in CANDIDATES}
    oofs = {n: d for n, d in oofs.items() if d is not None}
    feat = pd.read_parquet(FEAT, columns=["season", "match_id", "player_id", "player_name", "team_id", "brownlow_votes"] + ERROR_LAB_FEATURES)
    feat["player_id"] = feat["player_id"].astype(str)
    summary = {"models": list(oofs)}
    by_season, pooled = comparison_tables(oofs)
    by_season.to_csv(OUT / "comparison_by_season.csv", index=False); pooled.to_csv(OUT / "comparison_pooled.csv", index=False)
    # recent-8 window (the deployed Production window) as a second view
    oofs8 = {n: load_oof(n, window="recent8") for n in CANDIDATES}
    oofs8 = {n: d for n, d in oofs8.items() if d is not None and len(d)}
    if oofs8:
        b8, p8 = comparison_tables(oofs8)
        b8.to_csv(OUT / "comparison_by_season_recent8.csv", index=False); p8.to_csv(OUT / "comparison_pooled_recent8.csv", index=False)
        summary["pooled_recent8"] = p8.to_dict("records")
    pairs = []
    for ref in ("A_structural_pl", "baseline_phase4_pl_legacy"):
        if ref in oofs:
            for other in oofs:
                if other != ref:
                    for metric in ("correct_3", "log_loss_p3", "exact_321"):
                        pairs.append({"a": ref, "b": other, **paired_bootstrap(oofs[ref], oofs[other], metric)})
    pd.DataFrame(pairs).to_csv(OUT / "paired_bootstrap_vs_A.csv", index=False)
    calibration_study({n: oofs[n] for n in oofs if n != "baseline_phase4_pl_legacy"}).to_csv(OUT / "calibration_study.csv", index=False)
    mt, ds = disagreement_study(oofs)
    mt.to_csv(OUT / "disagreement_matches.csv", index=False); ds.to_csv(OUT / "disagreement_summary.csv", index=False)
    bias_study(oofs, feat).to_csv(OUT / "bias_study.csv", index=False)
    forensic_by_season(oofs, feat).to_csv(OUT / "forensic_by_season.csv", index=False)
    el = error_lab(oofs, feat); el.to_parquet(OUT / "error_lab.parquet", index=False)
    # ensemble research
    comps = {n: oofs[n] for n in ("A_structural_pl", _b_name(oofs), "C_stats_only_pl") if n in oofs}
    if len(comps) >= 2:
        learned, equal, w = walk_forward_ensemble(comps, sorted(oofs["A_structural_pl"]["season"].unique()))
        ens_rows = []
        for label, df in (("Learned ensemble (A+B+C)", learned), ("Equal-weight ensemble", equal), *[(CANDIDATES[n], comps[n][comps[n]["season"].isin(learned["season"].unique())]) for n in comps]):
            for s, g in df.groupby("season"):
                mm = M.match_metrics(g); sm = M.season_metrics(g)["by_season"][0]
                ens_rows.append({"model": label, "season": int(s), "correct_3": mm["correct_3"], "log_loss_p3": mm["log_loss_p3"], "ece_p3": mm["ece_p3"], "exact_321": mm["exact_321"], "season_mae": sm["season_mae"], "spearman": sm["spearman"]})
        er = pd.DataFrame(ens_rows); er.to_csv(OUT / "ensemble_by_season.csv", index=False); w.to_csv(OUT / "ensemble_weights.csv", index=False)
        learned[["season", "match_id", "player_id", "brownlow_votes", "p3", "p2", "p1", "p0", "expected_votes"]].to_parquet(OOF / "ENS_learned_ABC.parquet", index=False)
        summary["ensemble_pooled"] = er.groupby("model")[["correct_3", "log_loss_p3", "ece_p3", "exact_321", "season_mae", "spearman"]].mean().round(4).to_dict("index")
        pv = er.pivot(index="season", columns="model", values="log_loss_p3"); pa = er.pivot(index="season", columns="model", values="correct_3")
        comp = [c for c in pv.columns if "ensemble" not in c.lower()]
        summary["ensemble_seasons_learned_beats_best_single_logloss"] = int((pv["Learned ensemble (A+B+C)"] < pv[comp].min(axis=1)).sum())
        summary["ensemble_seasons_beats_each_component_logloss"] = {c: int((pv["Learned ensemble (A+B+C)"] < pv[c]).sum()) for c in comp}
        summary["ensemble_seasons_beats_each_component_correct3"] = {c: int((pa["Learned ensemble (A+B+C)"] > pa[c]).sum()) for c in comp}
        summary["ensemble_n_seasons"] = int(er["season"].nunique())
    summary["pooled"] = pooled.to_dict("records")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    s = run()
    print(json.dumps({k: v for k, v in s.items() if k != "pooled"}, indent=1, default=str))
    sys.exit(0)
