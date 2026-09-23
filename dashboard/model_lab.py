"""Loaders for the 2027 Model Lab page. Read-only over data/experiments/."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "experiments"
AN = EXP / "analysis"


def available() -> bool:
    return (AN / "comparison_pooled.csv").exists()


@st.cache_data
def csv(name: str) -> pd.DataFrame:
    p = AN / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def metrics_csv(name: str) -> pd.DataFrame:
    p = EXP / "metrics" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def registry() -> pd.DataFrame:
    p = EXP / "registry.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        agg = r.get("aggregate", {})
        first = next(iter(agg.values()), {}) if isinstance(agg, dict) else {}
        out.append({"experiment_id": r["experiment_id"], "name": r["name"], "timestamp": r["timestamp"][:19], "git_commit": r.get("git_commit"),
                    "model_family": r["model_family"], "window": r.get("window"), "n_features": r["n_features"], "families": ", ".join(r.get("feature_families", [])),
                    "validation_seasons": f"{min(r['validation_seasons'])}-{max(r['validation_seasons'])}", "status": r["status"], "reason": r.get("reason", ""),
                    "correct_3": first.get("correct_3_mean"), "exact_321": first.get("exact_321_mean"), "log_loss_p3": first.get("log_loss_p3_mean"),
                    "season_mae": first.get("season_mae_mean"), "spearman": first.get("spearman_mean"), "notes": r.get("notes", "")})
    return pd.DataFrame(out)


@st.cache_data
def error_lab() -> pd.DataFrame:
    p = AN / "error_lab.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def summary() -> dict:
    p = AN / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def ablation_table() -> pd.DataFrame:
    rows = []
    full = metrics_csv("ABL_full")
    if full.empty:
        return pd.DataFrame()
    fm = full.set_index("test_season")
    for p in sorted((EXP / "metrics").glob("ABL_*.csv")):
        if p.name.endswith("_extras.json") or p.stem == "ABL_full":
            continue
        m = pd.read_csv(p).set_index("test_season")
        common = m.index.intersection(fm.index)
        if len(common) == 0:
            continue
        d = m.loc[common]; f = fm.loc[common]
        rows.append({"variant": p.stem.replace("ABL_", ""), "n_seasons": len(common),
                     "d_correct_3": (d["correct_3"] - f["correct_3"]).mean(), "d_correct_3_recent": (d["correct_3"] - f["correct_3"])[common >= 2022].mean(),
                     "d_log_loss": (d["log_loss_p3"] - f["log_loss_p3"]).mean(), "d_log_loss_recent": (d["log_loss_p3"] - f["log_loss_p3"])[common >= 2022].mean(),
                     "d_season_mae": (d["season_mae"] - f["season_mae"]).mean(), "d_exact_321": (d["exact_321"] - f["exact_321"]).mean(),
                     "seasons_worse_correct3": int(((d["correct_3"] - f["correct_3"]) < 0).sum()), "sd_d_correct_3": (d["correct_3"] - f["correct_3"]).std(ddof=0),
                     "d_correct_3_2026": float((d["correct_3"] - f["correct_3"]).get(2026, float("nan")))})
    return pd.DataFrame(rows)
