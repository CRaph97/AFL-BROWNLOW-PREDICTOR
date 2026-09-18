"""
Streamlit-facing loader layer for the External Benchmarks section. Thin
cached wrappers over src/external/ -- no identity/aggregation logic lives
here, only loading + light reshaping for display. Pages must import from
here, not re-implement a source join inline (mirrors dashboard/data.py's
established convention).

Read-only against Production/Objective; cannot mutate either model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "external" / "processed"
REPORTS = ROOT / "reports"


@st.cache_data
def load_source_status() -> dict:
    path = PROCESSED / "source_status.json"
    if not path.exists():
        return {"retrieved_at": None, "sources": {}}
    return json.loads(path.read_text())


@st.cache_data
def load_external_overview() -> pd.DataFrame:
    path = PROCESSED / "external_overview.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_wheelo_season() -> pd.DataFrame:
    path = PROCESSED / "wheelo_season.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype(str)
    return df


@st.cache_data
def load_wheelo_match_level() -> pd.DataFrame:
    path = PROCESSED / "wheelo_match_level.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype("Int64").astype(str)
    return df


@st.cache_data
def load_espn_season() -> pd.DataFrame:
    path = PROCESSED / "espn_season.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype("Int64").astype(str)
    return df


@st.cache_data
def load_betfair_season() -> pd.DataFrame:
    path = PROCESSED / "betfair_season.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype("Int64").astype(str)
    return df


@st.cache_data
def load_order_scenarios() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "2026_order_scenarios.csv")


@st.cache_data
def player_round_by_round_external(player_id: str) -> pd.DataFrame:
    """Round-by-round Wheelo EV/P3 for one player, alongside Production and
    Objective match-level EV for the same rounds -- all three already-real,
    joined on (round, player_id) only, no value recomputed."""
    player_id = str(player_id)
    wheelo = load_wheelo_match_level()
    wheelo = wheelo[wheelo["player_id"] == player_id][["round", "wheelo_ev", "wheelo_p3_pct", "match_label"]]

    prod = pd.read_csv(REPORTS / "2026_predicted_votes.csv")
    prod["player_id"] = prod["player_id"].astype(str)
    prod = prod[prod["player_id"] == player_id][["round", "expected_votes"]].rename(
        columns={"expected_votes": "production_ev"}
    )

    obj = pd.read_csv(REPORTS / "2026_objective_votes.csv")
    obj["player_id"] = obj["player_id"].astype(str)
    obj = obj[obj["player_id"] == player_id][["round", "expected_votes"]].rename(
        columns={"expected_votes": "objective_ev"}
    )

    merged = wheelo.merge(prod, on="round", how="outer").merge(obj, on="round", how="outer")
    return merged.sort_values("round").reset_index(drop=True)
