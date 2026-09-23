"""
Loaders for the 2026 post-Brownlow evaluation page (pages/33_2026_Evaluation.py).

Everything is read from data/evaluation/2026/, built offline by
`python -m src.evaluation.build_2026_evaluation`. Nothing here recomputes a
metric, settles a bet or touches a model output -- presentation-side reads
only, so the page can never alter the frozen forecasts or the actual-vote
ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation" / "2026"

MODEL_COLOURS = {"Production": "#1f77b4", "Objective": "#ff7f0e", "Wheelo": "#2ca02c", "Bookmaker implied": "#7f7f7f"}


def available() -> bool:
    return (EVAL_DIR / "manifest.json").exists()


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = EVAL_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"player_id": str, "actual_3_player_id": str, "model_top_player_id": str} if name in (
        "season_players", "final_order_top20", "disagreement_cases", "match_table") else None)
    return df


@st.cache_data
def load_json(name: str):
    path = EVAL_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def signed(x, digits: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.{digits}f}"


def display_team(team_id) -> str:
    return "" if team_id is None or pd.isna(team_id) else str(team_id).replace("_", " ").title().replace("Greater Western Sydney", "GWS")
