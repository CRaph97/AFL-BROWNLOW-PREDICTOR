"""
Read-only loader layer for the Neds/PointsBet Brownlow Betting Opportunities
page. Mirrors this project's established pattern (dashboard/betting_data.py,
dashboard/external_data.py): pages never compute a probability, edge, or
classification themselves -- everything here is a load of an already-built
file from scripts/refresh_brownlow_odds.py's output
(data/betting/processed/*), never a live scrape.

Never scrapes on import or on any function call -- the refresh pipeline
(scripts/refresh_brownlow_odds.py) is a separate, manually-run step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "betting" / "processed"


@st.cache_data
def load_opportunities() -> pd.DataFrame:
    path = PROCESSED / "priced_opportunities.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_combinations() -> pd.DataFrame:
    path = PROCESSED / "combinations.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_price_comparison() -> pd.DataFrame:
    path = PROCESSED / "price_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_refresh_summary() -> dict:
    path = PROCESSED / "refresh_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


CONFIDENCE_SECTIONS = [
    ("HIGH_CONFIDENCE_WHEELO_CONFIRMED", "High Confidence + Wheelo Confirmed"),
    ("HIGH_CONFIDENCE_WHEELO_NEUTRAL", "High Confidence / Wheelo Neutral"),
    ("MEDIUM_CONFIDENCE", "Medium Confidence"),
    ("HIGH_RISK_HIGH_REWARD", "High Risk / High Reward"),
    ("MODEL_DISAGREEMENT", "Model Disagreement"),
]
