"""
Experiment registry: append-only JSONL at data/experiments/registry.jsonl.
Each record: experiment_id, timestamp, git commit, model family, features,
training/validation seasons, hyperparameters, metrics by season, aggregate
metrics, notes, status (candidate | promoted | rejected | baseline), reason.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG_PATH = ROOT / "data" / "experiments" / "registry.jsonl"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def experiment_id(name: str, config: dict) -> str:
    h = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{name}__{h}"


def record(name: str, model_family: str, features: list[str], feature_families: list[str], train_seasons: list[int],
           validation_seasons: list[int], hyperparameters: dict, metrics_by_season: list[dict], aggregate: dict,
           notes: str = "", status: str = "candidate", reason: str = "", window: str = "expanding") -> dict:
    config = {"model_family": model_family, "features": sorted(features), "hyperparameters": hyperparameters,
              "validation_seasons": validation_seasons, "window": window}
    rec = {
        "experiment_id": experiment_id(name, config), "name": name, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "model_family": model_family, "window": window, "n_features": len(features),
        "feature_families": feature_families, "features": features, "train_seasons": train_seasons,
        "validation_seasons": validation_seasons, "hyperparameters": hyperparameters,
        "metrics_by_season": metrics_by_season, "aggregate": aggregate, "notes": notes, "status": status, "reason": reason,
    }
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REG_PATH.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def load() -> pd.DataFrame:
    if not REG_PATH.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in REG_PATH.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def update_status(experiment_id_: str, status: str, reason: str) -> None:
    rows = [json.loads(l) for l in REG_PATH.read_text().splitlines() if l.strip()]
    for r in rows:
        if r["experiment_id"] == experiment_id_:
            r["status"], r["reason"] = status, reason
    REG_PATH.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
