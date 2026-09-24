"""
Machine-readable guard against the contaminated reputation artefacts
(data/canonical/contaminated_artefacts.json). Any 2027 feature, model or
evaluation pipeline calls `assert_not_denied` on the columns it is about to
consume; the experiment driver refuses denied features unless the caller is
the historical-record ablation explicitly named in the denylist.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DENYLIST_PATH = ROOT / "data" / "canonical" / "contaminated_artefacts.json"


def load() -> dict:
    return json.loads(DENYLIST_PATH.read_text())


def denied_feature_columns() -> set[str]:
    return set(load()["denied_feature_columns"])


def denied_output_columns(path: str) -> set[str]:
    return set(load()["denied_output_columns"].get(path, []))


class DeniedFeatureError(ValueError):
    pass


def assert_not_denied(columns, context: str = "") -> None:
    bad = sorted(set(columns) & denied_feature_columns())
    if bad:
        raise DeniedFeatureError(f"{context or 'pipeline'} attempted to consume contaminated columns {bad}; see {DENYLIST_PATH.relative_to(ROOT)}")


def assert_output_columns_not_consumed(path: str, columns) -> None:
    bad = sorted(set(columns) & denied_output_columns(path))
    if bad:
        raise DeniedFeatureError(f"{path}: columns {bad} are VOID / contaminated and must not be consumed; see {DENYLIST_PATH.relative_to(ROOT)}")
