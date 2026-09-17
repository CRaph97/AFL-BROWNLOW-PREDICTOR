"""
Back-compat shim. The round-normalization logic that was originally written here for 2026
only (see docs/2026_ROUND_INTEGRITY_AUDIT.md) has been generalized into
src/data/round_normalization.py, after the same afltables mislabeling was independently
confirmed for 2024-2025 too (docs/ROUND_NORMALIZATION.md). This module now just binds the
general functions to season=2026 so existing callers (build_2026_extension.py,
scripts/build_2026_match_identity_audit.py) keep working unchanged -- same function names,
same single-argument signatures, same return values for every 2026 input.

`fix_match_id` is re-exported as-is: it already parses the season out of the match_id
string itself, so the same implementation works for 2026 and every other season.
"""
from src.data.round_normalization import (
    FIRST_SHIFTED_SEASON,
    fix_match_id,
    official_round as _official_round_general,
    official_round_label as _official_round_label_general,
)

OPENING_ROUND_RAW = "1"
_SEASON = 2026


def official_round(raw_round) -> int:
    return _official_round_general(raw_round, _SEASON)


def official_round_label(off_round: int) -> str:
    return _official_round_label_general(off_round, _SEASON)
