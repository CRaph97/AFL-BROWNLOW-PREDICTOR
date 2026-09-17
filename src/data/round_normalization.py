"""
Generalized AFL round-label normalization.

afltables' raw `Round` column (data/raw/fitzroy_data/afldata.rda) is not always the AFL's
official round number. This was first found and fixed for 2026 only (see
docs/2026_ROUND_INTEGRITY_AUDIT.md), then audited for 2023-2025 as a follow-up
(docs/ROUND_NORMALIZATION.md). This module is the single source of truth for both.

The shift is NOT a uniform "every season with an Opening Round" rule. It was verified
directly against footywire's independently-sourced Round column (player_stats.rda),
matched by calendar date, for every round of every season 2023-2026:

  - 2023: footywire's "Round N" == afltables' raw round N for every N (1-24). No shift.
    (2023's Opening Round was itself labelled "Round 1" by both sources -- the AFL did not
    yet treat it as unnumbered that year.)
  - 2024, 2025, 2026: footywire labels the unnumbered Opening Round "Round 0" and every
    subsequent round one less than afltables' raw label. afltables raw round 1 -> official
    "Opening Round" (represented as 0); afltables raw round N (N>=2) -> official round N-1.
  - 2022 and earlier: no Opening Round concept existed; afltables' round 1 is a full
    18-team round and already equals the official round number. No shift.

Officially-numbered round is an integer: 0 for the Opening Round, 1-24 for the rest of the
season. This keeps every existing `.astype(int)` sort call in the codebase working
unchanged. `round` is descriptive metadata only -- grep-audited (see
docs/2026_ROUND_INTEGRITY_AUDIT.md and docs/ROUND_NORMALIZATION.md) to confirm it is never
used as a join key, sort-then-index key, or model feature anywhere in src/ or dashboard/.
"""
import re

FIRST_SHIFTED_SEASON = 2024  # 2023 independently confirmed unaffected -- see docs/ROUND_NORMALIZATION.md
OPENING_ROUND_RAW = "1"


def official_round(raw_round, season) -> int:
    season = int(season)
    raw_round = str(raw_round)
    n = int(raw_round)
    if season < FIRST_SHIFTED_SEASON:
        return n
    return 0 if raw_round == OPENING_ROUND_RAW else n - 1


def official_round_label(off_round: int, season) -> str:
    if int(season) < FIRST_SHIFTED_SEASON:
        return f"Round {off_round}"
    return "Opening Round" if off_round == 0 else f"Round {off_round}"


_MATCH_ID_PREFIX_RE = re.compile(r"^(\d{4})_R(\d+)_")


def fix_match_id(match_id: str) -> str:
    """Rewrites the season/round prefix of a '{season}_R{raw_round}_{home}_v_{away}_{date}'
    match_id to use the official round for that season (parsed from the match_id itself).
    Leaves the rest of the identifier (teams, date) untouched -- match identity was never
    actually broken, only this cosmetic round substring."""
    m = _MATCH_ID_PREFIX_RE.match(match_id)
    if not m:
        return match_id
    season, raw_round = m.group(1), m.group(2)
    new_prefix = f"{season}_R{official_round(raw_round, int(season))}_"
    return _MATCH_ID_PREFIX_RE.sub(new_prefix, match_id, count=1)
