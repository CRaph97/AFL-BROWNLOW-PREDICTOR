"""
External-source player-identity resolution against the canonical Production +
Objective player set.

This project has a documented history of identity-join bugs (see
docs/2026_ROUND_INTEGRITY_AUDIT.md's sibling incidents and the two
hyphenated-surname/collision fixes in src/data/build_2026_extension.py):
truncating a surname to resolve one source's abbreviation convention can
create a NEW collision against an unrelated teammate who shares the resulting
short key, unless a given-name component is also required.

The internal footywire fix truncated to "the segment after the last hyphen"
because footywire's OWN player_stats.rda genuinely abbreviates one side of a
hyphenated surname to a single initial ("Wanganeen-Milera" -> "W-Milera").
External sources here are different: Wheelo, ESPN, and Betfair all spell
hyphenated surnames IN FULL (confirmed directly against real 2026 data, e.g.
Wheelo's CSV has "Darcy Byrne-Jones", "Nasiah Wanganeen-Milera" verbatim -- no
abbreviation). Truncating here would be unnecessary AND would reintroduce the
exact Byrne-Jones/Jones-style collision this project already fixed once.
Instead this module keeps the FULL surname (letters only, hyphens/apostrophes
stripped, not truncated) and additionally requires the first-name INITIAL to
agree -- mirroring the internal fix's disambiguation principle without
copying its abbreviation-specific truncation, which does not apply here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def _normalise_full_surname(name: str) -> str:
    """Everything after the first whitespace-separated token, letters only,
    lowercased. Handles multi-word surnames (e.g. a historical "Jay Kennedy
    Harris" -- see docs/2026_OBJECTIVE_MODEL.md-adjacent audit work) by taking
    everything after the FIRST token rather than only the last one."""
    parts = str(name).split(maxsplit=1)
    surname = parts[1] if len(parts) > 1 else parts[0]
    return "".join(c for c in surname.lower() if c.isalpha())


def _first_initial(name: str) -> str:
    parts = str(name).split(maxsplit=1)
    first = parts[0] if parts else ""
    letters = "".join(c for c in first.lower() if c.isalpha())
    return letters[:1]


# Wheelo's short team codes -> this project's canonical team_id (confirmed
# against every one of the 18 real 2026 clubs appearing in the Wheelo CSV).
WHEELO_TEAM_MAP = {
    "Adel": "adelaide", "Bris": "brisbane_lions", "Carl": "carlton",
    "Coll": "collingwood", "Ess": "essendon", "Frem": "fremantle",
    "GC": "gold_coast", "GWS": "greater_western_sydney", "Geel": "geelong",
    "Haw": "hawthorn", "Melb": "melbourne", "NM": "north_melbourne",
    "Port": "port_adelaide", "Rich": "richmond", "St K": "st_kilda",
    "Syd": "sydney", "WB": "western_bulldogs", "WC": "west_coast",
}

# ESPN/Betfair use standard 3-4 letter club abbreviations in their tables.
ESPN_BETFAIR_TEAM_MAP = {
    "COLL": "collingwood", "GEE": "geelong", "WB": "western_bulldogs",
    "ADEL": "adelaide", "MELB": "melbourne", "CARL": "carlton",
    "BRIS": "brisbane_lions", "GC": "gold_coast", "STK": "st_kilda",
    "PORT": "port_adelaide", "SYD": "sydney", "HAW": "hawthorn",
    "GWS": "greater_western_sydney", "RICH": "richmond", "ESS": "essendon",
    "FRE": "fremantle", "WCE": "west_coast", "NM": "north_melbourne",
    "BL": "brisbane_lions", "GCS": "gold_coast",
    "ADE": "adelaide", "MEL": "melbourne", "PA": "port_adelaide",
}


def load_canonical_players() -> pd.DataFrame:
    """The canonical 2026 player identity set: the union of every player_id
    appearing in the Production or Objective leaderboards, with player_id
    normalised to str (Production stores it as int64, Objective as str --
    same underlying values, confirmed in dashboard.data.load_dual_model_comparison,
    reproduced here independently so src/external does not depend on the
    Streamlit-layer dashboard/ package)."""
    prod = pd.read_csv(REPORTS / "2026_leaderboard.csv")[["player_id", "player_name", "team_id"]].copy()
    obj = pd.read_csv(REPORTS / "2026_objective_leaderboard.csv")[["player_id", "player_name", "team_id"]].copy()
    prod["player_id"] = prod["player_id"].astype(str)
    obj["player_id"] = obj["player_id"].astype(str)
    combined = pd.concat([prod, obj], ignore_index=True).drop_duplicates(subset="player_id")

    # Exclude unresolved-identity placeholder rows (player_id starting
    # "NOID2026_", an existing convention documented in dashboard/data.py for
    # 2026 CORE rows whose real afltables provider id never resolved). These
    # are NOT genuinely distinct players -- discovered here as a real,
    # pre-existing bug: Objective's leaderboard build assigns a UNIQUE
    # NOID2026_* id per unresolved row instead of collapsing them, so a name
    # like "Charlie Cameron" or "Jack Graham" appears dozens of times under
    # different synthetic ids. Left unfixed (Objective is out of scope for
    # this task) but excluded here so they don't manufacture false
    # surname/team collisions against real, resolvable external players.
    combined = combined[~combined["player_id"].str.startswith("NOID")].copy()
    combined["surname_key"] = combined["player_name"].map(_normalise_full_surname)
    combined["first_initial"] = combined["player_name"].map(_first_initial)
    return combined.reset_index(drop=True)


def resolve_external_players(
    external: pd.DataFrame,
    name_col: str,
    team_col: str,
    canonical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve external rows to canonical player_id via (surname_key,
    first_initial, team_id). Returns `external` with new columns:
    player_id (NaN if unresolved), match_status ("resolved" / "unresolved" /
    "ambiguous"). Never guesses: a canonical key with more than one candidate
    player_id for the same team is left unresolved (ambiguous), never forced
    to whichever appears first."""
    if canonical is None:
        canonical = load_canonical_players()

    ext = external.copy()
    ext["surname_key"] = ext[name_col].map(_normalise_full_surname)
    ext["first_initial"] = ext[name_col].map(_first_initial)

    # Flag canonical-side ambiguity up front: two different real players who
    # share (surname_key, first_initial, team_id) -- e.g. this project's own
    # Chad Warner / Corey Warner internal collision does NOT recur here since
    # their first names differ, but the guard is kept generic and re-derived
    # independently rather than assumed safe.
    dupe_keys = (
        canonical.groupby(["surname_key", "first_initial", "team_id"])["player_id"]
        .nunique()
    )
    ambiguous_keys = set(dupe_keys[dupe_keys > 1].index)

    lookup = canonical.drop_duplicates(subset=["surname_key", "first_initial", "team_id"]).set_index(
        ["surname_key", "first_initial", "team_id"]
    )["player_id"]

    def _match(row):
        key = (row["surname_key"], row["first_initial"], row[team_col])
        if key in ambiguous_keys:
            return pd.NA, "ambiguous"
        if key in lookup.index:
            return lookup.loc[key], "resolved"
        return pd.NA, "unresolved"

    matches = ext.apply(_match, axis=1, result_type="expand")
    ext["player_id"] = matches[0]
    ext["match_status"] = matches[1]
    return ext.drop(columns=["surname_key", "first_initial"])


def load_canonical_player_rounds() -> pd.DataFrame:
    """Per-ROUND canonical identity (round, surname_key, first_initial,
    team_id) -> player_id, built from Production's and Objective's own
    already-resolved match-level files. This is strictly finer-grained than
    load_canonical_players()'s season-level table, and resolves genuine
    same-team/same-initial collisions (e.g. Chad Warner vs Corey Warner, both
    real Sydney players) that a season-total-only key cannot: Production's
    predicted_votes.csv already nulls out any round where such a collision
    was ambiguous at the source (see src/data/build_2026_extension.py's
    ambiguity guard), so a player_id appearing there for a given round is
    real, already-validated ground truth for "who actually had this identity
    key in this specific round" -- not re-derived or guessed here."""
    prod = pd.read_csv(REPORTS / "2026_predicted_votes.csv")[["round", "player_id", "player_name", "team_id"]].copy()
    obj = pd.read_csv(REPORTS / "2026_objective_votes.csv")[["round", "player_id", "player_name", "team_id"]].copy()
    prod["player_id"] = prod["player_id"].astype(str)
    obj["player_id"] = obj["player_id"].astype(str)
    obj = obj[~obj["player_id"].str.startswith("NOID")]
    combined = pd.concat([prod, obj], ignore_index=True).drop_duplicates(subset=["round", "player_id"])
    combined["surname_key"] = combined["player_name"].map(_normalise_full_surname)
    combined["first_initial"] = combined["player_name"].map(_first_initial)
    return combined.reset_index(drop=True)


def resolve_external_match_players(
    external: pd.DataFrame,
    name_col: str,
    team_col: str,
    round_col: str,
    canonical_rounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match-level counterpart to resolve_external_players(): resolves using
    (round, surname_key, first_initial, team_id) instead of a season-total
    key, so a genuine same-team/same-initial collision is disambiguated
    per-round using already-validated internal data rather than being left
    ambiguous for the whole season. Falls back to "unresolved" (never a
    guess) for any external row whose round has no canonical match at all."""
    if canonical_rounds is None:
        canonical_rounds = load_canonical_player_rounds()

    ext = external.copy()
    ext["surname_key"] = ext[name_col].map(_normalise_full_surname)
    ext["first_initial"] = ext[name_col].map(_first_initial)

    dupe_keys = (
        canonical_rounds.groupby(["round", "surname_key", "first_initial", "team_id"])["player_id"]
        .nunique()
    )
    ambiguous_keys = set(dupe_keys[dupe_keys > 1].index)
    lookup = canonical_rounds.drop_duplicates(
        subset=["round", "surname_key", "first_initial", "team_id"]
    ).set_index(["round", "surname_key", "first_initial", "team_id"])["player_id"]

    def _match(row):
        key = (row[round_col], row["surname_key"], row["first_initial"], row[team_col])
        if key in ambiguous_keys:
            return pd.NA, "ambiguous"
        if key in lookup.index:
            return lookup.loc[key], "resolved"
        return pd.NA, "unresolved"

    matches = ext.apply(_match, axis=1, result_type="expand")
    ext["player_id"] = matches[0]
    ext["match_status"] = matches[1]
    return ext.drop(columns=["surname_key", "first_initial"])
