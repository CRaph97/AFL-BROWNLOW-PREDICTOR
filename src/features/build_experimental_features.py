"""
Phase 4, Model 5 support: real, fitted EXPERIMENTAL game-state features built
from the torpdata chains feed (2021-2025), using the corrected chain-level
scoring logic validated in Phase 3 (docs/EXPLORATORY_ANALYSIS.md §10) and the
>=4-periods completeness filter from the Phase 4 event-data investigation
(docs/EVENT_DATA_2021_AUDIT.md addendum / PHASE4 section A4).

For every match:
  1. Reconstruct the scoring timeline (goals + chain-level behinds, corrected
     team attribution).
  2. Assign every event (scoring or not) a "score state entering this moment"
     via an as-of merge on (period, period_seconds) against the scoring
     timeline (state BEFORE this event, i.e. the leverage the player faced
     when they acted, not after their own action changed it).
  3. Compute `simple_leverage` (closeness x time-elapsed, unfit, from
     event_feasibility.py) for every event.
  4. Aggregate, per player-match: leverage-weighted disposal volume
     (Kick + Handball events), leverage-weighted scoring involvement
     (Goal + Behind by that player), and the player's mean leverage state
     across all their events (a "how much of this player's game happened in
     a high-leverage moment" measure).

Matches with fewer than 4 distinct periods in the chain data are EXCLUDED
entirely (the two 2024 matches with partial coverage, and any equivalent
case in other seasons -- checked, none found for 2021/2022/2023/2025 at time
of writing) -- see docs/EVENT_DATA_2021_AUDIT.md for the exclusion rule.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .event_feasibility import simple_leverage

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "torpdata_pilot"
PROCESSED_DIR = ROOT / "data" / "processed"

DISPOSAL_DESCRIPTIONS = {"Kick", "Handball", "Ground Kick"}
SCORING_DESCRIPTIONS = {"Goal", "Behind"}


def _match_scoring_timeline(m: pd.DataFrame) -> pd.DataFrame:
    home_id, away_id = m["home_team_id"].iloc[0], m["away_team_id"].iloc[0]

    goal_events = m[m["description"] == "Goal"][["period", "period_seconds", "team_id"]].copy()
    goal_events["points"] = 6
    goal_events["scoring_team"] = goal_events["team_id"]

    chain_first = m.drop_duplicates(subset="chain_number", keep="first")
    is_behind = chain_first["final_state"] == "behind"
    is_rushed = chain_first["final_state"] == "rushed"
    is_rushed_opp = chain_first["final_state"] == "rushedOpp"
    behind_events = chain_first[is_behind | is_rushed | is_rushed_opp][
        ["period", "period_seconds", "final_state", "chain_team_id"]
    ].copy()
    behind_events["points"] = 1
    behind_events["scoring_team"] = np.where(
        behind_events["final_state"] == "rushedOpp",
        np.where(behind_events["chain_team_id"] == home_id, away_id, home_id),
        behind_events["chain_team_id"],
    )

    scoring = pd.concat([
        goal_events[["period", "period_seconds", "scoring_team", "points"]],
        behind_events[["period", "period_seconds", "scoring_team", "points"]],
    ]).sort_values(["period", "period_seconds"]).reset_index(drop=True)

    running_home, running_away = 0, 0
    states = []
    for _, ev in scoring.iterrows():
        if ev["scoring_team"] == home_id:
            running_home += ev["points"]
        elif ev["scoring_team"] == away_id:
            running_away += ev["points"]
        else:
            continue
        states.append({"period": ev["period"], "period_seconds": ev["period_seconds"],
                        "margin_abs": abs(running_home - running_away)})
    states = pd.DataFrame(states) if states else pd.DataFrame(columns=["period", "period_seconds", "margin_abs"])
    # prepend a 0-0 state at the very start of the match
    start = pd.DataFrame([{"period": 1, "period_seconds": 0, "margin_abs": 0}])
    return pd.concat([start, states], ignore_index=True)


def _attach_leverage(m: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    m = m.sort_values(["period", "period_seconds"]).copy()
    timeline = timeline.sort_values(["period", "period_seconds"])
    # merge_asof requires a single sort key across the whole frame; combine period+period_seconds
    # into one monotonically increasing key (1200s per quarter is generous vs actual AFL quarter length)
    m["_t"] = ((m["period"] - 1) * 1200 + m["period_seconds"]).astype("int64")
    timeline["_t"] = ((timeline["period"] - 1) * 1200 + timeline["period_seconds"]).astype("int64")
    merged = pd.merge_asof(m.sort_values("_t"), timeline[["_t", "margin_abs"]].sort_values("_t"),
                            on="_t", direction="backward")
    merged["margin_abs"] = merged["margin_abs"].fillna(0)
    merged["leverage"] = merged.apply(
        lambda r: simple_leverage(r["margin_abs"], r["period"], r["period_seconds"]), axis=1
    )
    return merged


def build_season(season: int) -> pd.DataFrame:
    path = RAW_DIR / f"chains_data_{season}_all.parquet"
    chains = pd.read_parquet(path)

    periods_per_match = chains.groupby("match_id")["period"].nunique()
    complete_matches = periods_per_match[periods_per_match >= 4].index
    excluded = set(periods_per_match.index) - set(complete_matches)
    if excluded:
        print(f"  season {season}: excluding {len(excluded)} incomplete match(es): {excluded}")
    chains = chains[chains["match_id"].isin(complete_matches)]

    rows = []
    for match_id, m in chains.groupby("match_id"):
        timeline = _match_scoring_timeline(m)
        m_lev = _attach_leverage(m, timeline)

        is_disposal = m_lev["description"].isin(DISPOSAL_DESCRIPTIONS)
        is_scoring = m_lev["description"].isin(SCORING_DESCRIPTIONS)
        attributed = m_lev["player_id"].notna()

        disp = m_lev[is_disposal & attributed].groupby("player_id").agg(
            leverage_weighted_disposals=("leverage", "sum"),
            mean_leverage_disposals=("leverage", "mean"),
            n_disposal_events=("leverage", "size"),
        )
        scoring_agg = m_lev[is_scoring & attributed].groupby("player_id").agg(
            leverage_weighted_scoring=("leverage", "sum"),
            n_scoring_events=("leverage", "size"),
        )
        # carry through identity fields needed to join back onto our canonical player_id later
        # (chains uses Champion-Data provider IDs, not afltables IDs -- see module docstring)
        identity = m_lev[attributed].groupby("player_id").agg(
            surname=("player_name_surname", "first"),
            team_id_raw=("team_id", "first"),
            home_team_id=("home_team_id", "first"),
            away_team_id=("away_team_id", "first"),
            home_team_name=("home_team_team_name", "first"),
            away_team_name=("away_team_team_name", "first"),
            date=("date", "first"),
        )
        combined = disp.join(scoring_agg, how="outer").join(identity, how="left").reset_index()
        # NOTE: `match_id` here is the chains feed's own Champion-Data provider ID
        # (e.g. "CD_M20210140101"), NOT our canonical CORE match_id -- resolved to the
        # latter in _join_to_core_player_id() below via (season, team, surname, DATE),
        # since date uniquely pins the match once team+player are known. Kept under a
        # different name to avoid the two being confused before that resolution happens.
        combined["cd_match_id"] = match_id
        combined["season"] = season
        rows.append(combined)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    for c in ["leverage_weighted_disposals", "mean_leverage_disposals", "n_disposal_events",
              "leverage_weighted_scoring", "n_scoring_events"]:
        if c in result.columns:
            result[c] = result[c].fillna(0)
    return result


def _join_to_core_player_id(result: pd.DataFrame) -> pd.DataFrame:
    """Chains data identifies players by Champion-Data provider ID, not our afltables
    numeric player_id -- resolve via normalised surname + team + season, exactly the
    same pattern (and the same documented collision caveat) as build_role_reference.py."""
    team_map = dict(pd.read_csv(ROOT / "config" / "team_mapping.csv")[["source_name", "canonical_team_id"]].values)

    result = result.copy()
    result["team_name"] = np.where(
        result["team_id_raw"] == result["home_team_id"], result["home_team_name"], result["away_team_name"]
    )
    unmapped = set(result["team_name"].dropna().unique()) - set(team_map.keys())
    if unmapped:
        print(f"  WARNING: unmapped team names from chains data, dropping their rows: {unmapped}")
    result["team_id"] = result["team_name"].map(team_map)
    result["surname_key"] = result["surname"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    result["date_str"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")

    core = pd.read_parquet(PROCESSED_DIR / "player_match_core_1984_2025.parquet")
    core_names = core[["season", "team_id", "player_id", "player_name", "match_id", "date"]].drop_duplicates().rename(
        columns={"player_id": "core_player_id", "match_id": "core_match_id"}
    )
    core_names["surname_key"] = core_names["player_name"].str.split().str[-1].str.lower().str.replace(r"[^a-z]", "", regex=True)
    core_names["date_str"] = pd.to_datetime(core_names["date"]).dt.strftime("%Y-%m-%d")

    result = result.rename(columns={"player_id": "cd_player_id"})
    merged = result.merge(
        core_names[["season", "team_id", "surname_key", "date_str", "core_player_id", "core_match_id"]],
        on=["season", "team_id", "surname_key", "date_str"], how="left",
    )

    # a surname+team+season+date key matching >1 core player is ambiguous -- drop, never guess
    dup = merged.duplicated(subset=["cd_match_id", "cd_player_id"], keep=False) & merged["core_player_id"].notna()
    if dup.any():
        merged = merged[~dup]

    keep_cols = ["season", "core_match_id", "core_player_id", "leverage_weighted_disposals", "mean_leverage_disposals",
                 "n_disposal_events", "leverage_weighted_scoring", "n_scoring_events"]
    final = merged.dropna(subset=["core_player_id", "core_match_id"])[keep_cols].rename(
        columns={"core_player_id": "player_id", "core_match_id": "match_id"}
    )
    print(f"  Player-identity join: {len(final):,} / {len(result):,} rows resolved to a core player_id + match_id "
          f"({len(result) - len(final):,} unresolved, dropped)")
    return final


def build():
    all_seasons = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        print(f"Processing {season}...")
        all_seasons.append(build_season(season))
    result = pd.concat(all_seasons, ignore_index=True)
    result = _join_to_core_player_id(result)
    result.to_parquet(PROCESSED_DIR / "experimental_gamestate_features.parquet", index=False)
    print(f"Wrote {len(result):,} rows -> data/processed/experimental_gamestate_features.parquet")
    return result


if __name__ == "__main__":
    build()
