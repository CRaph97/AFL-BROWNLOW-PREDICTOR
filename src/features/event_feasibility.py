"""
Phase 3, section O: EXPERIMENTAL event/game-state feature feasibility check on
the torpdata chains dataset (see docs/EVENT_DATA_2021_AUDIT.md for the source
validation). This module does NOT merge anything into the primary CORE/
ADVANCED/analytical datasets -- it only establishes whether reliable game-state
variables (score differential over time, quarter, time remaining, lead
changes, close-game state) can be reconstructed, and quantifies where the
reconstruction disagrees with official final scores.

Known issue carried over from docs/EVENT_DATA_2021_AUDIT.md: naively counting
`description == "Behind"` events undercounts true behinds by ~17% in the one
match spot-checked (rushed behinds are not always a dedicated event row).
This script checks whether that holds up across every 2024 match, not just
the one already checked, and whether GOALS (which validated perfectly in that
one match) also hold up at full-season scale -- reconciliation must never be
silently trusted just because one match passed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "torpdata_pilot"
REPORTS_DIR = ROOT / "reports"


def reconcile_scores(chains: pd.DataFrame) -> pd.DataFrame:
    """For every match, reconstruct total goals/behinds per team from event
    descriptions and compare against the official final score columns carried
    on every row.

    CRITICAL fix found during Phase 3 full-scale validation (not caught by the
    single-match spot check in docs/EVENT_DATA_2021_AUDIT.md): `final_state`
    (e.g. "rushed", "rushedOpp") is a CHAIN-level outcome copied onto EVERY
    action row in that chain, not a per-event flag. Naively counting rows with
    final_state in ("rushed","rushedOpp") massively over-counts (a single
    rushed-behind chain can contain 5-8 action rows). The correct unit is one
    behind per DISTINCT (match_id, chain_number) whose final_state indicates a
    score. Team attribution also differs by final_state:
      - "behind": scoring team = chain_team_id (verified against the explicit
        "Behind"-description row within the same chain -- always equal).
      - "rushed": scoring team = chain_team_id (the chain's own attacking team
        forced it through under no defensive pressure).
      - "rushedOpp": scoring team = the OTHER team in the match (the defence
        rushed it through under attacking pressure from chain_team_id) -- per
        the field-naming convention documented in the original torp-predecessor
        scoring script found during the Phase 2 event-data investigation.
    """
    is_goal = chains["description"] == "Goal"

    chain_level = (
        chains[["match_id", "chain_number", "final_state", "chain_team_id", "home_team_id", "away_team_id"]]
        .drop_duplicates(subset=["match_id", "chain_number"])
    )
    is_behind_chain = chain_level["final_state"] == "behind"
    is_rushed_chain = chain_level["final_state"] == "rushed"
    is_rushed_opp_chain = chain_level["final_state"] == "rushedOpp"

    scoring_team = pd.Series(pd.NA, index=chain_level.index, dtype="object")
    scoring_team[is_behind_chain | is_rushed_chain] = chain_level.loc[is_behind_chain | is_rushed_chain, "chain_team_id"]
    flipped = np.where(
        chain_level.loc[is_rushed_opp_chain, "chain_team_id"] == chain_level.loc[is_rushed_opp_chain, "home_team_id"],
        chain_level.loc[is_rushed_opp_chain, "away_team_id"],
        chain_level.loc[is_rushed_opp_chain, "home_team_id"],
    )
    scoring_team[is_rushed_opp_chain] = flipped
    chain_level["scoring_team"] = scoring_team

    behinds_naive_desc = chains[chains["description"] == "Behind"].groupby(["match_id", "team_id"]).size().rename("reconstructed_behinds_naive")
    goals = chains[is_goal].groupby(["match_id", "team_id"]).size().rename("reconstructed_goals")
    behinds_fixed = (
        chain_level.dropna(subset=["scoring_team"])
        .groupby(["match_id", "scoring_team"]).size()
        .rename("reconstructed_behinds_fixed")
    )
    behinds_fixed.index = behinds_fixed.index.set_names(["match_id", "team_id"])

    recon = pd.concat([goals, behinds_naive_desc, behinds_fixed], axis=1).reset_index().fillna(0)

    official = chains[[
        "match_id", "home_team_id", "away_team_id",
        "home_team_score_goals", "home_team_score_behinds",
        "away_team_score_goals", "away_team_score_behinds",
    ]].drop_duplicates(subset=["match_id"])

    home = official.merge(
        recon, left_on=["match_id", "home_team_id"], right_on=["match_id", "team_id"], how="left"
    ).rename(columns={"home_team_score_goals": "official_goals", "home_team_score_behinds": "official_behinds"})
    away = official.merge(
        recon, left_on=["match_id", "away_team_id"], right_on=["match_id", "team_id"], how="left"
    ).rename(columns={"away_team_score_goals": "official_goals", "away_team_score_behinds": "official_behinds"})

    both = pd.concat([home, away], ignore_index=True)
    both["goals_match_official"] = both["reconstructed_goals"] == both["official_goals"]
    both["behinds_naive_match_official"] = both["reconstructed_behinds_naive"] == both["official_behinds"]
    both["behinds_fixed_match_official"] = both["reconstructed_behinds_fixed"] == both["official_behinds"]
    return both


def reconstruct_game_state(chains: pd.DataFrame, match_id: str) -> pd.DataFrame:
    """Build a running score-differential-over-time timeline for one match.
    Goals: one per action row with description=='Goal' (validated at 99.1% vs
    official scores across the full 2024 season). Behinds: one per DISTINCT
    CHAIN (not action row -- see reconcile_scores docstring for why row-level
    counting is wrong) whose final_state is behind/rushed/rushedOpp, with the
    same team-attribution rule used there. Uses each chain's first action row
    for period/period_seconds (a chain's timing is materially constant across
    its own actions for this purpose)."""
    m = chains[chains["match_id"] == match_id].sort_values(["period", "period_seconds", "display_order"]).copy()
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
    ]).sort_values(["period", "period_seconds"])

    running_home, running_away = 0, 0
    rows = []
    prev_margin = 0
    for _, ev in scoring.iterrows():
        if ev["scoring_team"] == home_id:
            running_home += ev["points"]
        elif ev["scoring_team"] == away_id:
            running_away += ev["points"]
        else:
            continue  # unattributed scoring event -- skip rather than guess
        margin = running_home - running_away
        rows.append({
            "period": ev["period"], "period_seconds": ev["period_seconds"],
            "home_score": running_home, "away_score": running_away,
            "margin_home_perspective": margin, "absolute_margin": abs(margin),
            "lead_changed": np.sign(margin) != np.sign(prev_margin) and margin != 0 and prev_margin != 0,
            "scores_level": margin == 0,
        })
        prev_margin = margin

    return pd.DataFrame(rows)


def simple_leverage(absolute_margin: float, period: int, period_seconds: float, total_periods: int = 4,
                     period_length_seconds: float = 1200) -> float:
    """A minimal, transparent leverage proxy: higher when the game is close AND late.
    NOT fit to any Brownlow outcome -- this is a feasibility placeholder per the Phase 3
    instruction to establish whether game state can be reconstructed before inventing
    action values. time_fraction_elapsed in [0, 1]; closeness in (0, 1]."""
    time_fraction_elapsed = min(1.0, ((period - 1) * period_length_seconds + period_seconds) /
                                 (total_periods * period_length_seconds))
    closeness = 1.0 / (1.0 + absolute_margin / 6.0)  # 6 points = one goal
    return closeness * time_fraction_elapsed


def build():
    frames = []
    for season in [2024]:  # start with the season already validated in Phase 2; extend later if this passes
        path = RAW_DIR / f"chains_data_{season}_all.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    chains = pd.concat(frames, ignore_index=True)

    recon = reconcile_scores(chains)
    n_matches = recon["match_id"].nunique()
    print(f"Reconciliation across {n_matches} matches, {len(recon)} team-sides:")
    print(f"  Goals reconstruct exactly:            {recon['goals_match_official'].mean():.1%}")
    print(f"  Behinds (naive count) reconstruct:    {recon['behinds_naive_match_official'].mean():.1%}")
    print(f"  Behinds (rushed-fixed) reconstruct:   {recon['behinds_fixed_match_official'].mean():.1%}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    recon.to_csv(REPORTS_DIR / "event_score_reconciliation_2024.csv", index=False)

    # demonstrate on the already-validated match
    sample_match = "CD_M20240140909"
    timeline = reconstruct_game_state(chains, sample_match)
    timeline["leverage"] = timeline.apply(
        lambda r: simple_leverage(r["absolute_margin"], r["period"], r["period_seconds"]), axis=1
    )
    timeline.to_csv(REPORTS_DIR / "event_game_state_sample_match.csv", index=False)
    print(f"\nSample reconstructed timeline ({sample_match}, {len(timeline)} scoring events) "
          f"-> reports/event_game_state_sample_match.csv")
    final_row = timeline.iloc[-1]
    print(f"Final reconstructed score: home {final_row['home_score']} - away {final_row['away_score']} "
          f"(official: 90 - 90)")

    return recon, timeline


if __name__ == "__main__":
    build()
