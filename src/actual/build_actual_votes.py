"""
Build the 2026 Brownlow ACTUAL-vote ground-truth tables from the raw AFL
tracker snapshots written by src/actual/fetch_afl_tracker.py.

Outputs (data/actual/):
  2026_brownlow_match_votes.csv   CANONICAL source of truth -- one row per
                                  awarded vote (3 rows per match, 621 total).
  2026_brownlow_player_round.csv  one row per leaderboard player x round
                                  (0..24) with actual + AFL-predictor votes
                                  and bye / did-not-play markers.
  2026_brownlow_leaderboard.csv   one row per player on the AFL leaderboard
                                  (after full "Show next 15 results"
                                  pagination), totals reconstructed from
                                  match votes and reconciled to the AFL total.
  2026_brownlow_validation.json   machine-readable results of every check in
                                  validate().

Identity: AFL players are mapped to this project's canonical 2026 player_id
with the existing external-identity logic (src/external/identity.py) --
round-level (round, surname, first initial, team) first, then the
season-level resolver for rows the round-level lookup could not see (those
are canonical NOID2026_* rows, see identity.py). Nothing is guessed:
anything not uniquely resolvable is left with player_id blank and
match_status != "resolved". AFL's own provider id (CD_I...) is kept on every
row as `afl_player_id` so the mapping is auditable.

Matches are mapped to canonical `match_id` on (official round, home team,
away team) with the local-venue date cross-checked against the date embedded
in the canonical id. Round 0 == Opening Round, in line with
src/data/round_normalization.py.

This is POST-EVENT ground truth only. It reads the frozen prediction files
read-only for identity and never writes anywhere but data/actual/.

Run:  python -m src.actual.build_actual_votes
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.external.identity import (
    load_canonical_player_rounds,
    load_canonical_players,
    resolve_external_match_players,
    resolve_external_players,
)

ROOT = Path(__file__).resolve().parents[2]
ACTUAL_DIR = ROOT / "data" / "actual"
RAW_DIR = ACTUAL_DIR / "raw"
REPORTS = ROOT / "reports"

SEASON = 2026
HOME_AWAY_ROUNDS = list(range(0, 25))  # 0 = Opening Round, 1..24
VOTE_VALUES = (3, 2, 1)
BUILDER_VERSION = "1.0.0"

# AFL API team abbreviations (confirmed against all 18 clubs in the raw
# snapshot; note GCFC / NMFC / WCE / BL differ from the ESPN/Betfair codes in
# src/external/identity.py, so this is its own explicit map).
AFL_TEAM_MAP = {
    "ADEL": "adelaide", "BL": "brisbane_lions", "CARL": "carlton", "COLL": "collingwood",
    "ESS": "essendon", "FRE": "fremantle", "GCFC": "gold_coast", "GEEL": "geelong",
    "GWS": "greater_western_sydney", "HAW": "hawthorn", "MELB": "melbourne",
    "NMFC": "north_melbourne", "PORT": "port_adelaide", "RICH": "richmond",
    "STK": "st_kilda", "SYD": "sydney", "WB": "western_bulldogs", "WCE": "west_coast",
}

# AFL players whose canonical 2026 identity genuinely does not exist: the
# frozen 2026 CORE build never resolved an afltables provider id for them, so
# every canonical row is a NOID2026_* placeholder (see identity.py). They are
# kept in the outputs with player_id blank and identity_status="unresolved";
# the identity gate passes only if every unresolved row is listed HERE, so a
# NEW unresolved player still fails the build.
KNOWN_UNRESOLVED_AFL_PLAYERS = {
    "CD_I1006133": "Jack Ross (Richmond): only NOID2026_* placeholder rows in canonical 2026 data",
}

MATCH_ID_RE = re.compile(r"^(\d{4})_R(\d+)_(.+)_v_(.+)_(\d{4}-\d{2}-\d{2})$")

# Files this build must never touch (frozen 2026 prediction / model /
# simulation outputs). Hashed before and after as a hard guard.
FROZEN_PREDICTION_FILES = [
    REPORTS / "2026_leaderboard.csv",
    REPORTS / "2026_predicted_votes.csv",
    REPORTS / "2026_objective_leaderboard.csv",
    REPORTS / "2026_objective_votes.csv",
    REPORTS / "2026_match_probabilities.csv",
    REPORTS / "2026_simulation_summary.csv",
    REPORTS / "2026_order_scenarios.csv",
    ROOT / "data" / "processed" / "model_core_2026.parquet",
    ROOT / "data" / "processed" / "model_advanced_2026.parquet",
    ROOT / "data" / "processed" / "scenario_predictions_2026.parquet",
    ROOT / "data" / "processed" / "all_2026_scenarios_and_ensemble.parquet",
    ROOT / "data" / "processed" / "mc_totals_2026.npy",
    ROOT / "data" / "processed" / "mc_totals_objective_2026.npy",
]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_file_hashes() -> dict[str, str | None]:
    return {str(p.relative_to(ROOT)): _sha256(p) for p in FROZEN_PREDICTION_FILES}


def round_label(r: int) -> str:
    return "Opening Round" if r == 0 else f"Round {r}"


def _full_name(p: dict) -> str:
    return f"{p['givenName']} {p['surname']}".strip()


# --------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------
def load_raw(raw_dir: Path = RAW_DIR) -> dict:
    def _j(name):
        return json.loads((raw_dir / name).read_text())
    return {
        "season": _j("afl_bfawards_season_CD_S2026014.json"),
        "leaderboard": _j("afl_bfawards_leaderboard_CD_S2026014.json"),
        "predictor_pages": _j("aflapi_award_brownlow_predictor_pages.json"),
        "matches": _j("aflapi_matches_compseason85.json"),
        "teams": _j("aflapi_teams_compseason85.json"),
        "dom": _j("dom_leaderboard_snapshot.json"),
        "status": _j("fetch_status.json"),
    }


# --------------------------------------------------------------------------
# Fixture / match mapping
# --------------------------------------------------------------------------
def load_canonical_matches() -> pd.DataFrame:
    """The 207 canonical 2026 home-and-away match_ids (read-only, from the
    frozen predicted-votes file), parsed into (round, home, away, date)."""
    ids = pd.read_csv(REPORTS / "2026_predicted_votes.csv", usecols=["match_id"])["match_id"].unique()
    rows = []
    for mid in ids:
        m = MATCH_ID_RE.match(mid)
        if not m:
            raise ValueError(f"unparseable canonical match_id {mid!r}")
        rows.append({"match_id": mid, "round": int(m.group(2)), "home_team": m.group(3),
                     "away_team": m.group(4), "date": m.group(5)})
    return pd.DataFrame(rows)


def build_fixture(raw: dict) -> pd.DataFrame:
    """AFL matchId -> round / teams / local date for home-and-away matches."""
    rows = []
    for m in raw["matches"]["body"]["matches"]:
        rn = m["round"]["roundNumber"]
        if rn not in HOME_AWAY_ROUNDS:
            continue
        utc = datetime.fromisoformat(m["utcStartTime"].replace("+0000", "+00:00"))
        tz = (m.get("venue") or {}).get("timezone") or "Australia/Melbourne"
        local = utc.astimezone(ZoneInfo(tz))
        rows.append({
            "afl_match_id": m["providerId"],
            "round": rn,
            "round_label": m["round"]["name"],
            "home_team_abbr": m["home"]["team"]["abbreviation"],
            "away_team_abbr": m["away"]["team"]["abbreviation"],
            "home_team": AFL_TEAM_MAP[m["home"]["team"]["abbreviation"]],
            "away_team": AFL_TEAM_MAP[m["away"]["team"]["abbreviation"]],
            "utc_start_time": m["utcStartTime"],
            "venue": (m.get("venue") or {}).get("name"),
            "venue_timezone": tz,
            "date": local.strftime("%Y-%m-%d"),
            "afl_status": m["status"],
        })
    fx = pd.DataFrame(rows)
    canon = load_canonical_matches()
    merged = fx.merge(canon, on=["round", "home_team", "away_team"], how="left",
                      suffixes=("", "_canonical"))
    merged["match_status"] = "resolved"
    merged.loc[merged["match_id"].isna(), "match_status"] = "unresolved"
    merged.loc[merged["match_id"].notna() & (merged["date"] != merged["date_canonical"]),
               "match_status"] = "date_mismatch"
    return merged


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
_MIDDLE_INITIAL_RE = re.compile(r"\b[A-Z]\.?\s+")


def lookup_name(afl_name: str) -> str:
    """AFL disambiguates same-name players with a middle initial ("Bailey J.
    Williams" = West Coast's Bailey Williams, vs the Bulldogs' Bailey
    Williams). Canonical names carry no middle initial, so strip a lone
    capital-letter token (with or without a period) from between the given
    name and surname before lookup. The raw AFL name is kept in the output."""
    parts = str(afl_name).split(maxsplit=1)
    if len(parts) < 2:
        return afl_name
    return parts[0] + " " + _MIDDLE_INITIAL_RE.sub("", parts[1] + " ").strip()


def _norm_full_name(name: str) -> str:
    return "".join(c for c in str(name).lower() if c.isalpha())


def resolve_players(df: pd.DataFrame, canonical_rounds=None, canonical_players=None) -> pd.DataFrame:
    """Round-level identity first (round, surname, first initial, team);
    then, for rows the round-level key found AMBIGUOUS (two real teammates
    sharing surname + initial in that round, e.g. Chad / Corey Warner), an
    EXACT full-name match against that round's canonical roster -- resolved
    only if exactly one canonical player in that round+team has the identical
    normalised full name; then the season-level (team-keyed) resolver for
    rows the round-level table cannot see at all. Nothing is guessed: every
    path requires uniqueness, otherwise the row stays unresolved/ambiguous."""
    if canonical_rounds is None:
        canonical_rounds = load_canonical_player_rounds()
    work = df.copy()
    work["_lookup_name"] = work["player_name"].map(lookup_name)
    out = resolve_external_match_players(work, "_lookup_name", "player_team", "round",
                                         canonical_rounds=canonical_rounds)
    out = out.rename(columns={"match_status": "identity_status"})

    amb = out["identity_status"] == "ambiguous"
    if amb.any():
        cr = canonical_rounds.copy()
        cr["_full"] = cr["player_name"].map(_norm_full_name)
        grp = cr.groupby(["round", "team_id", "_full"])["player_id"].agg(lambda s: sorted(set(s)))
        for i in out.index[amb]:
            key = (out.at[i, "round"], out.at[i, "player_team"], _norm_full_name(out.at[i, "_lookup_name"]))
            if key in grp.index and len(grp.loc[key]) == 1:
                out.at[i, "player_id"] = grp.loc[key][0]
                out.at[i, "identity_status"] = "resolved_exact_full_name"

    need = out["identity_status"].isin(["unresolved"])
    if need.any():
        fb = resolve_external_players(out.loc[need, ["_lookup_name", "player_team"]].copy(),
                                      "_lookup_name", "player_team", canonical=canonical_players)
        ok = fb["match_status"] == "resolved"
        out.loc[fb.index[ok], "player_id"] = fb.loc[ok, "player_id"]
        out.loc[fb.index[ok], "identity_status"] = "resolved_season_fallback"
        amb2 = (fb["match_status"] == "ambiguous") & ~ok
        out.loc[fb.index[amb2], "identity_status"] = "ambiguous"
    out["player_id"] = out["player_id"].astype("string")
    return out.drop(columns=["_lookup_name"])


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def build_match_votes(raw: dict, fixture: pd.DataFrame, **canon) -> pd.DataFrame:
    fx = fixture.set_index("afl_match_id")
    rows = []
    for m in raw["season"]["body"]["matchVotes"]:
        f = fx.loc[m["matchId"]]
        for v in m["votes"]:
            rows.append({
                "season": SEASON,
                "round": int(m["roundNumber"]),
                "round_label": round_label(int(m["roundNumber"])),
                "match_id": f["match_id"],
                "afl_match_id": m["matchId"],
                "date": f["date"],
                "home_team": f["home_team"],
                "away_team": f["away_team"],
                "player_name": _full_name(v["player"]),
                "afl_player_id": v["player"]["playerId"],
                "player_team": AFL_TEAM_MAP[v["team"]["teamAbbr"]],
                "player_team_abbr": v["team"]["teamAbbr"],
                "actual_brownlow_votes": int(v["votes"]),
                "eligible": bool(v["eligible"]),
            })
    df = pd.DataFrame(rows)
    df = resolve_players(df, **canon)
    df = df.sort_values(["round", "date", "match_id", "actual_brownlow_votes"],
                        ascending=[True, True, True, False]).reset_index(drop=True)
    cols = ["season", "round", "round_label", "match_id", "afl_match_id", "date", "home_team",
            "away_team", "player_id", "afl_player_id", "player_name", "player_team",
            "player_team_abbr", "actual_brownlow_votes", "eligible", "identity_status"]
    return df[cols]


def _predictor_index(raw: dict) -> dict[str, dict]:
    idx = {}
    for page in raw["predictor_pages"]:
        for p in page["body"]["players"]:
            idx[p["providerId"]] = p
    return idx


def build_player_round(raw: dict, match_votes: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    pred = _predictor_index(raw)
    fx = fixture.set_index("afl_match_id")
    ident = (match_votes.drop_duplicates("afl_player_id")
             .set_index("afl_player_id")[["player_id", "identity_status"]])
    lb = raw["leaderboard"]["body"]["leaderboard"]
    rows = []
    for order, e in enumerate(lb, start=1):
        pid = e["player"]["playerId"]
        team = AFL_TEAM_MAP[e["team"]["teamAbbr"]]
        actual_by_round = {int(x["roundNumber"]): (x["matchId"], int(x["votes"])) for x in e["roundByRoundVotes"]}
        p = pred.get(pid)
        for r in HOME_AWAY_ROUNDS:
            entry = (p or {}).get("rounds", {}).get(str(r), [None])[0] if p else None
            afl_match_id, actual, predicted = None, 0, 0
            if entry is None:
                status = "played" if p is not None else "unknown"
            elif entry.get("bye"):
                status, actual, predicted = "bye", None, None
            elif entry.get("played") is False:
                status, actual, predicted, afl_match_id = "did_not_play", None, None, entry.get("providerId")
            else:
                status, predicted, afl_match_id = "played", int(entry.get("points") or 0), entry.get("providerId")
            if r in actual_by_round:
                afl_match_id, actual = actual_by_round[r]
                if status in ("bye", "did_not_play"):
                    status = "played"  # actual vote wins over a stale predictor marker
            rows.append({
                "season": SEASON, "afl_leaderboard_order": order,
                "player_id": ident["player_id"].get(pid, pd.NA),
                "afl_player_id": pid, "player_name": _full_name(e["player"]), "player_team": team,
                "eligible": bool(e["eligible"]), "round": r, "round_label": round_label(r),
                "afl_match_id": afl_match_id,
                "match_id": fx["match_id"].get(afl_match_id) if afl_match_id else None,
                "status": status,
                "actual_votes": actual, "afl_predicted_votes": predicted,
            })
    df = pd.DataFrame(rows)
    df["actual_votes"] = df["actual_votes"].astype("Int64")
    df["afl_predicted_votes"] = df["afl_predicted_votes"].astype("Int64")
    df["player_id"] = df["player_id"].astype("string")
    return df


def build_leaderboard(raw: dict, match_votes: pd.DataFrame, player_round: pd.DataFrame) -> pd.DataFrame:
    lb = raw["leaderboard"]["body"]["leaderboard"]
    pred = _predictor_index(raw)
    recon = (match_votes.groupby("afl_player_id")["actual_brownlow_votes"]
             .agg(reconstructed_total_votes="sum",
                  n_3_votes=lambda s: int((s == 3).sum()),
                  n_2_votes=lambda s: int((s == 2).sum()),
                  n_1_votes=lambda s: int((s == 1).sum()),
                  games_with_votes="size"))
    ident = (match_votes.drop_duplicates("afl_player_id")
             .set_index("afl_player_id")[["player_id", "identity_status"]])
    pr = player_round.groupby("afl_player_id")["status"].agg(
        byes=lambda s: int((s == "bye").sum()),
        did_not_play=lambda s: int((s == "did_not_play").sum()),
        games_played=lambda s: int((s == "played").sum()))
    rows = []
    for order, e in enumerate(lb, start=1):
        pid = e["player"]["playerId"]
        p = pred.get(pid, {})
        rows.append({
            "season": SEASON, "afl_leaderboard_order": order,
            "player_id": ident["player_id"].get(pid, pd.NA),
            "identity_status": ident["identity_status"].get(pid, "unresolved"),
            "afl_player_id": pid, "player_name": _full_name(e["player"]),
            "player_team": AFL_TEAM_MAP[e["team"]["teamAbbr"]], "player_team_abbr": e["team"]["teamAbbr"],
            "eligible": bool(e["eligible"]), "winner": bool(e["winner"]),
            "afl_total_actual_votes": int(e["totalVotes"]),
            "afl_total_predicted_votes": int(p["totalVotes"]) if p else pd.NA,
        })
    df = pd.DataFrame(rows).set_index("afl_player_id").join(recon).join(pr).reset_index()
    for c in ("reconstructed_total_votes", "n_3_votes", "n_2_votes", "n_1_votes", "games_with_votes",
              "byes", "did_not_play", "games_played"):
        df[c] = df[c].fillna(0).astype(int)
    df["afl_total_predicted_votes"] = df["afl_total_predicted_votes"].astype("Int64")
    df["player_id"] = df["player_id"].astype("string")
    df["rank"] = df["afl_total_actual_votes"].rank(method="min", ascending=False).astype(int)
    df["eligible_rank"] = (df.loc[df["eligible"], "afl_total_actual_votes"]
                           .rank(method="min", ascending=False)).reindex(df.index).astype("Int64")
    df = df.sort_values(["afl_leaderboard_order"]).reset_index(drop=True)
    cols = ["season", "rank", "eligible_rank", "afl_leaderboard_order", "player_id", "afl_player_id",
            "player_name", "player_team", "player_team_abbr", "eligible", "winner",
            "afl_total_actual_votes", "reconstructed_total_votes", "afl_total_predicted_votes",
            "n_3_votes", "n_2_votes", "n_1_votes", "games_with_votes", "games_played", "byes",
            "did_not_play", "identity_status"]
    return df[cols]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(raw: dict, fixture: pd.DataFrame, mv: pd.DataFrame, pr: pd.DataFrame,
             lb: pd.DataFrame, hashes_before: dict, hashes_after: dict) -> dict:
    checks: dict[str, dict] = {}

    def add(name, ok, **detail):
        checks[name] = {"ok": bool(ok), **detail}

    canon = load_canonical_matches()
    per_match = mv.groupby("match_id")["actual_brownlow_votes"]
    totals = per_match.sum()
    sets = per_match.apply(lambda s: tuple(sorted(s, reverse=True)))
    add("every_match_has_3_2_1", (sets == (3, 2, 1)).all(),
        bad=sets[sets != (3, 2, 1)].to_dict())
    add("match_vote_total_is_6", (totals == 6).all(), bad=totals[totals != 6].to_dict())
    dup = mv.duplicated(["match_id", "actual_brownlow_votes"]).sum()
    add("no_duplicate_vote_positions", dup == 0, duplicates=int(dup))
    dup_player = mv.duplicated(["match_id", "afl_player_id"]).sum()
    add("no_player_voted_twice_in_a_match", dup_player == 0, duplicates=int(dup_player))
    add("match_count_207", mv["match_id"].nunique() == 207, matches=int(mv["match_id"].nunique()),
        vote_rows=int(len(mv)))
    add("all_afl_matches_mapped_to_canonical",
        mv["match_id"].notna().all() and (fixture.loc[fixture["afl_match_id"].isin(mv["afl_match_id"]), "match_status"] == "resolved").all(),
        unmapped=mv.loc[mv["match_id"].isna(), "afl_match_id"].unique().tolist(),
        date_mismatch=fixture.loc[fixture["match_status"] == "date_mismatch", "afl_match_id"].tolist())
    add("all_canonical_matches_have_votes", set(canon["match_id"]) == set(mv["match_id"].dropna()),
        missing=sorted(set(canon["match_id"]) - set(mv["match_id"].dropna())))
    per_round_actual = mv.groupby("round")["match_id"].nunique().to_dict()
    per_round_canon = canon.groupby("round")["match_id"].nunique().to_dict()
    add("round_coverage_0_to_24", per_round_actual == per_round_canon and sorted(per_round_actual) == HOME_AWAY_ROUNDS,
        actual=per_round_actual, canonical=per_round_canon)
    add("opening_round_is_round_0",
        (mv.loc[mv["round"] == 0, "round_label"] == "Opening Round").all() and (mv["round"] == 0).sum() == 15,
        opening_round_vote_rows=int((mv["round"] == 0).sum()))
    add("player_team_is_home_or_away",
        ((mv["player_team"] == mv["home_team"]) | (mv["player_team"] == mv["away_team"])).all(),
        bad=mv.loc[~((mv["player_team"] == mv["home_team"]) | (mv["player_team"] == mv["away_team"])),
                   ["match_id", "player_name", "player_team"]].to_dict("records"))

    # Totals reconciliation
    mism = lb[lb["afl_total_actual_votes"] != lb["reconstructed_total_votes"]]
    add("reconstructed_totals_equal_afl_leaderboard", mism.empty,
        mismatches=mism[["player_name", "afl_total_actual_votes", "reconstructed_total_votes"]].to_dict("records"))
    add("sum_of_leaderboard_totals_equals_all_votes",
        int(lb["afl_total_actual_votes"].sum()) == int(mv["actual_brownlow_votes"].sum()) == 207 * 6,
        leaderboard_sum=int(lb["afl_total_actual_votes"].sum()), match_votes_sum=int(mv["actual_brownlow_votes"].sum()))
    voters = set(mv["afl_player_id"])
    add("every_vote_getter_on_leaderboard_and_vice_versa", voters == set(lb["afl_player_id"]),
        only_in_matches=sorted(voters - set(lb["afl_player_id"])), only_on_leaderboard=sorted(set(lb["afl_player_id"]) - voters))
    d = lb[(lb["player_name"] == "Nick Daicos")]
    add("nick_daicos_total_47", len(d) == 1 and int(d["afl_total_actual_votes"].iloc[0]) == 47
        and int(d["reconstructed_total_votes"].iloc[0]) == 47 and bool(d["winner"].iloc[0]),
        rows=d[["afl_total_actual_votes", "reconstructed_total_votes", "winner"]].to_dict("records"))

    # Pagination completeness
    st = raw["status"]
    api_n = len(raw["leaderboard"]["body"]["leaderboard"])
    dom_n = len(raw["dom"]["rows"])
    add("leaderboard_pagination_exhausted",
        (not st["load_more_button_present_at_end"]) and dom_n == api_n == len(lb),
        api_players=api_n, dom_rows=dom_n, clicks=len(st["pagination"]) - 1,
        button_present_at_end=st["load_more_button_present_at_end"],
        last_click_label=st["pagination"][-1].get("label"))
    # DOM cross-check: per-cell actual votes and totals equal the API tables
    dom_rows = {r["afl_player_id"]: r for r in raw["dom"]["rows"]}
    cell_mismatch, total_mismatch = [], []
    for _, row in lb.iterrows():
        dr = dom_rows.get(row["afl_player_id"])
        if dr is None:
            total_mismatch.append((row["player_name"], "missing in DOM")); continue
        if str(row["afl_total_actual_votes"]) != dr["total_actual"]:
            total_mismatch.append((row["player_name"], row["afl_total_actual_votes"], dr["total_actual"]))
        sub = pr[pr["afl_player_id"] == row["afl_player_id"]].set_index("round")
        for r in HOME_AWAY_ROUNDS:
            cell = dr["cells"][r]
            api_actual = sub.loc[r, "actual_votes"]
            api_str = "" if pd.isna(api_actual) or api_actual == 0 else str(int(api_actual))
            dom_str = "" if cell["bye"] or cell["not_played"] else cell["actual"]
            if api_str != dom_str or (cell["bye"] != (sub.loc[r, "status"] == "bye")):
                cell_mismatch.append((row["player_name"], r, api_str, cell["actual"], cell["bye"], cell["not_played"]))
    add("dom_leaderboard_matches_api", not cell_mismatch and not total_mismatch,
        cell_mismatches=cell_mismatch[:50], total_mismatches=total_mismatch[:50])

    # Identity
    unresolved = mv[~mv["identity_status"].str.startswith("resolved")]
    unexpected = unresolved[~unresolved["afl_player_id"].isin(KNOWN_UNRESOLVED_AFL_PLAYERS)]
    add("identity_unresolved_only_known_gaps", unexpected.empty,
        unresolved_rows=unresolved[["round", "match_id", "player_name", "afl_player_id", "player_team", "identity_status"]].to_dict("records"),
        unexpected_unresolved_rows=unexpected[["round", "match_id", "player_name", "afl_player_id", "player_team"]].to_dict("records"),
        known_gaps=KNOWN_UNRESOLVED_AFL_PLAYERS,
        status_counts=mv["identity_status"].value_counts().to_dict())
    res = mv[mv["player_id"].notna()]
    pid_per_afl = res.groupby("afl_player_id")["player_id"].nunique()
    afl_per_pid = res.groupby("player_id")["afl_player_id"].nunique()
    add("identity_one_to_one", (pid_per_afl == 1).all() and (afl_per_pid == 1).all(),
        afl_ids_with_multiple_canonical=pid_per_afl[pid_per_afl > 1].to_dict(),
        canonical_ids_with_multiple_afl=afl_per_pid[afl_per_pid > 1].to_dict())
    lb_dupe = lb["player_id"].dropna().duplicated().sum()
    add("leaderboard_player_id_unique", lb_dupe == 0 and lb["afl_player_id"].is_unique, duplicated_canonical=int(lb_dupe))

    # Ineligible / anomalies (reported, not failures)
    inel = lb[~lb["eligible"]]
    checks["ineligible_players_report"] = {
        "ok": True, "count": int(len(inel)),
        "players": inel[["player_name", "player_team", "afl_total_actual_votes", "rank"]].to_dict("records"),
        "note": "AFL lists ineligible players with their votes; votes are kept in match_votes with eligible=False.",
    }
    stale = pr[(pr["actual_votes"].fillna(0) > 0) & (pr["afl_predicted_votes"].isna())]
    checks["anomalies_report"] = {
        "ok": True,
        "actual_vote_on_predictor_bye_or_dnp_marker": stale[["player_name", "round", "actual_votes"]].to_dict("records"),
        "players_without_predictor_record": sorted(set(lb["afl_player_id"]) - set(_predictor_index(raw))),
        "afl_leaderboard_status": raw["leaderboard"]["body"].get("status"),
        "afl_season_status": raw["season"]["body"].get("status"),
    }

    changed = {k: (hashes_before.get(k), v) for k, v in hashes_after.items() if hashes_before.get(k) != v}
    add("frozen_prediction_files_untouched", not changed, changed=changed)

    hard = [k for k, v in checks.items() if not k.endswith("_report")]
    return {
        "builder_version": BUILDER_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_retrieved_at": raw["status"]["retrieved_at"],
        "all_hard_checks_pass": all(checks[k]["ok"] for k in hard),
        "n_hard_checks": len(hard),
        "counts": {"players": int(len(lb)), "matches": int(mv["match_id"].nunique()), "vote_rows": int(len(mv)),
                   "player_round_rows": int(len(pr)), "eligible_players": int(lb["eligible"].sum())},
        "checks": checks,
        "frozen_file_hashes": hashes_after,
    }


def build(raw_dir: Path = RAW_DIR, out_dir: Path = ACTUAL_DIR) -> dict:
    hashes_before = frozen_file_hashes()
    raw = load_raw(raw_dir)
    fixture = build_fixture(raw)
    canon_rounds = load_canonical_player_rounds()
    canon_players = load_canonical_players()
    mv = build_match_votes(raw, fixture, canonical_rounds=canon_rounds, canonical_players=canon_players)
    pr = build_player_round(raw, mv, fixture)
    lb = build_leaderboard(raw, mv, pr)
    out_dir.mkdir(parents=True, exist_ok=True)
    mv.to_csv(out_dir / "2026_brownlow_match_votes.csv", index=False)
    pr.to_csv(out_dir / "2026_brownlow_player_round.csv", index=False)
    lb.to_csv(out_dir / "2026_brownlow_leaderboard.csv", index=False)
    report = validate(raw, fixture, mv, pr, lb, hashes_before, frozen_file_hashes())
    (out_dir / "2026_brownlow_validation.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main() -> int:
    report = build()
    print(json.dumps({"all_hard_checks_pass": report["all_hard_checks_pass"], "counts": report["counts"]}))
    for name, c in report["checks"].items():
        flag = "OK  " if c["ok"] else "FAIL"
        print(f"{flag} {name}")
    return 0 if report["all_hard_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
