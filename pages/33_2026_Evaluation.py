"""
2026 Evaluation -- post-Brownlow scorecard for the frozen pre-count outputs
of the Production model, the Objective model and Wheelo, plus the settled
pre-count bookmaker markets.

Presentation only. Every number is read from data/evaluation/2026/ (built by
`python -m src.evaluation.build_2026_evaluation`); nothing on this page
recomputes a metric, settles a bet, reruns a simulation, blends a forecast
or writes anywhere. Actual votes are post-event ground truth
(data/actual/); bookmaker prices are the 2026-09-18 snapshot, captured three
days before the count -- no post-result price is used anywhere.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import evaluation as ev

st.set_page_config(page_title="2026 Evaluation", layout="wide")
d.highlight_objective_stats_nav()
st.title("2026 Evaluation")
st.caption(
    "How the frozen 2026 forecasts (Production, Objective, Wheelo) and the pre-count bookmaker "
    "markets performed against the AFL's actual 3-2-1 votes. Retrospective analysis only -- "
    "no model was retrained, no simulation rerun, no forecast blended."
)

if not ev.available():
    st.info("No evaluation outputs found. Run `python -m src.evaluation.build_2026_evaluation` first.")
    st.stop()

manifest = ev.load_json("manifest")
scorecard = ev.load_csv("scorecard")
match_sc = ev.load_csv("match_scorecard")
season = ev.load_csv("season_players")
final_order = ev.load_csv("final_order_top20")
coverage = ev.load_json("simulation_coverage")
match_table = ev.load_csv("match_table")
round_table = ev.load_csv("round_table")
disagreement = ev.load_csv("disagreement_summary")
cases = ev.load_csv("disagreement_cases")
season_bias = ev.load_csv("season_bias")
match_bias = ev.load_csv("match_bias")
settled = ev.load_csv("betting_settled")
bet_summary = ev.load_csv("betting_summary")
team_leaders = ev.load_csv("team_leaders")
cal_bins = ev.load_csv("calibration_bins")
cal_summary = ev.load_csv("calibration_summary")
learnings = ev.load_json("learnings")

W = "stretch"
MODELS = ["Production", "Objective", "Wheelo"]


def _dl(df: pd.DataFrame, name: str, key: str) -> None:
    st.download_button(f"Download {name} (CSV)", ev.csv_bytes(df), file_name=f"2026_evaluation_{name}.csv", mime="text/csv", key=key)


def _table(df: pd.DataFrame, key: str, name: str | None = None, height: int | None = None, **kwargs) -> None:
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, hide_index=True, width=W, **kwargs)
    if name:
        _dl(df, name, key)


with st.expander("Data integrity for this page", expanded=False):
    c = manifest.get("counts", {})
    st.markdown(
        f"- **Ground truth**: `data/actual/2026_brownlow_match_votes.csv` (207 matches, 621 votes) -- read-only.\n"
        f"- **Forecasts**: frozen pre-count files, SHA-256 verified unchanged during the build "
        f"({'OK' if manifest.get('frozen_inputs_unchanged_during_build') else 'CHANGED -- investigate'}).\n"
        f"- **Bookmaker prices**: captured {', '.join(x[:16].replace('T', ' ') for x in manifest.get('bookmaker_snapshots_retrieved_at', []))} UTC; "
        f"count night {manifest.get('count_date_utc', '')[:10]} -- all pre-count: {manifest.get('all_bookmaker_snapshots_pre_count')}.\n"
        f"- **Coverage**: {c.get('season_players')} players, {c.get('matches')} matches, {c.get('priced_selections')} priced selections of which "
        f"{c.get('settled_selections')} settled, {c.get('unmodelled_excluded')} unmodelled (no pre-count model probability) and "
        f"{c.get('unsettleable', 0) - c.get('unmodelled_excluded', 0)} unsettleable (player identity unresolved pre-count).\n"
        f"- **Unresolved actual-vote rows**: {c.get('unresolved_actual_vote_rows')} (Jack Ross, Richmond -- no canonical id in the frozen 2026 build).\n"
        f"- Built {manifest.get('built_at', '')[:19].replace('T', ' ')} UTC. Methodology: `docs/2026_BROWNLOW_EVALUATION.md`."
    )

tabs = st.tabs(["Overall Scorecard", "Final Leaderboard", "Players", "Match Accuracy", "Model Disagreement",
                "Bias & Roles", "Betting Results", "Calibration", "2027 Learnings"])

# ---------------------------------------------------------------- 1. Scorecard
with tabs[0]:
    st.subheader("Season-total scorecard")
    common = scorecard[scorecard["universe"].str.startswith("common")].set_index("model")
    own = scorecard[scorecard["universe"] == "own coverage"].set_index("model")
    winner_row = season[season["actual_rank"] == 1].iloc[0]
    st.markdown(f"**Actual winner:** {winner_row['player_name']} ({ev.display_team(winner_row['team_id'])}, {int(winner_row['actual_votes'])} votes). "
                f"**Predicted EV leader:** " + ", ".join(f"{m}: {own.loc[m, 'predicted_leader']} ({own.loc[m, 'predicted_leader_ev']:.1f})" for m in MODELS) + ".")
    cols = st.columns(3)
    for col, m in zip(cols, MODELS):
        with col:
            st.markdown(f"#### {m}")
            r = common.loc[m]
            st.metric("Season-total MAE (common universe)", f"{r['mae']:.3f}", help=f"n = {int(r['n_players'])} players covered by all three sources")
            st.metric("RMSE", f"{r['rmse']:.3f}")
            st.metric("Spearman rank correlation", f"{r['spearman_rho']:.3f}")
            st.metric("Top-10 hit rate", ev.pct(r["top10_hit_rate"], 0), help=f"share of the actual top-{int(r['top10_n'])} inside the model's top-{int(r['top10_n'])} by EV")
            st.metric("Winner's predicted rank", int(r["winner_predicted_rank"]))
    st.markdown("**Full season-level metrics** (own coverage = every player the source scored; common = the players all three scored, the fair comparison).")
    show = scorecard[["model", "universe", "n_players", "mae", "rmse", "mean_bias", "mae_relevant", "n_relevant", "spearman_rho", "kendall_tau",
                      "top3_hit_rate", "top5_hit_rate", "top10_hit_rate", "top20_hit_rate", "winner_predicted_correctly", "winner_rank_error",
                      "mean_abs_rank_error_actual_top30"]].rename(columns={"mae_relevant": "mae (actual>=1 or EV>=1)", "n_relevant": "n relevant",
                                                                            "mean_abs_rank_error_actual_top30": "mean |rank error| (actual top 30)"})
    _table(show, "dl_scorecard", "scorecard")
    st.caption("Top-N hit rates use ties in the actual order: the actual top-5 contains 7 players (three tied on 27), so denominators are shown in the download.")

    st.subheader("Match-level scorecard (207 matches)")
    ms = match_sc.set_index("model")
    cols = st.columns(3)
    for col, m in zip(cols, MODELS):
        with col:
            st.markdown(f"#### {m}")
            r = ms.loc[m]
            st.metric("Named the 3-vote player", ev.pct(r["hit_3_rate"]))
            st.metric("3-voter inside top-3 by P3", ev.pct(r["actual_3_in_top3_rate"]))
            st.metric("Exact 3-2-1", ev.pct(r["exact_321_rate"]))
            st.metric("Mean P3 log loss", f"{r['mean_log_loss_p3']:.3f}", help=f"n = {int(r['n_log_loss'])} matches where the 3-voter was in the model's roster; lower is better")
            st.metric("Brier (P3, per roster player)", f"{r['mean_brier_p3']:.4f}")
    _table(match_sc, "dl_match_sc", "match_scorecard")
    st.caption("Wheelo publishes a per-match P3 % and match rank, so its 3-vote accuracy, log loss and Brier are genuine; it publishes no P2/P1, "
               "so those series exist only for Production and Objective. Expected-vote error is per roster player (rosters differ: Production ~41, Objective/Wheelo ~46).")

# ---------------------------------------------------------------- 2. Final leaderboard
with tabs[1]:
    st.subheader("Actual top 20 vs each model")
    fo = final_order.copy()
    view = pd.DataFrame({
        "Actual rank": fo["actual_rank"], "Player": fo["player_name"], "Team": fo["team_id"].map(ev.display_team), "Eligible": fo["eligible"],
        "Actual votes": fo["actual_votes"],
        "Prod EV": fo["production_ev"].round(1), "Prod rank": fo["production_rank"], "Prod rank err": fo["production_rank_error"],
        "Obj EV": fo["objective_ev"].round(1), "Obj rank": fo["objective_rank"], "Obj rank err": fo["objective_rank_error"],
        "Wheelo EV": fo["wheelo_ev"].round(1), "Wheelo rank": fo["wheelo_rank"], "Wheelo rank err": fo["wheelo_rank_error"],
        "Prod P(win)": fo["production_prob_rank_1"], "Prod P(top10)": fo["production_prob_top10_rank"], "Prod 95% band": fo.apply(lambda r: f"{r['production_sim_p2_5']:.0f}-{r['production_sim_p97_5']:.0f}" if pd.notna(r["production_sim_p2_5"]) else "", axis=1),
        "Obj P(win)": fo["objective_prob_rank_1"], "Obj P(top10)": fo["objective_prob_top10_rank"], "Obj 95% band": fo.apply(lambda r: f"{r['objective_sim_p2_5']:.0f}-{r['objective_sim_p97_5']:.0f}" if pd.notna(r["objective_sim_p2_5"]) else "", axis=1),
    })
    _table(view, "dl_final", "final_order_top20", column_config={
        "Prod P(win)": st.column_config.NumberColumn(format="%.3f"), "Obj P(win)": st.column_config.NumberColumn(format="%.3f"),
        "Prod P(top10)": st.column_config.NumberColumn(format="%.2f"), "Obj P(top10)": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Rank error = predicted rank minus actual rank (negative = the model had the player higher than they finished). "
               "Actual rank includes ineligible players, as the AFL leaderboard and bookmakers' 'Includes Ineligible' markets do.")
    st.subheader("Top-N inclusion")
    inc = []
    for n in (3, 5, 10, 20):
        row = {"Top N": n}
        for m in MODELS:
            r = common.loc[m]
            row[m] = f"{r[f'top{n}_hit_rate'] * 100:.0f}% of {int(r[f'top{n}_n'])}"
        inc.append(row)
    st.dataframe(pd.DataFrame(inc), hide_index=True, width=W)

    st.subheader("Finishing Order simulations (existing draws, not rerun)")
    c1, c2 = st.columns(2)
    for col, m in zip((c1, c2), ("production", "objective")):
        with col:
            st.markdown(f"#### {m.title()} simulation ({coverage.get(f'{m}_n_sims', 0):,} draws)")
            st.metric("P(actual winner finishes 1st)", ev.pct(coverage.get(f"{m}_winner_prob_rank_1")))
            st.metric("Actual top 3 inside simulated top 3 (by mean rank)", f"{coverage.get(f'{m}_actual_top3_covered_by_sim_top3_mean_rank')} of {coverage.get(f'{m}_actual_top3_n')}")
            st.metric("Actual top 5 (7 players incl. ties) inside simulated top 5", f"{coverage.get(f'{m}_actual_top5_covered_by_sim_top5_mean_rank')} of {coverage.get(f'{m}_actual_top5_n')}")
            st.metric("Actual top 10 inside simulated top 10", f"{coverage.get(f'{m}_actual_top10_covered_by_sim_top10_mean_rank')} of {coverage.get(f'{m}_actual_top10_n')}")
            st.metric("Exact-order probability of the actual top 3", ev.pct(coverage.get(f"{m}_exact_top3_order_prob"), 2),
                      help="Daicos > Smith > Bontempelli in that order, from the joint simulation draws")
            st.metric("P(actual top 3 are the top 3, any order)", ev.pct(coverage.get(f"{m}_top3_set_prob"), 2))
            st.metric("P(actual top 4 are the top 4, any order)", ev.pct(coverage.get(f"{m}_top4_set_prob"), 2))
    st.caption(coverage.get("top5_note", ""))

# ---------------------------------------------------------------- 3. Players
with tabs[2]:
    st.subheader("Player-level evaluation")
    s = season.copy()
    f1, f2, f3, f4 = st.columns(4)
    teams = ["All"] + sorted(s["team_id"].dropna().unique().tolist())
    team = f1.selectbox("Team", teams, format_func=lambda t: "All" if t == "All" else ev.display_team(t))
    vmin, vmax = int(s["actual_votes"].min()), int(s["actual_votes"].max())
    vr = f2.slider("Actual votes", vmin, vmax, (1, vmax))
    dis = f3.selectbox("Model disagreement (Prod vs Obj gap)", ["Any", ">= 2 votes", ">= 5 votes"])
    ou = f4.selectbox("Direction", ["Any", "All models over-predicted", "All models under-predicted", "Models split"])
    if team != "All":
        s = s[s["team_id"] == team]
    s = s[(s["actual_votes"] >= vr[0]) & (s["actual_votes"] <= vr[1])]
    if dis != "Any":
        s = s[s["prod_obj_gap"].abs() >= (2 if dis == ">= 2 votes" else 5)]
    errs = s[["production_error", "objective_error", "wheelo_error"]]
    if ou == "All models over-predicted":
        s = s[(errs > 0).all(axis=1) & errs.notna().all(axis=1)]
    elif ou == "All models under-predicted":
        s = s[(errs < 0).all(axis=1) & errs.notna().all(axis=1)]
    elif ou == "Models split":
        s = s[((errs > 0).any(axis=1)) & ((errs < 0).any(axis=1))]
    view = pd.DataFrame({
        "Player": s["player_name"], "Team": s["team_id"].map(ev.display_team), "Role": s["role"], "Actual": s["actual_votes"], "Actual rank": s["actual_rank"],
        "Prod EV": s["production_ev"].round(1), "Prod err": s["production_error"].round(1), "Prod rank": s["production_rank"],
        "Obj EV": s["objective_ev"].round(1), "Obj err": s["objective_error"].round(1), "Obj rank": s["objective_rank"],
        "Wheelo EV": s["wheelo_ev"].round(1), "Wheelo err": s["wheelo_error"].round(1), "Wheelo rank": s["wheelo_rank"],
        "Best source": s["best_source"].str.title(), "Worst source": s["worst_source"].str.title(), "Spread": s["cross_model_spread"].round(1),
    })
    st.caption(f"{len(view)} players shown. Error = EV minus actual (positive = over-predicted). Blank = source did not cover the player.")
    _table(view, "dl_players", "season_players", height=480)

    st.subheader("Highlights")
    full = season[season["in_production"] & season["in_objective"]].copy()
    def _show(title, df, cols, key, n=10):
        st.markdown(f"**{title}** (n = {len(df)})")
        if df.empty:
            st.caption("None.")
            return
        v = df.head(n)[cols].copy()
        v.columns = [c.replace("_ev", " EV").replace("_error", " err").replace("_", " ").title() for c in cols]
        st.dataframe(v.round(1), hide_index=True, width=W)
    base_cols = ["player_name", "team_id", "actual_votes", "production_ev", "objective_ev", "wheelo_ev", "production_error", "objective_error", "wheelo_error"]
    _show("Biggest over-predictions (consensus EV minus actual)", full.sort_values("consensus_error", ascending=False), base_cols + ["consensus_error"], "h1")
    _show("Biggest under-predictions", full.sort_values("consensus_error"), base_cols + ["consensus_error"], "h2")
    miss_all = full[(full["production_abs_error"] >= 5) & (full["objective_abs_error"] >= 5) & (full["wheelo_abs_error"].fillna(99) >= 5)]
    _show("Players all three sources missed by 5+ votes", miss_all.sort_values("actual_votes", ascending=False), base_cols, "h3")
    p_right = full[(full["production_abs_error"] <= 2) & (full["objective_abs_error"] >= 5)]
    _show("Production within 2, Objective off by 5+", p_right.sort_values("objective_abs_error", ascending=False), base_cols, "h4")
    o_right = full[(full["objective_abs_error"] <= 2) & (full["production_abs_error"] >= 5)]
    _show("Objective within 2, Production off by 5+", o_right.sort_values("production_abs_error", ascending=False), base_cols, "h5")
    w_uniq = full[(full["wheelo_abs_error"] + 2 <= full[["production_abs_error", "objective_abs_error"]].min(axis=1)) & (full["actual_votes"] >= 5)]
    _show("Wheelo uniquely closer (by 2+ votes vs both internal models, actual >= 5)", w_uniq.sort_values("actual_votes", ascending=False), base_cols, "h6")
    cons_fail = full[(full["prod_obj_gap"].abs() <= 2) & (full["consensus_error"].abs() >= 6)]
    _show("Consensus predictions that failed (models within 2 of each other, consensus off by 6+)", cons_fail.sort_values("consensus_error", key=abs, ascending=False), base_cols + ["consensus_error"], "h7")
    clear = full[(full["prod_obj_gap"].abs() >= 5) & ((full["production_abs_error"] - full["objective_abs_error"]).abs() >= 4)]
    _show("Disagreement cases where one model clearly won (gap 5+, error difference 4+)", clear.assign(winner=clear["best_source"].str.title()).sort_values("prod_obj_gap", key=abs, ascending=False), base_cols + ["winner"], "h8")

# ---------------------------------------------------------------- 4. Match accuracy
with tabs[3]:
    st.subheader("Accuracy by round")
    rt = round_table.pivot(index="round", columns="model", values="hit_3_rate")
    st.line_chart(rt, height=260)
    st.caption("Share of matches in each round where the model's highest-P3 player was the actual 3-vote player.")
    rt2 = round_table[["model", "round", "n_matches", "hit_3", "hit_3_rate", "in_top3_rate", "exact_321", "unordered_top3", "mean_ev_mae", "mean_log_loss"]]
    with st.expander("Round-by-round table"):
        _table(rt2, "dl_round", "round_table")
    best = round_table.groupby("model").apply(lambda g: g.loc[g["hit_3_rate"].idxmax(), "round"], include_groups=False)
    worst = round_table.groupby("model").apply(lambda g: g.loc[g["hit_3_rate"].idxmin(), "round"], include_groups=False)
    st.markdown("**Strongest / weakest rounds (by 3-vote hit rate):** " + "; ".join(f"{m}: best R{int(best[m])}, worst R{int(worst[m])}" for m in MODELS) + ". Round 0 = Opening Round.")

    st.subheader("Match-level misses and agreement")
    mt = match_table.copy()
    piv = mt.pivot(index="match_id", columns="model", values="model_top_player")
    hits = mt.pivot(index="match_id", columns="model", values="hit_3")
    a3 = mt.drop_duplicates("match_id").set_index("match_id")
    all_wrong = hits[(hits[MODELS] == False).all(axis=1)].index
    same_wrong = [m for m in all_wrong if piv.loc[m, MODELS].nunique() == 1]
    one_right = hits[(hits[MODELS].sum(axis=1) == 1)].index
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches where all three named the 3-voter", int((hits[MODELS] == True).all(axis=1).sum()))
    c2.metric("All three wrong", len(all_wrong))
    c3.metric("All agreed on the same wrong player", len(same_wrong))
    c4.metric("Exactly one model found the 3-voter", len(one_right))
    mode = st.radio("Show", ["Models agreed but were wrong", "Exactly one model identified the true poller", "All matches"], horizontal=True)
    if mode == "Models agreed but were wrong":
        ids = same_wrong
    elif mode == "Exactly one model identified the true poller":
        ids = list(one_right)
    else:
        ids = list(a3.index)
    rows = []
    for mid in ids:
        r = a3.loc[mid]
        row = {"Round": int(r["round"]), "Match": f"{ev.display_team(r['home_team'])} v {ev.display_team(r['away_team'])}", "Actual 3": r["actual_3_player"],
               "Actual 2": r["actual_2_player"], "Actual 1": r["actual_1_player"], "3-voter role": r["actual_3_role"]}
        for m in MODELS:
            sub = mt[(mt["match_id"] == mid) & (mt["model"] == m)]
            if len(sub):
                sr = sub.iloc[0]
                row[f"{m} pick"] = f"{sr['model_top_player']} ({sr['model_top_p3'] * 100:.0f}%)" + (" ✓" if sr["hit_3"] else "")
                row[f"{m} 3-voter P3 rank"] = sr["actual_3_p3_rank"]
        row["match_id"] = mid
        rows.append(row)
    miss_df = pd.DataFrame(rows).sort_values(["Round", "Match"]) if rows else pd.DataFrame()
    _table(miss_df.drop(columns=["match_id"], errors="ignore"), "dl_matches", "match_misses", height=420)
    if not miss_df.empty:
        pick = st.selectbox("Open a match in Match Detail", ["--"] + [f"R{r['Round']} {r['Match']} ({r['match_id']})" for _, r in miss_df.iterrows()])
        if pick != "--" and st.button("Open Match Detail"):
            st.session_state["match_detail_preselect_match_id"] = pick.split("(")[-1].rstrip(")")
            st.switch_page("pages/6_Match_Detail.py")
    with st.expander("Full match table (all models, all metrics)"):
        _table(match_table, "dl_match_table", "match_table", height=400)

# ---------------------------------------------------------------- 5. Disagreement
with tabs[4]:
    st.subheader("When Production and Objective disagreed, who was closer?")
    dv = disagreement.rename(columns={"gap_bin_votes": "|Prod - Obj| EV gap", "n_players": "players", "n_with_actual_votes": "with actual votes",
                                      "production_closer": "Prod closer", "objective_closer": "Obj closer", "production_mae": "Prod MAE", "objective_mae": "Obj MAE",
                                      "mean_min_abs_error": "better model's |err|", "mean_max_abs_error": "worse model's |err|", "mean_consensus_abs_error": "mean-of-two |err|",
                                      "wheelo_tiebreaks": "Wheelo tie-breaks", "wheelo_tiebreak_correct": "Wheelo sided with closer model", "wheelo_tiebreak_rate": "Wheelo tie-break rate"})
    _table(dv, "dl_dis", "disagreement_summary")
    big = disagreement.set_index("gap_bin_votes").loc["5+"]
    st.markdown(
        f"- In the **5+ vote** disagreement bin (n = {int(big['n_players'])}), Production was closer {int(big['production_closer'])} times and Objective {int(big['objective_closer'])} "
        f"(MAE {big['production_mae']:.2f} vs {big['objective_mae']:.2f}).\n"
        f"- **Disagreement predicted error**: even the better model's absolute error rises from "
        f"{disagreement.set_index('gap_bin_votes').loc['0-2', 'mean_min_abs_error']:.2f} (0-2 bin) to {big['mean_min_abs_error']:.2f} (5+ bin).\n"
        f"- **Wheelo as a tie-break**: where Wheelo sat closer to one internal model, that model was the closer one {big['wheelo_tiebreak_rate'] * 100:.0f}% of the time in the 5+ bin "
        f"(n = {int(big['wheelo_tiebreaks'])}) and {disagreement.set_index('gap_bin_votes').loc['all', 'wheelo_tiebreak_rate'] * 100:.0f}% overall."
    )
    st.subheader("Disagreement cases (gap of 5+ votes)")
    cv = cases[["player_name", "team_id", "role", "actual_votes", "production_ev", "objective_ev", "wheelo_ev", "prod_obj_gap", "closer", "wheelo_sided_with",
                "wheelo_tiebreak_correct", "production_rank", "objective_rank", "actual_rank", "rank_closer"]].copy()
    cv["team_id"] = cv["team_id"].map(ev.display_team)
    cv.columns = ["Player", "Team", "Role", "Actual", "Prod EV", "Obj EV", "Wheelo EV", "Prod - Obj", "Closer", "Wheelo sided with", "Wheelo correct", "Prod rank", "Obj rank", "Actual rank", "Rank closer"]
    _table(cv.round(1), "dl_cases", "disagreement_cases")
    st.markdown("**Rank reversals**: rows where the models ordered the player differently by 10+ places.")
    rr = cases[cases["rank_gap"].abs() >= 10][["player_name", "actual_rank", "production_rank", "objective_rank", "rank_closer"]]
    st.dataframe(rr.rename(columns={"player_name": "Player", "actual_rank": "Actual rank", "production_rank": "Prod rank", "objective_rank": "Obj rank", "rank_closer": "Closer"}), hide_index=True, width=W)
    st.markdown("**Archetype strengths** -- see Bias & Roles for the role-level error split each model shows.")

# ---------------------------------------------------------------- 6. Bias
with tabs[5]:
    st.subheader("Season-total error by role and team")
    scope = st.radio("Population", ["relevant (actual>=1 or EV>=1)", "all"], horizontal=True, help="'all' includes hundreds of zero-vote, near-zero-EV players that pull every mean toward 0.")
    dim = st.radio("Dimension", ["role", "team_id"], horizontal=True, format_func=lambda x: "Role / position group" if x == "role" else "Team")
    sb = season_bias[(season_bias["scope"] == scope) & (season_bias["dimension"] == dim)]
    piv = sb.pivot(index="group", columns="model", values=["n", "mean_error", "mae", "over_predicted", "under_predicted"])
    flat = pd.DataFrame(index=piv.index)
    for m in MODELS:
        flat[f"{m} n"] = piv[("n", m)]
        flat[f"{m} mean err"] = piv[("mean_error", m)]
        flat[f"{m} MAE"] = piv[("mae", m)]
        flat[f"{m} over/under"] = piv[("over_predicted", m)].astype("Int64").astype(str) + "/" + piv[("under_predicted", m)].astype("Int64").astype(str)
    flat = flat.reset_index().rename(columns={"group": "Group"})
    if dim == "team_id":
        flat["Group"] = flat["Group"].map(lambda g: "ALL" if g == "ALL" else ev.display_team(g))
    _table(flat, "dl_bias", "season_bias")
    st.caption("Mean error = EV minus actual (positive = over-predicted). Over/under counts use a 0.5-vote threshold. Groups with n < 10 are small samples -- do not over-interpret.")

    st.subheader("Defender blind spot test (match level)")
    mb = match_bias.copy()
    mbv = mb.pivot(index="actual_3_role", columns="model", values=["n_matches", "hit_3_rate", "in_top3_rate", "mean_p3_of_actual_3"])
    out = pd.DataFrame(index=mbv.index)
    out["3-voters (n)"] = mbv[("n_matches", "Production")].astype(int)
    for m in MODELS:
        out[f"{m} found 3-voter"] = mbv[("hit_3_rate", m)].map(lambda x: ev.pct(x, 0))
        out[f"{m} mean P3 on 3-voter"] = mbv[("mean_p3_of_actual_3", m)].round(3)
    share = mb[mb["model"] == "Production"].set_index("actual_3_role")
    out["share of all 3-voters"] = share["share_of_all_3_voters"].map(lambda x: ev.pct(x, 0))
    for m in MODELS:
        sp = mb[mb["model"] == m].set_index("actual_3_role")["share_of_model_top_picks"] if "share_of_model_top_picks" in mb else None
        if sp is not None:
            out[f"share of {m} top picks"] = sp.reindex(out.index).map(lambda x: ev.pct(x, 0) if pd.notna(x) else "0%")
    out = out.reset_index().rename(columns={"actual_3_role": "Role of actual 3-voter"})
    _table(out, "dl_mbias", "match_bias")
    st.caption("Roles are the existing lagged 2026 classification (prior-games proxy, UNKNOWN where history was insufficient). "
               "Phase 4 found key defenders were identified as 3-vote winners only 12.5% of the time historically; the 2026 rows above are the direct test, with small defender samples.")

# ---------------------------------------------------------------- 7. Betting
with tabs[6]:
    st.subheader("Pre-count bookmaker markets, settled against the actual votes")
    st.caption("Prices are the 2026-09-18 Neds / PointsBet snapshot (three days before the count). Flat 1-unit stake per selection is a retrospective analytical metric only -- "
               "it is not a staking plan, and model accuracy (hit rate vs probability) is reported separately from profitability (ROI). Pushes return the stake; dead heats pay "
               "the standard fractional reduction; ties in H2H are pushes. Ranks include ineligible players, matching the bookmakers' own market wording.")
    sett = settled[settled["result"] != "unsettleable"]
    mk = bet_summary[(bet_summary["grouping"] == "market_type") & (bet_summary["subset"] == "all priced selections")].copy()
    mv = mk[["group", "bets", "wins", "pushes", "losses", "hit_rate", "avg_model_prob_conservative", "avg_implied_prob", "flat_unit_pnl", "roi", "small_sample"]].rename(columns={
        "group": "Market", "hit_rate": "Hit rate", "avg_model_prob_conservative": "Avg model prob (conservative)", "avg_implied_prob": "Avg implied prob", "flat_unit_pnl": "Flat 1u P/L", "roi": "ROI", "small_sample": "n<20"})
    _table(mv, "dl_bet_mk", "betting_by_market", column_config={"Hit rate": st.column_config.NumberColumn(format="%.1%"), "ROI": st.column_config.NumberColumn(format="%.1%"),
                                                                  "Avg model prob (conservative)": st.column_config.NumberColumn(format="%.1%"), "Avg implied prob": st.column_config.NumberColumn(format="%.1%")})
    st.caption("'ALL' backs every priced selection including both sides of every O/U and every runner in the winner market -- a sanity baseline (it should lose roughly the bookmaker margin), not a strategy.")

    market_labels = {"TO_POLL_A_VOTE": "A. To Poll a Vote", "PLAYER_H2H": "B. Player H2H", "TEAM_VOTES_OU": "C/D. Team votes O/U", "PLAYER_VOTES_OU": "D. Player votes O/U",
                     "X_PLUS_VOTES": "E. X+ votes", "TOP_N": "F. Top-N finish", "WINNER": "F. Winner", "EXACT_POSITION": "F. Exact position"}
    choice = st.selectbox("Market detail", list(market_labels), format_func=lambda k: market_labels[k])
    g = sett[sett["market_type"] == choice].copy()
    detail = pd.DataFrame({
        "Bookmaker": g["source"], "Market": g["market_name"], "Selection": g["selection"], "Line/threshold/N": g["line"].fillna(g["threshold"]).fillna(g["n"]).fillna(g["position"]),
        "Odds": g["odds"], "Implied": g["implied_probability"], "Prod prob": g["production_probability"], "Obj prob": g["objective_probability"],
        "Bet Value": g["bet_value_label"], "Likelihood": g["likelihood_band"], "Wheelo": g["wheelo_support_label"].str.replace("_", " ").str.title(),
        "Actual": g["actual_value"], "Result": g["result"], "Detail": g["settlement_detail"], "Flat 1u P/L": g["flat_unit_pnl"].round(3),
        "Eligibility-sensitive": g["settlement_sensitive_to_eligibility"],
    })
    if choice == "PLAYER_H2H":
        detail.insert(4, "Opponent", g["opponent_name"]); detail.insert(5, "Opponent actual", g["opponent_actual_votes"])
    _table(detail.sort_values(["Bookmaker", "Market"]), f"dl_bet_{choice}", f"betting_{choice.lower()}", height=420,
           column_config={"Implied": st.column_config.NumberColumn(format="%.1%"), "Prod prob": st.column_config.NumberColumn(format="%.1%"), "Obj prob": st.column_config.NumberColumn(format="%.1%")})
    sub = bet_summary[(bet_summary["grouping"] == "market_type x bet_value") & (bet_summary["group"].str.startswith(choice + " |"))]
    if len(sub):
        st.markdown("**By Bet Value tier within this market**")
        st.dataframe(sub[["group", "bets", "wins", "pushes", "hit_rate", "avg_model_prob_conservative", "avg_implied_prob", "flat_unit_pnl", "roi"]].rename(columns={"group": "Tier"}), hide_index=True, width=W,
                     column_config={"hit_rate": st.column_config.NumberColumn(format="%.1%"), "roi": st.column_config.NumberColumn(format="%.1%")})
    if choice == "TEAM_VOTES_OU":
        st.markdown("**Team leader (model accuracy only -- no team-top-poller price was captured pre-count)**")
        tl = team_leaders.copy()
        tl["team_id"] = tl["team_id"].map(ev.display_team)
        st.dataframe(tl[["team_id", "actual_leader", "actual_leader_votes", "actual_tie", "production_predicted_leader", "production_hit", "objective_predicted_leader", "objective_hit", "wheelo_predicted_leader", "wheelo_hit", "team_actual_votes", "production_team_ev_total", "objective_team_ev_total", "wheelo_team_ev_total"]].rename(columns=lambda c: c.replace("_", " ")), hide_index=True, width=W)
        st.caption("Hits: " + ", ".join(f"{m} {int(tl[f'{m.lower()}_hit'].sum())}/18" for m in MODELS) + ". A tied actual leader counts as a hit for either tied player.")
    unmod = settled[settled["market_type"] == "UNMODELLED"]
    st.caption(f"Not evaluated: {len(unmod)} unmodelled selections (Leader After Round N, exotics, group markets -- no pre-count model probability existed) and "
               f"{int(((settled['result'] == 'unsettleable') & (settled['market_type'] != 'UNMODELLED')).sum())} selections whose player identity was flagged ambiguous pre-count (Chad Warner, Charlie Cameron) and carried no model probability.")

    st.subheader("Betting signal analysis")
    st.caption("Does each pre-count signal separate winners from losers? Hit rate and ROI by tier, with sample sizes. No claim of future profitability.")
    for grouping, title, order in (("bet_value", "Bet Value tier", ["Strong Bet Value + Wheelo Support", "Strong Bet Value", "Moderate Bet Value", "Speculative Bet Value", "Model Disagreement", "No Value"]),
                                   ("likelihood_band", "Likelihood band (conservative model probability)", ["Very High", "High", "Moderate", "Low", "Very Low"]),
                                   ("wheelo_support", "Wheelo support", ["STRONG_WHEELO_SUPPORT", "PARTIAL_WHEELO_SUPPORT", "WHEELO_NEUTRAL", "WHEELO_DISAGREES", "INSUFFICIENT_WHEELO_DATA"]),
                                   ("model_agreement", "Production / Objective agreement on value", ["both see value", "only one sees value", "neither sees value", "one model missing"]),
                                   ("internal_gap_band", "Model disagreement (|Prod - Obj| probability gap)", ["<=7.5pp", "7.5-15pp", "15-30pp", ">30pp"]),
                                   ("edge_band", "Production price edge (model prob - implied)", ["<=0pp", "0-5pp", "5-15pp", ">15pp"]),
                                   ("prob_band", "Conservative probability band", None)):
        sub = bet_summary[(bet_summary["grouping"] == grouping) & (bet_summary["subset"] == "all priced selections")].copy()
        if order:
            sub["_o"] = sub["group"].map({k: i for i, k in enumerate(order)}); sub = sub.sort_values("_o")
        st.markdown(f"**{title}**")
        st.dataframe(sub[["group", "bets", "wins", "pushes", "hit_rate", "avg_model_prob_conservative", "avg_implied_prob", "flat_unit_pnl", "roi", "small_sample"]].rename(columns={
            "group": title, "hit_rate": "Hit rate", "avg_model_prob_conservative": "Avg model prob", "avg_implied_prob": "Avg implied", "flat_unit_pnl": "Flat 1u P/L", "roi": "ROI", "small_sample": "n<20"}),
            hide_index=True, width=W, column_config={"Hit rate": st.column_config.NumberColumn(format="%.1%"), "ROI": st.column_config.NumberColumn(format="%.1%"),
                                                     "Avg model prob": st.column_config.NumberColumn(format="%.1%"), "Avg implied": st.column_config.NumberColumn(format="%.1%")})
    _dl(bet_summary, "betting_summary", "dl_bet_summary")
    _dl(settled, "betting_settled_all_rows", "dl_bet_all")

# ---------------------------------------------------------------- 8. Calibration
with tabs[7]:
    st.subheader("Calibration: predicted probability vs actual frequency")
    series = st.selectbox("Series", cal_summary["series"].unique().tolist())
    cs = cal_summary[cal_summary["series"] == series]
    st.dataframe(cs[["model", "n", "ece", "brier", "base_rate", "mean_predicted"]].rename(columns={"n": "n", "ece": "ECE", "brier": "Brier", "base_rate": "Actual rate", "mean_predicted": "Mean predicted"}), hide_index=True, width=W)
    cb = cal_bins[cal_bins["series"] == series]
    chart = cb.pivot(index="bin", columns="model", values="actual_frequency")
    chart.index = pd.CategoricalIndex(chart.index, categories=[f"{i*10}-{(i+1)*10}%" for i in range(10)], ordered=True); chart = chart.sort_index()
    st.line_chart(chart, height=260)
    st.caption("Line = actual frequency within each predicted-probability bin (a perfectly calibrated series rises 0.05, 0.15, ... 0.95). Bins with n = 0 are blank.")
    _table(cb.rename(columns={"mean_predicted": "mean predicted", "actual_frequency": "actual frequency", "gap_pp": "gap (pp)"}), "dl_cal", "calibration_bins")
    st.caption("ECE = expected calibration error (bin-weighted |actual - predicted|). Brier/log loss are reported only where a genuine probability exists: "
               "Production and Objective (simulation / Plackett-Luce), Wheelo's published match P3 %, and bookmaker implied probabilities. Market series settle dead heats as fractional wins and drop pushes.")
    _dl(cal_summary, "calibration_summary", "dl_cal_sum")

# ---------------------------------------------------------------- 9. Learnings
with tabs[8]:
    st.subheader("What we learned for 2027")
    st.caption("Deterministic, generated from the evaluation tables above (no editorial text). Findings and candidates for investigation -- not retraining decisions.")
    for i, l in enumerate(learnings, start=1):
        st.markdown(f"**{i}. {l['title']}** (n = {l['n']})  \n{l['finding']}  \n<span style='color:gray;font-size:0.85em'>Source: {l['evidence']}</span>", unsafe_allow_html=True)
