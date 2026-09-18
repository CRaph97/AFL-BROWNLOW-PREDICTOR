"""
Brownlow Betting Opportunities (Neds / PointsBet) -- decision-support only.

Never places a bet, sizes a stake, or automates a bookmaker account. Never
scrapes on page load: everything here is read from
data/betting/processed/*, built by `python scripts/refresh_brownlow_odds.py`.
See docs/BETTING_OPPORTUNITIES.md for the full methodology.

Redesigned into 9 sections for readability -- a reader should be able to
understand every bet without knowing this project's internal market_type /
confidence vocabulary. All numeric values (probabilities, EV, edges,
classifications) are unchanged from the pricing/classification pipeline --
this file only formats and organises them for display.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import betting_opportunities as bo
from dashboard import data as d
from dashboard import external_data as ed

st.set_page_config(page_title="Brownlow Betting Opportunities", layout="wide")
d.highlight_objective_stats_nav()
st.title("Brownlow Betting Opportunities")
st.caption(
    "Neds / PointsBet markets priced against Production and Objective's own simulations, "
    "with Wheelo as independent quantitative corroboration. Decision-support only -- this "
    "page never places a bet, sizes a stake, or automates a bookmaker account, and never "
    "changes any model's predictions."
)

summary = bo.load_refresh_summary()
raw_opportunities = bo.load_opportunities()
combinations = bo.load_combinations()
price_comparison = bo.load_price_comparison()
external_overview = ed.load_external_overview()

# --------------------------------------------------------------------------
# Top summary
# --------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Odds refresh", summary.get("refresh_completed_at", "never")[:19] if summary.get("refresh_completed_at") else "never")
c2.metric("Markets scraped", summary.get("markets_scraped", 0))
c3.metric("Markets modelled", summary.get("markets_modelled", 0))
c4.metric("High Confidence", summary.get("high_confidence_count", 0))
c5.metric("Medium / Speculative", f"{summary.get('medium_confidence_count', 0)} / {summary.get('speculative_count', 0)}")

with st.expander("Source status / provenance", expanded=raw_opportunities.empty):
    if not summary:
        st.warning("No refresh has been run yet. Run `python scripts/refresh_brownlow_odds.py`.")
    for s in summary.get("source_statuses", []):
        icon = "✅" if s.get("status") == "OK" else "⚠️"
        st.markdown(f"{icon} **{s.get('source')}** -- {s.get('status')}  \n*{s.get('detail', '')}*")
    st.caption("Full methodology: docs/BETTING_OPPORTUNITIES.md")

if raw_opportunities.empty:
    st.info(
        "**No priced betting opportunities are currently available.** Run "
        "`python scripts/refresh_brownlow_odds.py` to fetch fresh markets. See Source status "
        "above and docs/BETTING_OPPORTUNITIES.md for the full, honest explanation of the most "
        "recent attempt."
    )
    st.stop()

opportunities = bo.prepare_display(raw_opportunities)

st.divider()

# ==========================================================================
# 1. TOP OPPORTUNITIES
# ==========================================================================
st.header("1. Top Opportunities")
qualifying = bo.qualifying_opportunities(opportunities)

if qualifying.empty:
    st.caption(
        "No opportunities currently qualify (High/Medium Confidence or a genuinely positive-EV "
        "Speculative pick, with no data-quality flags). See section 9 for the full raw market list."
    )
else:
    if "top_opps_shown" not in st.session_state:
        st.session_state.top_opps_shown = 10
    n_shown = min(st.session_state.top_opps_shown, len(qualifying))
    shown = qualifying.head(n_shown)

    display_cols = {
        "bet": "Selection", "best_bookmaker": "Best bookmaker", "best_odds": "Best odds",
        "implied_probability": "Implied %", "production_probability": "Production %",
        "objective_probability": "Objective %", "conservative_edge_pp": "Model gap (pp)",
        "wheelo_support": "Wheelo evidence", "confidence_badge": "Confidence",
    }
    # Each opportunity row already IS one bookmaker's price (source-specific),
    # so "best odds" for a single row is just that row's own odds -- the
    # cross-bookmaker pivot (with_bookmaker_odds) is used in sections 2/3/5/7
    # instead, where multiple sources need collapsing into one line.
    shown = shown.assign(best_bookmaker=shown["source"].apply(lambda s: "PointsBet" if str(s).startswith("pointsbet") else "Neds"),
                          best_odds=shown["odds"])
    table = shown[[c for c in display_cols if c in shown.columns]].rename(columns=display_cols)
    for pct_col in ["Implied %", "Production %", "Objective %"]:
        if pct_col in table.columns:
            table[pct_col] = shown[{"Implied %": "implied_probability", "Production %": "production_probability",
                                     "Objective %": "objective_probability"}[pct_col]].apply(bo.format_pct)
    if "Best odds" in table.columns:
        table["Best odds"] = shown["odds"].apply(bo.format_odds)
    if "Model gap (pp)" in table.columns:
        table["Model gap (pp)"] = shown["internal_gap_pp"].apply(bo.format_pp)
    st.dataframe(table, use_container_width=True, hide_index=True)

    for _, row in shown.iterrows():
        with st.expander(f"{row['bet']} -- {row['confidence_badge']}"):
            st.markdown(f"**Why this classification:** {row.get('classification_rationale', 'n/a')}")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Production EV", bo.format_ev(row.get("production_ev")))
            cc2.metric("Objective EV", bo.format_ev(row.get("objective_ev")))
            cc3.metric("Implied probability", bo.format_pct(row.get("implied_probability")))
            if pd.notna(row.get("wheelo_ev")):
                st.markdown(f"- **Wheelo:** EV {row['wheelo_ev']:.2f}, rank {int(row['wheelo_rank']) if pd.notna(row.get('wheelo_rank')) else '?'} -- {row.get('wheelo_support')}")
            else:
                st.markdown("- **Wheelo:** insufficient data for this selection")
            st.markdown(f"- **External context:** {row.get('external_support', 'n/a')}")
            st.markdown(f"- **Settlement notes:** {row.get('data_quality_flags') or 'none'}")

    st.caption(f"Showing {n_shown} of {len(qualifying)} qualifying opportunities.")
    if n_shown < len(qualifying):
        if st.button("Show 5 more"):
            st.session_state.top_opps_shown += 5
            st.rerun()

st.divider()

# ==========================================================================
# 2. FINISHING POSITION EXPLORER
# ==========================================================================
st.header("2. Finishing Position Explorer")
threshold_options = ["Winner"] + [f"Top {int(n)}" for n in sorted(opportunities.loc[opportunities["market_type"] == "TOP_N", "n"].dropna().unique())]
if len(threshold_options) == 1 and "WINNER" not in opportunities["market_type"].values:
    st.caption("No finishing-position markets currently available.")
else:
    choice = st.radio("Finishing threshold", threshold_options, horizontal=True)
    if choice == "Winner":
        sub = opportunities[opportunities["market_type"] == "WINNER"]
    else:
        n_val = float(choice.split(" ")[1])
        sub = opportunities[(opportunities["market_type"] == "TOP_N") & (opportunities["n"] == n_val)]
    only_positive = st.toggle("Only show positive edge", value=False)
    piv = bo.with_bookmaker_odds(sub, ["player_id"])
    if not piv.empty:
        piv["edge_indicator"] = ((piv["production_edge_pp"] > 0) | (piv["objective_edge_pp"] > 0)).map({True: "✅", False: ""})
        if only_positive:
            piv = piv[(piv["production_edge_pp"] > 0) | (piv["objective_edge_pp"] > 0)]
        show = pd.DataFrame({
            "Player": piv["player_name"],
            "Production %": piv["production_probability"].apply(bo.format_pct),
            "Objective %": piv["objective_probability"].apply(bo.format_pct),
            "Wheelo": piv.apply(lambda r: f"{r['wheelo_ev']:.2f} (rank {int(r['wheelo_rank'])})" if pd.notna(r.get("wheelo_ev")) and pd.notna(r.get("wheelo_rank")) else "N/A", axis=1),
            "Neds odds": piv["neds_odds"].apply(bo.format_odds),
            "PointsBet odds": piv["pointsbet_odds"].apply(bo.format_odds),
            "Best odds": piv["best_odds"].apply(bo.format_odds),
            "Best bookmaker": piv["best_bookmaker"],
            "Implied %": piv["implied_probability"].apply(bo.format_pct),
            "Confidence": piv["confidence_badge"],
            "+Edge": piv["edge_indicator"],
        }).sort_values("Production %", ascending=False)
        st.dataframe(show, use_container_width=True, hide_index=True, height=500)
    else:
        st.caption("No selections for this threshold.")

st.divider()

# ==========================================================================
# 3. TO POLL A VOTE
# ==========================================================================
st.header("3. To Poll a Vote")
tpav = opportunities[opportunities["market_type"] == "TO_POLL_A_VOTE"]
if tpav.empty:
    st.caption("No 'To Poll a Vote' markets currently available.")
else:
    search3 = st.text_input("Player search", key="tpav_search")
    piv = bo.with_bookmaker_odds(tpav, ["player_id"])
    if search3:
        piv = piv[piv["player_name"].str.contains(search3, case=False, na=False)]
    piv = piv.assign(_sort=piv["conservative_internal_probability"] - piv["implied_probability"]).sort_values("_sort", ascending=False)
    show = pd.DataFrame({
        "Player": piv["player_name"],
        "Neds odds": piv["neds_odds"].apply(bo.format_odds),
        "PointsBet odds": piv["pointsbet_odds"].apply(bo.format_odds),
        "Best odds": piv["best_odds"].apply(bo.format_odds),
        "Implied %": piv["implied_probability"].apply(bo.format_pct),
        "Production %": piv["production_probability"].apply(bo.format_pct),
        "Objective %": piv["objective_probability"].apply(bo.format_pct),
        "Wheelo support": piv["wheelo_support"],
        "Confidence": piv["confidence_badge"],
    })
    st.dataframe(show, use_container_width=True, hide_index=True, height=500)

st.divider()

# ==========================================================================
# 4. LEADER AFTER ROUND X
# ==========================================================================
st.header("4. Leader After Round X")
st.caption(
    "Cumulative totals using match-level data only, on this app's official normalized round. "
    "Wheelo's own Round column uses the same convention (verified)."
)
prod_votes = pd.read_csv("reports/2026_predicted_votes.csv")[["round", "player_id", "player_name", "team_id", "expected_votes"]]
prod_votes["player_id"] = prod_votes["player_id"].astype(str)
obj_votes = pd.read_csv("reports/2026_objective_votes.csv")[["round", "player_id", "player_name", "team_id", "expected_votes"]]
obj_votes["player_id"] = obj_votes["player_id"].astype(str)
obj_votes = obj_votes[~obj_votes["player_id"].str.startswith("NOID")]
wheelo_ml = ed.load_wheelo_match_level()

min_round, max_round = int(prod_votes["round"].min()), int(prod_votes["round"].max())
round_n = st.slider("Leader after round", min_round, max_round, max_round)
top_n_round = st.radio("Show top", [5, 10, 20], index=1, horizontal=True, key="round_top_n")


def _cum(df, value_col, name_col, team_col):
    sub = df[df["round"] <= round_n]
    agg = sub.groupby("player_id", as_index=False).agg(cum=(value_col, "sum"), player_name=(name_col, "first"), team_id=(team_col, "first"))
    agg["rank"] = agg["cum"].rank(ascending=False, method="min")
    return agg


prod_cum = _cum(prod_votes, "expected_votes", "player_name", "team_id").rename(columns={"cum": "production_cum", "rank": "production_rank"})
obj_cum = _cum(obj_votes, "expected_votes", "player_name", "team_id").rename(columns={"cum": "objective_cum", "rank": "objective_rank"})
wheelo_resolved = wheelo_ml[wheelo_ml["match_status"] == "resolved"]
wheelo_cum = _cum(wheelo_resolved, "wheelo_ev", "wheelo_player_name", "team_id").rename(columns={"cum": "wheelo_cum", "rank": "wheelo_rank"})

merged4 = prod_cum[["player_id", "player_name", "team_id", "production_cum", "production_rank"]].merge(
    obj_cum[["player_id", "objective_cum", "objective_rank"]], on="player_id", how="outer"
).merge(wheelo_cum[["player_id", "wheelo_cum", "wheelo_rank"]], on="player_id", how="outer")
merged4["player_name"] = merged4["player_name"].fillna(merged4["player_id"].map(wheelo_cum.set_index("player_id")["player_name"]))
merged4["consensus_cum"] = merged4[["production_cum", "objective_cum", "wheelo_cum"]].mean(axis=1, skipna=True)
merged4["consensus_rank"] = merged4["consensus_cum"].rank(ascending=False, method="min")
ranks = merged4[["production_rank", "objective_rank", "wheelo_rank"]]
merged4["disagreement"] = ranks.max(axis=1) - ranks.min(axis=1)

display4 = merged4.sort_values("consensus_rank", na_position="last").head(top_n_round)
st.dataframe(display4, use_container_width=True, hide_index=True, height=450)

st.caption(f"Cumulative trajectory -- top {min(top_n_round, 10)} by consensus")
leaders = display4.head(min(top_n_round, 10))["player_id"].tolist()
traj_rows = []
for pid in leaders:
    series = prod_votes[prod_votes["player_id"] == pid].groupby("round")["expected_votes"].sum().cumsum()
    name = merged4.loc[merged4["player_id"] == pid, "player_name"].iloc[0]
    for r, v in series.items():
        traj_rows.append({"round": r, "player": name, "cumulative_production_ev": v})
if traj_rows:
    traj_df = pd.DataFrame(traj_rows).pivot(index="round", columns="player", values="cumulative_production_ev")
    st.line_chart(traj_df)

st.divider()

# ==========================================================================
# 5. TEAM EXPLORER
# ==========================================================================
st.header("5. Team Explorer")
teams = d.team_list()
team_choice = st.selectbox("Select team", teams, format_func=d._display_team)

team_breakdown_df = d.team_breakdown(team_choice)
team_prod_total = team_breakdown_df["production_ev"].sum() if "production_ev" in team_breakdown_df.columns else None
team_obj_total = team_breakdown_df["objective_ev"].sum() if "objective_ev" in team_breakdown_df.columns else None
team_wheelo_total = external_overview.loc[external_overview["team_id"] == team_choice, "wheelo_ev"].sum()

tc1, tc2, tc3 = st.columns(3)
tc1.metric("Production team total EV", f"{team_prod_total:.1f}" if team_prod_total is not None else "N/A")
tc2.metric("Objective team total EV", f"{team_obj_total:.1f}" if team_obj_total is not None else "N/A")
tc3.metric("Wheelo team total EV", f"{team_wheelo_total:.1f}" if pd.notna(team_wheelo_total) else "N/A")

team_markets = opportunities[(opportunities["market_type"] == "TEAM_VOTES_OU") & (opportunities["team_id"] == team_choice)]
if not team_markets.empty:
    piv = bo.with_bookmaker_odds(team_markets, ["team_id", "line", "side"])
    st.caption("Bookmaker team-total markets for this team:")
    st.dataframe(pd.DataFrame({
        "Bet": piv["bet"], "Neds odds": piv["neds_odds"].apply(bo.format_odds),
        "PointsBet odds": piv["pointsbet_odds"].apply(bo.format_odds),
        "Best odds": piv["best_odds"].apply(bo.format_odds), "Best bookmaker": piv["best_bookmaker"],
    }), use_container_width=True, hide_index=True)
else:
    st.caption("No bookmaker team-total market currently available for this team.")

st.subheader("Team player leaderboard (both models)")
rankings = d.dual_model_team_rankings(team_choice)
wheelo_team = external_overview[external_overview["team_id"] == team_choice][["player_id", "wheelo_ev", "wheelo_rank"]]
wheelo_team["player_id"] = wheelo_team["player_id"].astype(str)
rankings["player_id"] = rankings["player_id"].astype(str)
rankings = rankings.merge(wheelo_team, on="player_id", how="left")
st.dataframe(
    rankings[["player_name", "production_ev", "production_team_rank", "objective_ev", "objective_team_rank", "wheelo_ev", "wheelo_rank"]]
    .sort_values("production_team_rank"),
    use_container_width=True, hide_index=True, height=400,
)

st.subheader("Available player markets for this team")
team_player_markets = opportunities[opportunities["player_id"].astype(str).isin(rankings["player_id"])]
if team_player_markets.empty:
    st.caption("No player-level bookmaker markets currently resolved to this team's players.")
else:
    st.dataframe(pd.DataFrame({
        "Bet": team_player_markets["bet"], "Best odds": team_player_markets["odds"].apply(bo.format_odds),
        "Bookmaker": team_player_markets["source"], "Confidence": team_player_markets["confidence_badge"],
    }), use_container_width=True, hide_index=True, height=300)

st.divider()

# ==========================================================================
# 6. PLAYER COMPARISON
# ==========================================================================
st.header("6. Player Comparison")
cmp = d.load_dual_model_comparison()
names6 = sorted(cmp["player_name"].tolist())
p6a, p6b = st.columns(2)
player_a = p6a.selectbox("Player A", names6, index=names6.index("Nick Daicos") if "Nick Daicos" in names6 else 0, key="cmp_a")
player_b = p6b.selectbox("Player B", names6, index=1 if len(names6) > 1 else 0, key="cmp_b")

if player_a == player_b:
    st.warning("Select two different players.")
else:
    def _wheelo_row(name):
        m = external_overview[external_overview["player_name"] == name]
        return m.iloc[0] if not m.empty else None

    def _player_card(col, name):
        row = cmp[cmp["player_name"] == name].iloc[0]
        w = _wheelo_row(name)
        col.subheader(name)
        col.metric("Production EV", f"{row['production_ev']:.2f}" if row["in_production"] else "N/A",
                   f"Rank #{int(row['production_rank'])}" if row["in_production"] else None)
        col.metric("Objective EV", f"{row['objective_ev']:.2f}" if row["in_objective"] else "N/A",
                   f"Rank #{int(row['objective_rank'])}" if row["in_objective"] else None)
        if w is not None and pd.notna(w.get("wheelo_ev")):
            col.metric("Wheelo EV", f"{w['wheelo_ev']:.2f}", f"Rank #{int(w['wheelo_rank'])}" if pd.notna(w.get("wheelo_rank")) else None)
        else:
            col.metric("Wheelo EV", "N/A")
        player_opps = opportunities[opportunities["player_name"] == name]
        for mtype, label in [("WINNER", "Winner"), ("TOP_N", "Top 5"), ("TO_POLL_A_VOTE", "To Poll a Vote")]:
            sel = player_opps[player_opps["market_type"] == mtype]
            if mtype == "TOP_N":
                sel = sel[sel["n"] == 5]
            if not sel.empty:
                col.caption(f"{label}: Production {bo.format_pct(sel.iloc[0]['production_probability'])} / Objective {bo.format_pct(sel.iloc[0]['objective_probability'])}")
        return row

    row_a = _player_card(p6a, player_a)
    row_b = _player_card(p6b, player_b)

    st.subheader("Model disagreement")
    if row_a["in_production"] and row_a["in_objective"]:
        st.write(f"**{player_a}** internal model gap: {row_a['absolute_difference']:.2f} EV")
    if row_b["in_production"] and row_b["in_objective"]:
        st.write(f"**{player_b}** internal model gap: {row_b['absolute_difference']:.2f} EV")

    st.subheader("Available H2H / player markets")
    h2h = opportunities[
        (opportunities["market_type"] == "PLAYER_H2H")
        & opportunities["market_name"].str.contains(player_a, na=False)
        & opportunities["market_name"].str.contains(player_b, na=False)
    ]
    both_players = opportunities[opportunities["player_name"].isin([player_a, player_b])]
    market_table = pd.concat([h2h, both_players]).drop_duplicates(subset=["selection_id"])
    if market_table.empty:
        st.caption("No bookmaker markets currently available directly comparing or covering both players.")
    else:
        st.dataframe(pd.DataFrame({
            "Player": market_table["player_name"], "Bet": market_table["bet"],
            "Bookmaker": market_table["source"], "Odds": market_table["odds"].apply(bo.format_odds),
            "Confidence": market_table["confidence_badge"],
        }), use_container_width=True, hide_index=True)

st.divider()

# ==========================================================================
# 7. PLAYER MARKET EXPLORER
# ==========================================================================
st.header("7. Player Market Explorer")
player_names = sorted(opportunities["player_name"].dropna().unique())
if not player_names:
    st.caption("No player-level markets currently available.")
else:
    search7 = st.text_input("Search player", key="pme_search")
    filtered_names = [n for n in player_names if search7.lower() in n.lower()] if search7 else player_names
    if not filtered_names:
        st.caption("No player matches that search.")
    else:
        player7 = st.selectbox("Player", filtered_names, key="pme_player")
        cmp_row = cmp[cmp["player_name"] == player7]
        w_row = external_overview[external_overview["player_name"] == player7]
        m1, m2, m3 = st.columns(3)
        if not cmp_row.empty:
            r = cmp_row.iloc[0]
            m1.metric("Production EV / rank", f"{r['production_ev']:.2f} / #{int(r['production_rank'])}" if r["in_production"] else "N/A")
            m2.metric("Objective EV / rank", f"{r['objective_ev']:.2f} / #{int(r['objective_rank'])}" if r["in_objective"] else "N/A")
        if not w_row.empty and pd.notna(w_row.iloc[0].get("wheelo_ev")):
            m3.metric("Wheelo EV / rank", f"{w_row.iloc[0]['wheelo_ev']:.2f} / #{int(w_row.iloc[0]['wheelo_rank'])}")
        else:
            m3.metric("Wheelo EV / rank", "N/A")

        player_markets = opportunities[opportunities["player_name"] == player7]
        piv = bo.with_bookmaker_odds(player_markets, ["market_type", "line", "side", "n", "position", "threshold"])
        st.dataframe(pd.DataFrame({
            "Market": piv["market_label"], "Neds odds": piv["neds_odds"].apply(bo.format_odds),
            "PointsBet odds": piv["pointsbet_odds"].apply(bo.format_odds), "Best odds": piv["best_odds"].apply(bo.format_odds),
            "Production %": piv["production_probability"].apply(bo.format_pct),
            "Objective %": piv["objective_probability"].apply(bo.format_pct),
            "Wheelo support": piv["wheelo_support"], "Confidence": piv["confidence_badge"],
        }), use_container_width=True, hide_index=True)

st.divider()

# ==========================================================================
# 8. SUGGESTED COMBINATIONS
# ==========================================================================
st.header("8. Suggested Combinations")
complete_combos = combinations[
    combinations["legs"].notna() & (combinations["legs"].str.strip() != "")
    & combinations["production_joint_probability"].notna() & combinations["objective_joint_probability"].notna()
] if not combinations.empty else combinations

if complete_combos.empty:
    st.caption("No complete combination candidates currently available (requires priced single legs).")
else:
    st.caption(
        "Joint probabilities computed from real Production/Objective simulation draws (never a "
        "product of marginals). Shows 'verify price with bookmaker' instead of a fabricated "
        "combined price whenever a real combined price isn't retrievable."
    )
    for n_legs, label in [(2, "2-leg -- Lower variance"), (3, "3-leg -- Balanced"),
                           (4, "4-leg -- Higher return"), (5, "5-leg -- Speculative")]:
        subset = complete_combos[complete_combos["n_legs"] == n_legs]
        if subset.empty:
            continue
        st.subheader(label)
        for _, row in subset.iterrows():
            legs = [l.strip() for l in str(row["legs"]).split(" + ") if l.strip()]
            wheelo_per_leg = [w.strip() for w in str(row.get("wheelo_support_per_leg", "")).split(";")]
            with st.container(border=True):
                for i, leg in enumerate(legs):
                    wl = wheelo_per_leg[i] if i < len(wheelo_per_leg) else "n/a"
                    st.markdown(f"**Leg {i + 1}:** {leg}  — *Wheelo: {wl}*")
                st.markdown(f"**Bookmaker:** {row['bookmaker']}")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Production joint %", bo.format_pct(row["production_joint_probability"]))
                cc2.metric("Objective joint %", bo.format_pct(row["objective_joint_probability"]))
                cc3.metric("Joint model gap", bo.format_pp(row["joint_model_gap"] * 100 if pd.notna(row["joint_model_gap"]) else None))
                st.caption(f"Risk tier: {row['risk_tier']} -- {row['status']}")

st.divider()

# ==========================================================================
# 9. ADVANCED / ALL MARKETS
# ==========================================================================
st.header("9. Advanced / All Markets")
st.caption("The exhaustive raw market inventory, including UNMODELLED and flagged rows, lives only here.")

f1, f2, f3, f4, f5, f6 = st.columns(6)
bookmaker_filter = f1.selectbox("Bookmaker", ["All", "Neds", "PointsBet"])
confidence_options = sorted(opportunities["confidence"].dropna().unique().tolist())
confidence_filter = f2.multiselect("Confidence", confidence_options, default=confidence_options)
market_options = sorted(opportunities["market_type"].dropna().unique().tolist())
market_filter = f3.multiselect("Market type", market_options, default=market_options)
search9 = f4.text_input("Player / team search", key="adv_search")
min_odds9 = f5.number_input("Minimum odds", min_value=1.0, value=1.0, step=0.5)
min_edge9 = f6.number_input("Minimum internal edge (pp)", value=-100.0, step=1.0)

adv = opportunities[
    opportunities["confidence"].isin(confidence_filter)
    & opportunities["market_type"].isin(market_filter)
    & (opportunities["odds"] >= min_odds9)
]
if bookmaker_filter != "All":
    adv = adv[adv["source"].apply(lambda s: "PointsBet" if str(s).startswith("pointsbet") else "Neds") == bookmaker_filter]
if search9:
    adv = adv[
        adv["player_name"].astype(str).str.contains(search9, case=False, na=False)
        | adv["team_id"].astype(str).str.contains(search9, case=False, na=False)
    ]

st.caption(f"{len(adv)} of {len(opportunities)} rows match the current filters.")
simple_view = pd.DataFrame({
    "Bet": adv["bet"], "Bookmaker": adv["source"].apply(lambda s: "PointsBet" if str(s).startswith("pointsbet") else "Neds"),
    "Odds": adv["odds"].apply(bo.format_odds), "Production %": adv["production_probability"].apply(bo.format_pct),
    "Objective %": adv["objective_probability"].apply(bo.format_pct), "Confidence": adv["confidence_badge"],
    "Data quality": adv["data_quality_flags"].fillna("none"),
})
st.dataframe(simple_view, use_container_width=True, hide_index=True, height=500)

with st.expander("Technical fields (internal market_type, raw ids, full precision)"):
    tech_cols = ["selection_id", "source", "market_type", "market_name", "selection", "player_id",
                 "player_name", "team_id", "line", "side", "n", "position", "threshold", "odds",
                 "implied_probability", "production_probability", "objective_probability",
                 "production_edge_pp", "objective_edge_pp", "production_ev", "objective_ev",
                 "internal_gap_pp", "wheelo_ev", "wheelo_rank", "confidence", "data_quality_flags"]
    st.dataframe(adv[[c for c in tech_cols if c in adv.columns]], use_container_width=True, hide_index=True)

st.divider()
if not price_comparison.empty:
    with st.expander("Best Bookmaker Price (Neds vs PointsBet, equivalent markets only)"):
        pc = price_comparison.copy()
        pc["Bet"] = pc.apply(bo.bet_description, axis=1) if "market_type" in pc.columns else pc.get("selection")
        st.dataframe(pc, use_container_width=True, hide_index=True)
