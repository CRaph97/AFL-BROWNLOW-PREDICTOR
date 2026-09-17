import pandas as pd
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Round-by-Round Leaderboard", layout="wide")
st.title("Round-by-Round Leaderboard")
st.caption(
    "Cumulative expected Brownlow votes using only matches with official round <= the "
    "selected round. Built entirely from the frozen reports/2026_match_probabilities.csv "
    "(expected_votes) and reports/2026_predicted_votes.csv (deterministic 3/2/1 picks) -- "
    "no probability is recomputed or renormalised here."
)


@st.cache_data
def _round_labels(rounds: list[int]) -> dict[int, str]:
    return {r: ("Opening Round" if r == 0 else f"Round {r}") for r in rounds}


@st.cache_data
def cumulative_table(upto_round: int) -> pd.DataFrame:
    mp = d.load_match_probabilities()
    pv = d.load_predicted_votes()

    sub_mp = mp[mp["round"] <= upto_round]
    cum = (
        sub_mp.groupby(["player_id", "player_name", "team_id"])["expected_votes"]
        .sum()
        .reset_index()
        .rename(columns={"expected_votes": "cumulative_ev"})
    )

    sub_pv = pv[pv["round"] <= upto_round]
    vote_counts = (
        sub_pv.pivot_table(index="player_id", columns="predicted_votes", values="match_id", aggfunc="count")
        .reindex(columns=[3, 2, 1], fill_value=0)
        .fillna(0)
        .astype(int)
        .rename(columns={3: "games_3", 2: "games_2", 1: "games_1"})
    )

    out = cum.merge(vote_counts, on="player_id", how="left")
    for c in ("games_3", "games_2", "games_1"):
        out[c] = out[c].fillna(0).astype(int)
    out = out.sort_values("cumulative_ev", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


mp_all = d.load_match_probabilities()
rounds = sorted(mp_all["round"].unique().tolist())
labels = _round_labels(rounds)

selected_round = st.selectbox(
    "Select round", rounds, index=len(rounds) - 1, format_func=lambda r: labels[r]
)

cur = cumulative_table(selected_round)

idx = rounds.index(selected_round)
prev = cumulative_table(rounds[idx - 1]) if idx > 0 else None

if prev is not None:
    merged = cur.merge(
        prev[["player_id", "rank", "cumulative_ev"]],
        on="player_id",
        how="left",
        suffixes=("", "_prev"),
    )
    merged["rank_change"] = merged["rank_prev"] - merged["rank"]  # positive = moved up
    merged["ev_change"] = merged["cumulative_ev"] - merged["cumulative_ev_prev"].fillna(0)
    is_new = merged["rank_prev"].isna()
    merged["rank_change_display"] = merged.apply(
        lambda r: "NEW" if pd.isna(r["rank_prev"]) else f"{int(r['rank_change']):+d}", axis=1
    )
else:
    merged = cur.copy()
    merged["ev_change"] = merged["cumulative_ev"]
    merged["rank_change_display"] = "--"

merged["team_display"] = merged["team_id"].map(lambda t: t.replace("_", " ").title())

n_show = st.slider("Players to show", 10, 100, 30, step=10)
display = merged.head(n_show)[
    [
        "rank",
        "player_name",
        "team_display",
        "cumulative_ev",
        "games_3",
        "games_2",
        "games_1",
        "rank_change_display",
        "ev_change",
    ]
].rename(
    columns={
        "rank": "Rank",
        "player_name": "Player",
        "team_display": "Team",
        "cumulative_ev": "Cumulative Expected Votes",
        "games_3": "3-Vote Games To Date",
        "games_2": "2-Vote Games To Date",
        "games_1": "1-Vote Games To Date",
        "rank_change_display": "Rank Change",
        "ev_change": f"EV Change vs {'previous round' if idx > 0 else 'n/a'}",
    }
)

st.caption(f"{labels[selected_round]} -- {len(merged)} player(s) with a nonzero cumulative expected vote.")
st.dataframe(
    display.style.format(
        {
            "Cumulative Expected Votes": "{:.2f}",
            display.columns[-1]: "{:+.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
