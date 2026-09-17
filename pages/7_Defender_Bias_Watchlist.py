import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Defender Bias Watchlist", layout="wide")
st.title("Defender Bias Watchlist")

st.markdown(
    "**Audit / watchlist only -- not a correction layer.** These 2026 games are parsed "
    "directly from `docs/2026_FINAL_AUDIT.md` (section 4), which found that of 1,768 "
    "defender-role-tagged player-match rows in 2026, 90 fall in the top 5% league-wide by a "
    "defensive-output proxy (disposals + 2x contested marks + 0.5x one-percenters); 58 of "
    "those 90 receive a model expected-votes estimate below 0.5. This is consistent with the "
    "Phase 4 finding that key defenders are correctly identified as 3-vote winners only 12.5% "
    "of the time vs. 62% for midfielders. **No prediction below is manually adjusted.**"
)

wl = d.load_defender_watchlist()

st.dataframe(
    wl.rename(columns={
        "round": "Round", "player_name": "Player", "team_id": "Team",
        "opponent_id": "Opponent", "disposals": "Disposals",
        "contested_marks": "Contested Marks", "one_percenters": "One-Percenters",
        "expected_votes": "Model EV", "reason_flagged": "Reason Flagged",
    }).assign(
        Team=lambda x: x["Team"].str.replace("_", " ").str.title(),
        Opponent=lambda x: x["Opponent"].str.replace("_", " ").str.title(),
    ).style.format({"Model EV": "{:.2f}"}),
    use_container_width=True, hide_index=True, height=560,
)

st.caption(f"{len(wl)} game(s) flagged. Full methodology: docs/2026_FINAL_AUDIT.md, section 4.")
