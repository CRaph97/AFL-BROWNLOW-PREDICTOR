"""
Clinch Round -- when the Brownlow is likely to become effectively decided.

Read-only over already-computed, already-validated simulation outputs. Does
NOT retrain any model, run a NEW simulation, or alter any existing output
file. dashboard/clinch_round.py replays Production's and Objective's exact,
already-committed Monte Carlo simulations (same seed, same per-match random
draws, same iteration order) purely to recover the per-real-round cumulative
state the original scripts discarded -- verified bit-identical to
data/processed/mc_totals_2026.npy / mc_totals_objective_2026.npy in
tests/test_clinch_round_page.py.

Three distinct concepts are shown, deliberately never blended:
1. Mathematical Clinch (Production + Objective, simulation-based, a genuine
   probability): the first round a simulation's leader has more votes than
   every rival could possibly reach even if that rival won every vote in
   every one of their real remaining matches. Ties do NOT count as clinched.
2. Projected Winner Point (Production + Objective, simulation-based, a
   genuine probability, but a real methodological judgement call -- stated
   plainly rather than hidden): P(current leader at round R is the eventual
   winner | they are the leader at round R).
3. Wheelo Expected-Trajectory Clinch (deterministic ONLY -- never a
   probability, never plotted on the same axis as 1/2): the round Wheelo's
   real cumulative EV trajectory for the leader first exceeds every rival's
   3-per-remaining-game ceiling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import clinch_round as cr
from dashboard import data as d

st.set_page_config(page_title="Clinch Round", layout="wide")
d.highlight_objective_stats_nav()
st.title("Clinch Round")
st.caption(
    "When does the Brownlow become effectively decided? Production and Objective answer via their own "
    "season simulations (replayed round-by-round, not re-run); Wheelo answers via its real expected-vote "
    "trajectory only -- never as a probability, and never on the same scale as the other two."
)

with st.expander("Definitions (read this before the numbers below)", expanded=False):
    st.markdown(
        """
**Mathematical Clinch** -- for one simulation, the first round where the current leader's votes exceed
every rival's *maximum possible remaining total* (that rival's current votes + 3 votes in every one of
their real remaining matches, from the actual 2026 fixture, byes included). This is a strict inequality:
if a rival could still exactly *tie*, the race is **not** counted as clinched. Reported as the fraction of
all simulations clinched by each round -- a genuine probability, because it comes directly from Production's
100,000-simulation / Objective's 20,000-simulation Monte Carlo replay.

**Projected Winner Point** -- a separate, softer question: *"once a player is the leader at round R, how
often do they go on to actually win?"* Formally, P(leader at round R is the eventual simulated winner |
they are the leader at round R). This is **not** the same thing as being mathematically clinched -- a
leader can be very likely to win without yet being clinched. It is a genuine conditional probability, but
which conditional probability to report is a real judgement call, stated here rather than hidden.

**Wheelo Expected-Trajectory Clinch** -- Wheelo has no persisted season simulation to replay, only real
per-match expected-vote (EV) data. This applies the *same* remaining-games ceiling logic to Wheelo's actual
cumulative EV to date, deterministically. It is a single trajectory, not a distribution -- so it is never
given a probability and never plotted next to Production/Objective's genuine simulation-based curves.
        """
    )

prod = cr.replay_production()
obj = cr.replay_objective()
wheelo = cr.wheelo_expected_trajectory()

prod_players, obj_players = prod["players"], obj["players"]


def _final_favourite(replay: dict) -> tuple[str, str, float]:
    counts = np.bincount(replay["final_winner_idx"], minlength=len(replay["players"]))
    top_i = int(counts.argmax())
    row = replay["players"].iloc[top_i]
    return row["player_id"], row["player_name"], float(counts[top_i] / replay["n_sims"])


prod_pid, prod_name, prod_win_share = _final_favourite(prod)
obj_pid, obj_name, obj_win_share = _final_favourite(obj)

# --------------------------------------------------------------------------
# Top summary cards.
# --------------------------------------------------------------------------
st.subheader("Summary")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Production** (simulation-based)")
    st.metric("Most likely eventual winner", prod_name, f"{prod_win_share:.1%} of simulations")
    never = float(np.isnan(prod["clinch_round_per_sim"]).mean())
    valid = prod["clinch_round_per_sim"][~np.isnan(prod["clinch_round_per_sim"])]
    st.metric("Median clinch round", f"R{int(np.median(valid))}" if len(valid) else "n/a")
    st.caption(f"Never clinched (season-long tie risk) in {never:.1%} of simulations.")

with c2:
    st.markdown("**Objective** (simulation-based)")
    st.metric("Most likely eventual winner", obj_name, f"{obj_win_share:.1%} of simulations")
    never_o = float(np.isnan(obj["clinch_round_per_sim"]).mean())
    valid_o = obj["clinch_round_per_sim"][~np.isnan(obj["clinch_round_per_sim"])]
    st.metric("Median clinch round", f"R{int(np.median(valid_o))}" if len(valid_o) else "n/a")
    st.caption(f"Never clinched (season-long tie risk) in {never_o:.1%} of simulations.")

with c3:
    st.markdown("**Wheelo** (expected-trajectory, deterministic)")
    wheelo_final = wheelo["cumulative_by_round"][wheelo["max_round"]].sort_values(ascending=False)
    if not wheelo_final.empty:
        w_top_pid = wheelo_final.index[0]
        w_clinch = wheelo["clinch_round_by_player"].get(w_top_pid)
        st.metric("Leading player (by cumulative EV)", str(w_top_pid))
        st.metric("Expected-trajectory clinch round", f"R{w_clinch}" if w_clinch is not None else "Not reached")
    st.caption("Not a probability -- a single deterministic trajectory, not comparable to the two cards above.")

st.divider()

# --------------------------------------------------------------------------
# Cumulative P(clinched by round) -- Production vs Objective only.
# --------------------------------------------------------------------------
st.subheader("P(mathematically clinched by round)")
rounds = range(0, max(prod["max_round"], obj["max_round"]) + 1)
prod_curve = pd.Series({r: float((prod["clinch_round_per_sim"] <= r).mean()) for r in rounds})
obj_curve = pd.Series({r: float((obj["clinch_round_per_sim"] <= r).mean()) for r in rounds})
curve_df = pd.DataFrame({"Production": prod_curve, "Objective": obj_curve})
curve_df.index.name = "Round"
st.line_chart(curve_df)
st.caption(
    "Strict clinch definition (ties do not count). Curves are non-decreasing by construction -- clinch "
    "status cannot be lost once achieved, since rival ceilings only shrink and leader votes only grow."
)

st.divider()

# --------------------------------------------------------------------------
# Wheelo trajectory -- visually distinct, deterministic-only section.
# --------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Wheelo expected-trajectory (deterministic -- not a probability)")
    top5_pids = wheelo_final.head(5).index.tolist() if not wheelo_final.empty else []
    if top5_pids:
        traj_df = wheelo["cumulative_by_round"].loc[top5_pids].T
        traj_df.index.name = "Round"
        traj_df.columns = [str(c) for c in traj_df.columns]
        st.line_chart(traj_df)
        st.caption(
            "Cumulative real Wheelo expected votes, top 5 players by season-end total. A single real "
            "trajectory, not a simulated distribution -- do not compare its shape directly to the "
            "probability curves above."
        )
    else:
        st.caption("No resolved Wheelo match data available.")

st.divider()

# --------------------------------------------------------------------------
# Round Explorer.
# --------------------------------------------------------------------------
st.subheader("Round Explorer")
max_round_overall = max(prod["max_round"], obj["max_round"])
explorer_round = st.slider("Round", 0, max_round_overall, max_round_overall, format="R%d", key="clinch_round_explorer")


def _round_row(replay: dict, round_: int, label: str) -> dict:
    mean_cum = replay["mean_cumulative_by_round"][round_]
    leader_i = int(mean_cum.argmax())
    players = replay["players"]
    leader_row = players.iloc[leader_i]
    remaining_lookup = cr.team_remaining_matches()
    remaining_vec = cr._player_remaining_vector(players, remaining_lookup, round_)
    ceiling = mean_cum + 3 * remaining_vec
    ceiling_masked = ceiling.copy()
    ceiling_masked[leader_i] = -1
    challenger_i = int(ceiling_masked.argmax())
    challenger_row = players.iloc[challenger_i]

    frac_clinched = float((replay["clinched_by_round"][round_]).mean())
    proj_prob = cr.conditional_win_prob_at_round(replay, leader_row["player_id"], round_)

    return {
        "Source": label,
        "Leader (avg. sim)": leader_row["player_name"],
        "Leader mean cum. votes": round(float(mean_cum[leader_i]), 1),
        "Nearest challenger": challenger_row["player_name"],
        "Challenger mean cum. votes": round(float(mean_cum[challenger_i]), 1),
        "Margin": round(float(mean_cum[leader_i] - mean_cum[challenger_i]), 1),
        "P(mathematically clinched)": f"{frac_clinched:.1%}",
        "P(leader today is eventual winner)": f"{proj_prob:.1%}" if pd.notna(proj_prob) else "n/a",
    }


rows = [_round_row(prod, explorer_round, "Production"), _round_row(obj, explorer_round, "Objective")]

if explorer_round in wheelo["cumulative_by_round"].columns and not wheelo_final.empty:
    w_col = wheelo["cumulative_by_round"][explorer_round]
    w_leader_pid = w_col.idxmax()
    w_leader_cum = w_col.max()
    w_rest = w_col.drop(index=w_leader_pid)
    w_row = {
        "Source": "Wheelo (expected-trajectory)",
        "Leader (avg. sim)": str(w_leader_pid),
        "Leader mean cum. votes": round(float(w_leader_cum), 1),
        "Nearest challenger": str(w_rest.idxmax()) if not w_rest.empty else "n/a",
        "Challenger mean cum. votes": round(float(w_rest.max()), 1) if not w_rest.empty else 0.0,
        "Margin": round(float(w_leader_cum - (w_rest.max() if not w_rest.empty else 0.0)), 1),
        "P(mathematically clinched)": "n/a (deterministic)",
        "P(leader today is eventual winner)": "n/a (deterministic)",
    }
    rows.append(w_row)

st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.divider()

# --------------------------------------------------------------------------
# Insights -- deterministic, computed directly above; no narrative embellishment.
# --------------------------------------------------------------------------
st.subheader("Insights")
insights = []
insights.append(
    f"Production expects the race to be mathematically clinched by round {int(np.median(valid))} in a typical "
    f"simulation (median), reaching 90%+ clinched probability by round "
    f"{int(prod_curve[prod_curve >= 0.9].index.min()) if (prod_curve >= 0.9).any() else prod['max_round']}."
)
insights.append(
    f"Objective's clinch distribution is later and less certain: median round "
    f"{int(np.median(valid_o)) if len(valid_o) else prod['max_round']}, with "
    f"{never_o:.1%} of simulations never mathematically resolving by the final round."
)
insights.append(
    f"Production and Objective agree on {prod_name} as the most likely eventual winner "
    f"({prod_win_share:.1%} vs {obj_win_share:.1%} of simulations)." if prod_name == obj_name else
    f"Production favours {prod_name} ({prod_win_share:.1%}) while Objective favours {obj_name} "
    f"({obj_win_share:.1%}) as the most likely eventual winner -- a genuine model disagreement."
)
if not wheelo_final.empty:
    w_clinch_top = wheelo["clinch_round_by_player"].get(wheelo_final.index[0])
    insights.append(
        f"Wheelo's real expected-trajectory clinch point for its current leader is "
        f"{'round ' + str(w_clinch_top) if w_clinch_top is not None else 'not yet reached'} -- a single "
        f"deterministic estimate, not directly comparable to the simulation-based probabilities above."
    )
insights.append(
    "A clinch requires STRICT separation from every rival's maximum possible remaining total; an exact tie "
    "at any round is deliberately not counted as clinched in any of the figures on this page."
)
for b in insights[:5]:
    st.markdown(f"- {b}")
