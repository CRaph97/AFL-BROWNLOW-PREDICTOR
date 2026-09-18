"""
Finishing Order -- cross-model placement analysis for betting/research.

Read-only over the already-generated Production/Objective Monte Carlo draw
arrays (data/processed/mc_totals_2026.npy, mc_totals_objective_2026.npy) via
dashboard/finishing_order.py + src/models/order_scenarios.py. No model is
retrained, reweighted, or re-simulated here -- every probability on this page
is a real count over existing simulation draws. Wheelo shows EV + season rank
only (it has no persisted simulation draws), never a fabricated Top-N or
exact-order probability.
"""
import pandas as pd
import streamlit as st

from dashboard import data as d
from dashboard import betting_opportunities as bo
from dashboard import finishing_order as fo

st.set_page_config(page_title="Finishing Order", layout="wide")
d.highlight_objective_stats_nav()
st.title("Finishing Order")
st.caption(
    "Cross-model finishing-position analysis, computed from real Production/Objective season "
    "simulations. Wheelo has no season simulation of its own -- shown as EV + rank only."
)

N_LABELS = {1: "Winner"} | {n: f"Top {n}" for n in range(2, fo.MAX_N + 1)}

n = st.slider("Finishing position threshold", 1, fo.MAX_N, 5, format="%d")
st.markdown(f"### {N_LABELS[n]}")

table = fo.topn_table(n)

# --------------------------------------------------------------------------
# Quick Model Top-N Summary -- side-by-side player lists, disagreement
# visible at a glance (bold = only in that model's list, not the other's).
# --------------------------------------------------------------------------
prod_top = set(table.sort_values("production_topn", ascending=False).head(n)["player_name"])
obj_top = set(table.sort_values("objective_topn", ascending=False).head(n)["player_name"])
wheelo_top = set(
    table.dropna(subset=["wheelo_rank"]).sort_values("wheelo_rank").head(n)["player_name"]
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**Production {N_LABELS[n]}**")
    for name in table.sort_values("production_topn", ascending=False).head(n)["player_name"]:
        mark = "**" if name not in obj_top else ""
        st.markdown(f"- {mark}{name}{mark}")
with c2:
    st.markdown(f"**Objective {N_LABELS[n]}**")
    for name in table.sort_values("objective_topn", ascending=False).head(n)["player_name"]:
        mark = "**" if name not in prod_top else ""
        st.markdown(f"- {mark}{name}{mark}")
with c3:
    st.markdown(f"**Wheelo {N_LABELS[n]} by EV**")
    if table["wheelo_ev"].notna().any():
        for name in table.dropna(subset=["wheelo_ev"]).sort_values("wheelo_ev", ascending=False).head(n)["player_name"]:
            mark = "**" if name not in (prod_top | obj_top) else ""
            st.markdown(f"- {mark}{name}{mark}")
    else:
        st.caption("No Wheelo data available.")
st.caption("**Bold** = this player appears in only one model's list above.")

st.divider()

# --------------------------------------------------------------------------
# Tabs: Combined / Production / Objective / Wheelo
# --------------------------------------------------------------------------
tab_combined, tab_prod, tab_obj, tab_wheelo = st.tabs(["Combined", "Production", "Objective", "Wheelo"])

DISPLAY_ROWS = max(20, n + 10)  # never a giant default table

with tab_combined:
    show = table.head(DISPLAY_ROWS).copy()
    st.dataframe(
        pd.DataFrame({
            "Player": show["player_name"],
            "Team": show["team_id"].apply(d._display_team),
            f"Production {N_LABELS[n]} %": show["production_topn"].apply(bo.format_pct),
            f"Objective {N_LABELS[n]} %": show["objective_topn"].apply(bo.format_pct),
            "Production rank": show["production_rank"],
            "Objective rank": show["objective_rank"],
            "Wheelo EV": show["wheelo_ev"].round(2),
            "Wheelo rank": show["wheelo_rank"],
            "Model gap (pp)": show["model_gap_pp"].round(1),
            "Agreement": show["agreement_label"],
        }),
        use_container_width=True, hide_index=True, height=500,
    )
    st.caption(f"Showing top {min(DISPLAY_ROWS, len(table))} of {len(table)} players by combined model evidence.")

with tab_prod:
    show = table.sort_values("production_topn", ascending=False).head(DISPLAY_ROWS)
    st.dataframe(
        pd.DataFrame({
            "Player": show["player_name"], "Team": show["team_id"].apply(d._display_team),
            f"Production {N_LABELS[n]} %": show["production_topn"].apply(bo.format_pct),
            "Production rank": show["production_rank"],
        }),
        use_container_width=True, hide_index=True, height=500,
    )

with tab_obj:
    show = table.sort_values("objective_topn", ascending=False).head(DISPLAY_ROWS)
    st.dataframe(
        pd.DataFrame({
            "Player": show["player_name"], "Team": show["team_id"].apply(d._display_team),
            f"Objective {N_LABELS[n]} %": show["objective_topn"].apply(bo.format_pct),
            "Objective rank": show["objective_rank"],
        }),
        use_container_width=True, hide_index=True, height=500,
    )

with tab_wheelo:
    st.caption("Wheelo publishes a single season point-prediction, not a season simulation -- EV and rank only, never a Top-N probability.")
    show = table.dropna(subset=["wheelo_ev"]).sort_values("wheelo_rank").head(DISPLAY_ROWS)
    if show.empty:
        st.caption("No Wheelo data resolved for this player set.")
    else:
        st.dataframe(
            pd.DataFrame({
                "Player": show["player_name"], "Team": show["team_id"].apply(d._display_team),
                "Wheelo EV": show["wheelo_ev"].round(2), "Wheelo season rank": show["wheelo_rank"],
            }),
            use_container_width=True, hide_index=True, height=500,
        )

st.divider()

# --------------------------------------------------------------------------
# Exact Order Builder
# --------------------------------------------------------------------------
st.header("Exact Order Builder")
st.caption(
    "Pick an exact finishing order for positions 1 to k (k capped at 5). Probabilities are real "
    "joint outcomes counted directly from existing simulation draws -- never a product of "
    "independent per-player probabilities."
)

k = st.slider("Positions to specify (k)", 1, 5, 3, key="eob_k")
name_options = sorted(table["player_name"].dropna().unique())
name_to_id = dict(zip(table["player_name"], table["player_id"]))
# Default to the current top contenders (by combined model evidence), not an
# alphabetical first-N -- a random/fringe default order is trivially "very
# rare" and tells a first-time user nothing useful about the feature.
default_order = table.sort_values("combined_evidence", ascending=False)["player_name"].head(5).tolist()

cols = st.columns(k)
chosen_names = []
for i, col in enumerate(cols):
    with col:
        default_name = default_order[i] if i < len(default_order) else name_options[0]
        default_idx = name_options.index(default_name)
        choice = st.selectbox(f"Position {i + 1}", name_options, index=default_idx, key=f"eob_pos_{i}")
        chosen_names.append(choice)

if len(set(chosen_names)) != len(chosen_names):
    st.warning("Each position must have a different player -- pick k distinct players.")
else:
    chosen_ids = [name_to_id[n] for n in chosen_names]
    result = fo.exact_order_probability(chosen_ids)
    order_str = " → ".join(f"{i + 1}. {n}" for i, n in enumerate(chosen_names))
    st.markdown(f"**Order:** {order_str}")

    ec1, ec2 = st.columns(2)
    for col, label, key in [(ec1, "Production", "production"), (ec2, "Objective", "objective")]:
        with col:
            r = result[key]
            st.markdown(f"**{label}**")
            if r["exact_order_supporting_draws"] < fo.MIN_SUPPORTING_DRAWS:
                st.metric("Exact order probability", "Very rare in current simulations")
                st.caption(f"Only {r['exact_order_supporting_draws']} of {r['n_sims']:,} simulations matched this exact order.")
            else:
                st.metric("Exact order probability", bo.format_pct(r["exact_order_prob"]))
                st.caption(f"{r['exact_order_supporting_draws']:,} of {r['n_sims']:,} simulations.")
            st.metric(f"All {k} players finish in Top {k} (any order)", bo.format_pct(r["all_in_topk_prob"]))
