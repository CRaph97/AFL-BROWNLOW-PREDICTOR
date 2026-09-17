import plotly.graph_objects as go
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Uncertainty", layout="wide")
st.title("Uncertainty")
st.caption(
    "Four distinct kinds of uncertainty, kept separate rather than compressed into one "
    "confidence score:\n"
    "- **Simulation uncertainty** (aleatoric): the 80%/95% Monte Carlo interval width for a "
    "fixed model -- reflects genuine voting randomness across simulated seasons.\n"
    "- **Model disagreement**: how much the historical / recent-era / stats-assisted scenarios "
    "differ for a player, holding the data fixed.\n"
    "- **Structural-break sensitivity**: how much a player's projection depends specifically on "
    "the untested 2026 umpire-statistics assumption.\n"
    "- **Data uncertainty**: not scored numerically here -- see docs/2026_DATA_VALIDATION.md and "
    "docs/2026_FINAL_AUDIT.md for known data-coverage gaps (e.g. unresolved player identities, "
    "footywire join gaps)."
)

lb = d.load_leaderboard()
n = st.slider("Number of top players to show", 5, 30, 15)
top = lb.sort_values("FINAL_ENSEMBLE", ascending=False).head(n).sort_values("FINAL_ENSEMBLE")

st.subheader("Simulation Uncertainty: 80% / 95% Intervals")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=top["sim_p2_5"], y=top["player_name"], mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"),
    showlegend=False, hoverinfo="skip",
))
for _, r in top.iterrows():
    fig.add_trace(go.Scatter(x=[r["sim_p2_5"], r["sim_p97_5"]], y=[r["player_name"]] * 2,
                              mode="lines", line=dict(color="#4a90d9", width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=[r["sim_p10"], r["sim_p90"]], y=[r["player_name"]] * 2,
                              mode="lines", line=dict(color="#e8a33d", width=7), showlegend=False))
    fig.add_trace(go.Scatter(x=[r["FINAL_ENSEMBLE"]], y=[r["player_name"]], mode="markers",
                              marker=dict(color="white", size=9, symbol="diamond"), showlegend=False))
fig.update_layout(
    template="plotly_dark", height=max(400, 28 * n), margin=dict(t=10, b=10),
    xaxis_title="Expected Votes  (thin blue = 95% CI, thick orange = 80% CI, white diamond = point estimate)",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Model Disagreement vs. Structural-Break Sensitivity")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=top["model_disagreement_range"], y=top["structural_break_sensitivity"],
    mode="markers+text", text=top["player_name"], textposition="top center",
    marker=dict(size=10, color=top["FINAL_ENSEMBLE"], colorscale="Blues", showscale=True,
                colorbar=dict(title="Final EV")),
))
fig2.update_layout(
    template="plotly_dark", height=520, margin=dict(t=10, b=10),
    xaxis_title="Model Disagreement (scenario spread)", yaxis_title="Structural-Break Sensitivity",
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "Players in the top-right are the ones whose ranking should be trusted least: both the "
    "underlying scenarios disagree AND that disagreement is specifically tied to the untested "
    "2026 rule change."
)
