"""
2027 Model Lab -- R&D view of the walk-forward candidate models
(Structural A, Performance ML B, Stats-only C, ensemble research), ablations,
calibration, disagreement, bias, Error Lab and the experiment registry.
Presentation only over data/experiments/ (built by src/validation/*).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import model_lab as ml

st.set_page_config(page_title="2027 Model Lab", layout="wide")
st.title("2027 Model Lab")
st.caption("Walk-forward R&D for the 2027 Brownlow models. Every number is out-of-sample (train < test season). No 2027 forecast exists yet. "
           "Methodology: docs/2027_MODEL_R&D_PLAN.md; results: docs/2027_MODEL_R&D_RESULTS.md.")
if not ml.available():
    st.info("No analysis outputs yet. Run `python -m src.validation.run_experiments --suite candidates` then `python -m src.validation.analyze`.")
    st.stop()

W = "stretch"
pooled = ml.csv("comparison_pooled"); by_season = ml.csv("comparison_by_season"); reg = ml.registry(); summ = ml.summary()
tabs = st.tabs(["Overview", "Model Comparison", "Walk-Forward", "Ablations", "Error Lab", "Calibration", "Disagreement", "Bias", "Experiments"])

with tabs[0]:
    st.subheader("Champion / challengers")
    champ = ml.csv("champion")
    if not champ.empty:
        for _, r in champ.iterrows():
            st.markdown(f"**{r['role']}**: {r['model']} -- {r['reason']}")
    allp = pooled[pooled["window_seasons"].str.startswith("all")].set_index("model")
    cols = st.columns(min(4, len(allp)))
    for col, (m, r) in zip(cols, allp.iterrows()):
        with col:
            st.markdown(f"#### {m}")
            st.metric("3-vote accuracy (pooled)", f"{r['correct_3'] * 100:.1f}%")
            st.metric("P3 log loss", f"{r['log_loss_p3']:.3f}")
            st.metric("Season MAE", f"{r['season_mae']:.3f}")
            st.metric("Seasons / matches", f"{int(r['n_seasons'])} / {int(r['n_matches']):,}")
    if not reg.empty:
        latest = reg.sort_values("timestamp").iloc[-1]
        st.caption(f"Latest experiment: {latest['name']} at {latest['timestamp']} (commit {latest['git_commit']}). Registry holds {len(reg)} experiments.")
    st.caption("Validation coverage: test seasons 2012-2026 (expanding and recent-8 windows); 2026 is included as the latest out-of-sample fold but has informed this research.")

with tabs[1]:
    st.subheader("Pooled comparison")
    scope = st.radio("Seasons", sorted(pooled["window_seasons"].unique()), horizontal=True)
    v = pooled[pooled["window_seasons"] == scope][["model", "n_seasons", "n_matches", "correct_3", "a3_in_top3", "exact_321", "unordered_top3", "log_loss_p3", "brier_p3", "ece_p3", "season_mae", "spearman", "rank_mae_top30", "top5_hit", "top10_hit", "winner_correct"]]
    st.dataframe(v.round(4), hide_index=True, width=W)
    pb = ml.csv("paired_bootstrap_vs_A")
    if not pb.empty:
        st.markdown("**Paired bootstrap vs Structural (A)** (mean difference A minus other, 95% CI over matches)")
        st.dataframe(pb.round(4), hide_index=True, width=W)
    ens = ml.csv("ensemble_by_season")
    if not ens.empty:
        st.markdown("**Ensemble research** (weights learned only from earlier seasons' OOF predictions)")
        st.dataframe(ens.groupby("model")[["correct_3", "log_loss_p3", "ece_p3", "exact_321", "season_mae", "spearman"]].mean().round(4).reset_index(), hide_index=True, width=W)
        st.caption(f"Learned ensemble beat the best single component on log loss in {summ.get('ensemble_seasons_learned_beats_best_single_logloss', '?')} of {summ.get('ensemble_n_seasons', '?')} seasons.")
        wts = ml.csv("ensemble_weights")
        if not wts.empty:
            st.dataframe(wts.round(3), hide_index=True, width=W)

with tabs[2]:
    st.subheader("Metrics by season")
    metric = st.selectbox("Metric", ["correct_3", "log_loss_p3", "exact_321", "a3_in_top3", "season_mae", "spearman", "ece_p3", "top10_hit"])
    piv = by_season.pivot(index="season", columns="model", values=metric)
    st.line_chart(piv, height=280)
    st.dataframe(piv.round(4).reset_index(), hide_index=True, width=W)
    st.caption("Stability: the standard deviation across seasons is in the pooled table (columns *_sd).")

with tabs[3]:
    st.subheader("Feature-family ablation (Structural model, recent-8 window, 2017-2026)")
    ab = ml.ablation_table()
    if ab.empty:
        st.info("Ablation outputs not available yet.")
    else:
        st.dataframe(ab.sort_values("d_correct_3").round(4), hide_index=True, width=W)
        st.caption("Rows are 'minus_<family>' (delta vs the full model; a NEGATIVE d_correct_3 / POSITIVE d_log_loss means the family helps) and 'plus_<family>' additions. "
                   "'seasons_worse_correct3' counts test seasons where removing the family hurt. A family is not promoted on 2026 alone.")

with tabs[4]:
    st.subheader("Error Lab")
    el = ml.error_lab()
    if el.empty:
        st.info("Error Lab dataset not built.")
    else:
        seasons = sorted(el["season"].unique())
        c1, c2, c3 = st.columns(3)
        season = c1.selectbox("Season", seasons, index=len(seasons) - 1)
        flt = c2.selectbox("Filter", ["All matches", "Missed 3-voter (A)", "Unanimous miss", "Structural-only correct", "ML-only correct", "Stats-only-only correct",
                                      "Defender 3-voter", "Key defender 3-voter", "Midfielder 3-voter", "Ruck 3-voter", "Forward 3-voter", "Losing-team 3-voter",
                                      "Close game", "Blowout", "Large disagreement", "Consensus failure"])
        s = el[el["season"] == season]
        rows = []
        for mid, g in s.groupby("match_id"):
            a3 = g[g["brownlow_votes"] == 3]
            if a3.empty:
                continue
            a3 = a3.iloc[0]
            picks = {k: g.loc[g[f"p3_{k}"].idxmax()] for k in ("A", "B", "C") if f"p3_{k}" in g and g[f"p3_{k}"].notna().any()}
            corr = {k: p["player_id"] == a3["player_id"] for k, p in picks.items()}
            spread = max(p[f"p3_{k}"] for k, p in picks.items()) - min(g.loc[a3.name, [f"p3_{k}" for k in picks]].astype(float)) if picks else 0
            rows.append({"match_id": mid, "actual_3": a3["player_name"], "role": a3["role"], "team_won": bool(a3["is_win"] == 1), "margin": a3["absolute_margin"],
                         "close": bool(a3["is_close_game"] == 1), "blowout": bool(a3["is_blowout"] == 1),
                         **{f"{k}_pick": f"{p['player_name']} ({p[f'p3_{k}'] * 100:.0f}%)" for k, p in picks.items()}, **{f"{k}_correct": v for k, v in corr.items()},
                         "A_p3_on_3voter": a3.get("p3_A"), "A_rank_of_3voter": a3.get("rank_in_match_A"), "spread": spread})
        t = pd.DataFrame(rows)
        if not t.empty:
            cond = {
                "All matches": t.index == t.index, "Missed 3-voter (A)": ~t["A_correct"], "Unanimous miss": ~t[[c for c in t if c.endswith("_correct")]].any(axis=1),
                "Structural-only correct": t["A_correct"] & ~t.get("B_correct", False) & ~t.get("C_correct", False),
                "ML-only correct": t.get("B_correct", False) & ~t["A_correct"] & ~t.get("C_correct", False),
                "Stats-only-only correct": t.get("C_correct", False) & ~t["A_correct"] & ~t.get("B_correct", False),
                "Defender 3-voter": t["role"].isin(["KEY_DEFENDER", "MEDIUM_DEFENDER"]), "Key defender 3-voter": t["role"] == "KEY_DEFENDER",
                "Midfielder 3-voter": t["role"] == "MIDFIELDER", "Ruck 3-voter": t["role"] == "RUCK", "Forward 3-voter": t["role"].isin(["KEY_FORWARD", "MEDIUM_FORWARD", "MIDFIELDER_FORWARD"]),
                "Losing-team 3-voter": ~t["team_won"], "Close game": t["close"], "Blowout": t["blowout"], "Large disagreement": t["spread"] > 0.35,
                "Consensus failure": ~t[[c for c in t if c.endswith("_correct")]].any(axis=1) & (t["spread"] < 0.1),
            }
            v = t[cond[flt]]
            c3.metric("Matches shown", f"{len(v)} of {len(t)}")
            st.dataframe(v.drop(columns=["match_id"]).round(3), hide_index=True, width=W, height=420)
            pick = st.selectbox("Match drill-down", ["--"] + v["match_id"].tolist())
            if pick != "--":
                g = s[s["match_id"] == pick].sort_values("p3_A", ascending=False).head(12)
                st.dataframe(g[["player_name", "team_id", "role", "brownlow_votes", "p3_A", "p3_B", "p3_C", "expected_votes_A", "expected_votes_B", "expected_votes_C", "rank_in_match_A", "disposals", "contested_possessions", "clearances", "goals", "marks", "tackles", "impact_z", "n_strong_teammates"]].round(3), hide_index=True, width=W)

with tabs[5]:
    st.subheader("Calibration (P3), calibrators fitted on earlier seasons only")
    cal = ml.csv("calibration_study")
    if not cal.empty:
        st.dataframe(cal.round(4), hide_index=True, width=W)
        st.caption("Isotonic / Platt are refit walk-forward and renormalised within match; a method is only worth adopting if ECE and log loss improve without hurting 3-vote accuracy.")

with tabs[6]:
    st.subheader("Disagreement as information")
    ds = ml.csv("disagreement_summary")
    if not ds.empty:
        st.dataframe(ds.round(4), hide_index=True, width=W)
        st.caption("Match level: when the three candidates' top picks agree vs differ, how often the consensus pick is right. Season level: |EV gap| between A and B vs error.")

with tabs[7]:
    st.subheader("Bias by role, margin and losing-team 3-voters")
    b = ml.csv("bias_study")
    if not b.empty:
        dim = st.selectbox("Dimension", sorted(b["dimension"].unique()))
        scope = st.radio("Scope", sorted(b["scope"].unique()), horizontal=True, key="bias_scope")
        st.dataframe(b[(b["dimension"] == dim) & (b["scope"] == scope)].drop(columns=["dimension", "scope"]).round(4), hide_index=True, width=W)
    fz = ml.csv("forensic_by_season")
    if not fz.empty:
        st.markdown("**2026 hypotheses tested year by year** (ruck EV bias, defender 3-voter hit rate)")
        model = st.selectbox("Model", sorted(fz["model"].unique()))
        st.dataframe(fz[fz["model"] == model].drop(columns=["model"]).round(3), hide_index=True, width=W)

with tabs[8]:
    st.subheader("Experiment registry")
    if reg.empty:
        st.info("Registry empty.")
    else:
        st.dataframe(reg.drop(columns=["notes"]).round(4), hide_index=True, width=W, height=420)
        st.download_button("Download registry (CSV)", reg.to_csv(index=False).encode(), file_name="2027_experiment_registry.csv", mime="text/csv")
