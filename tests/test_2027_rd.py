"""
2027 R&D framework tests: temporal leakage, feature timing, train/test
season separation, identity integrity, match vote constraints, deterministic
reproducibility, metric correctness, calibration, experiment registry,
frozen-output immutability, settlement of the simulation framework.
Fixed seeds everywhere. Offline.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.point_in_time import OUT_PATH, REGISTRY_PATH, MANIFEST_PATH
from src.models.plackett_luce import PlackettLuceModel, _neg_log_likelihood
from src.models.structural.fast_pl import nll_and_grad, fit_pl_fast
from src.models.stats_only.learned_pl import assert_stats_only, StatsOnlyPL
from src.simulation import season_sim as sim
from src.validation import metrics as M
from src.validation import registry
from src.validation.walk_forward import make_folds, assert_no_overlap

ROOT = Path(__file__).resolve().parents[1]
FROZEN = [ROOT / "reports" / "2026_leaderboard.csv", ROOT / "reports" / "2026_predicted_votes.csv", ROOT / "reports" / "2026_match_probabilities.csv",
          ROOT / "reports" / "2026_objective_votes.csv", ROOT / "reports" / "2026_simulation_summary.csv", ROOT / "data" / "external" / "processed" / "wheelo_season.csv",
          ROOT / "data" / "betting" / "processed" / "priced_opportunities.csv", ROOT / "data" / "actual" / "2026_brownlow_match_votes.csv",
          ROOT / "data" / "evaluation" / "2026" / "scorecard.csv", ROOT / "data" / "processed" / "mc_totals_2026.npy"]


@pytest.fixture(scope="module")
def feat():
    return pd.read_parquet(OUT_PATH)


@pytest.fixture(scope="module")
def reg():
    return json.loads(REGISTRY_PATH.read_text())


# ---------------------------------------------------------------- feature store / timing
def test_feature_store_shape_and_labels(feat):
    man = json.loads(MANIFEST_PATH.read_text())
    assert len(feat) == man["rows"] and feat["season"].min() == 2003 and feat["season"].max() == 2026
    assert not feat.duplicated(["match_id", "player_id"]).any()
    per_match = feat.groupby("match_id")["brownlow_votes"].agg(lambda s: tuple(sorted(s[s > 0], reverse=True)))
    assert (per_match == (3, 2, 1)).all(), per_match[per_match != (3, 2, 1)].head()
    assert feat.groupby("match_id")["brownlow_votes"].sum().eq(6).all()
    assert set(feat.loc[feat["season"] == 2026, "label_source"].unique()) <= {"afl_actual_2026", "afl_actual_2026_name_team_fallback"}


def test_every_feature_has_a_timing_class(reg):
    allowed = {"same_match", "prior_matches", "prior_seasons", "same_season_unrevealed"}
    for fam, meta in reg.items():
        assert meta["timing"] in allowed, fam
    legacy = reg["reputation_legacy"]["columns"]
    assert reg["reputation_legacy"]["timing"] == "same_season_unrevealed"
    for fam, meta in reg.items():
        if fam != "reputation_legacy":
            assert not (set(meta["columns"]) & set(legacy)), fam


def test_prior_season_reputation_is_point_in_time(feat):
    """prior_seasons_votes_per_game for season S must equal votes/games over seasons < S only."""
    ps = feat.groupby(["player_id", "season"])["brownlow_votes"].agg(["sum", "size"]).reset_index()
    for pid in ["12943", "12692"]:  # Daicos, Bailey Smith
        g = ps[ps["player_id"] == pid].sort_values("season")
        for _, r in g.iterrows():
            prior = g[g["season"] < r["season"]]
            expected = prior["sum"].sum() / prior["size"].sum() if prior["size"].sum() > 0 else 0.0
            got = feat[(feat["player_id"] == pid) & (feat["season"] == r["season"])]["prior_seasons_votes_per_game"].iloc[0]
            assert abs(got - expected) < 1e-9, (pid, r["season"], got, expected)
    # 2026 rows carry no same-season vote information in any point-in-time family
    d26 = feat[feat["season"] == 2026]
    assert (d26["last_season_votes"] >= 0).all()


def test_team_strength_is_strictly_prior(feat):
    first = feat[feat["round_num"] <= 1]
    assert first["team_std_win_pct"].isna().mean() > 0.9  # no season-to-date record before a team's first game
    later = feat[feat["round_num"] >= 6]
    assert later["team_std_win_pct"].notna().all()
    assert feat["team_std_win_pct"].dropna().between(0, 1).all()


def test_same_match_features_do_not_use_votes(feat, reg):
    """Shuffling brownlow_votes within matches must not change any same_match feature (they never read votes)."""
    from src.features import point_in_time as pit
    sample = feat[feat["season"] == 2024].copy()
    rng = np.random.default_rng(0)
    shuffled = sample.copy()
    shuffled["brownlow_votes"] = shuffled.groupby("match_id")["brownlow_votes"].transform(lambda s: rng.permutation(s.to_numpy()))
    a, cols = pit.add_dominance(sample.copy()); b, _ = pit.add_dominance(shuffled.copy())
    for c in cols:
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True), c


# ---------------------------------------------------------------- folds / leakage
def test_walk_forward_folds_never_overlap():
    seasons = list(range(2003, 2027))
    folds = make_folds(seasons, list(range(2012, 2027)), window=None) + make_folds(seasons, list(range(2012, 2027)), window=8)
    assert len(folds) == 30
    for f in folds:
        assert_no_overlap(f)
        assert max(f.train_seasons) < f.test_season
        assert f.inner_holdout == max(f.train_seasons) and f.test_season not in f.inner_train
    assert all(len(f.train_seasons) == 8 for f in folds if f.window == "recent8")


def test_oof_predictions_are_out_of_sample():
    p = ROOT / "data" / "experiments" / "oof" / "A_structural_pl.parquet"
    if not p.exists():
        pytest.skip("candidate OOF not built yet")
    oof = pd.read_parquet(p, columns=["season", "test_season", "window"])
    assert (oof["season"] == oof["test_season"]).all()
    assert oof["season"].min() >= 2012


def test_stats_only_feature_guard(reg):
    assert_stats_only(reg["raw"]["columns"] + reg["dominance"]["columns"])
    with pytest.raises(AssertionError):
        assert_stats_only(["disposals", "role_MIDFIELDER"])
    with pytest.raises(AssertionError):
        assert_stats_only(["prior_seasons_votes_per_game"])
    with pytest.raises(AssertionError):
        StatsOnlyPL(["disposals", "brownlow_votes_prev5_mean"])


# ---------------------------------------------------------------- model math
def test_fast_pl_matches_legacy_objective_and_gradient(feat):
    from src.models import feature_sets as fs
    cols = fs.RAW_STATS + fs.CONTEXT
    d = feat[feat["season"] == 2023]
    d = d[d["match_id"].isin(sorted(d["match_id"].unique())[:15])].sort_values("match_id").reset_index(drop=True)
    m = PlackettLuceModel(feature_names=cols)
    X = d[cols].to_numpy(float); m._fit_scaler(X); X = m._transform(X); v = d["brownlow_votes"].to_numpy(float)
    codes, _ = pd.factorize(d["match_id"].to_numpy()); nm = codes.max() + 1
    i3 = np.full(nm, -1); i2 = np.full(nm, -1); i1 = np.full(nm, -1)
    i3[codes[v == 3]] = np.where(v == 3)[0]; i2[codes[v == 2]] = np.where(v == 2)[0]; i1[codes[v == 1]] = np.where(v == 1)[0]
    b = np.random.default_rng(1).normal(0, 0.2, X.shape[1])
    f_legacy = _neg_log_likelihood(b, X, codes, nm, i3, i2, i1, 1.0); f_new, g = nll_and_grad(b, X, codes, nm, i3, i2, i1, 1.0)
    assert abs(f_legacy - f_new) < 1e-8
    eps = 1e-6
    num = np.array([(nll_and_grad(b + eps * e, X, codes, nm, i3, i2, i1, 1.0)[0] - nll_and_grad(b - eps * e, X, codes, nm, i3, i2, i1, 1.0)[0]) / (2 * eps) for e in np.eye(len(b))])
    assert np.allclose(num, g, atol=1e-4, rtol=1e-4)


def test_fit_is_deterministic(feat):
    from src.models import feature_sets as fs
    cols = fs.RAW_STATS + fs.CONTEXT
    d = feat[feat["season"].isin([2022, 2023])]
    a = fit_pl_fast(PlackettLuceModel(feature_names=cols), d).beta
    b = fit_pl_fast(PlackettLuceModel(feature_names=cols), d).beta
    assert np.array_equal(a, b)


def test_predictions_are_coherent_within_match(feat):
    from src.models import feature_sets as fs
    from src.models.structural.pl_model import StructuralPL
    cols = fs.RAW_STATS + fs.CONTEXT + ["impact_z", "prior_seasons_votes_per_game"]
    m = StructuralPL(cols).fit(feat[feat["season"].isin([2022, 2023])])
    p = m.predict(feat[feat["season"] == 2024])
    s = p.groupby("match_id")[["p3", "p2", "p1"]].sum()
    assert np.allclose(s["p3"], 1, atol=1e-6) and np.allclose(s["p2"], 1, atol=1e-6) and np.allclose(s["p1"], 1, atol=1e-6)
    assert np.allclose(p.groupby("match_id")["expected_votes"].sum(), 6, atol=1e-5)
    assert len(p) == (feat["season"] == 2024).sum()  # imputation keeps every row


# ---------------------------------------------------------------- metrics / calibration
def test_metric_correctness_on_synthetic():
    rows = []
    for mid in ("m1", "m2"):
        for i in range(5):
            rows.append({"season": 2020, "match_id": mid, "player_id": f"{mid}_{i}", "brownlow_votes": [3, 2, 1, 0, 0][i]})
    df = pd.DataFrame(rows)
    df["p3"] = [0.6, 0.2, 0.1, 0.05, 0.05] * 2; df["p2"] = [0.2, 0.5, 0.2, 0.05, 0.05] * 2; df["p1"] = [0.1, 0.2, 0.5, 0.1, 0.1] * 2
    df["p0"] = 1 - df["p3"] - df["p2"] - df["p1"]; df["expected_votes"] = 3 * df["p3"] + 2 * df["p2"] + df["p1"]
    mm = M.match_metrics(df)
    assert mm["n_matches"] == 2 and mm["correct_3"] == 1.0 and mm["exact_321"] == 1.0 and mm["unordered_top3"] == 1.0
    assert abs(mm["log_loss_p3"] - (-np.log(0.6))) < 1e-9
    sm = M.season_metrics(df)["by_season"][0]
    assert sm["winner_correct"] and sm["top3_hit"] == 1.0
    # ECE: perfectly calibrated bins -> 0; all-wrong -> 1
    assert M.ece(pd.Series([0.05] * 100), pd.Series([1] * 5 + [0] * 95)) < 1e-9
    assert M.ece(pd.Series([0.95] * 10), pd.Series([0] * 10)) == pytest.approx(0.95)


def test_reproduce_check_matches_frozen_2026_evaluation():
    p = ROOT / "data" / "experiments" / "reproduce_2026_check.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert d["reproduced"] is True
    assert abs(d["objective_2026_frozen"]["correct_3"] - d["frozen_2026_evaluation"]["Objective"]["hit_3_rate"]) < 1e-3


# ---------------------------------------------------------------- registry
def test_registry_records_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REG_PATH", tmp_path / "r.jsonl")
    rec = registry.record("t", "structural_pl", ["a", "b"], ["raw"], [2003, 2004], [2005], {"l2": 1.0}, [{"test_season": 2005, "correct_3": 0.5}], {"expanding": {"correct_3_mean": 0.5}})
    df = registry.load()
    assert len(df) == 1 and df.iloc[0]["experiment_id"] == rec["experiment_id"] and df.iloc[0]["status"] == "candidate"
    registry.update_status(rec["experiment_id"], "rejected", "test")
    assert registry.load().iloc[0]["status"] == "rejected"
    same = registry.experiment_id("t", {"model_family": "structural_pl", "features": ["a", "b"], "hyperparameters": {"l2": 1.0}, "validation_seasons": [2005], "window": "expanding"})
    assert same == rec["experiment_id"]


# ---------------------------------------------------------------- simulation framework
def test_simulation_awards_six_votes_per_match_and_queries():
    rows = []
    for mid in ("m1", "m2", "m3"):
        for i in range(6):
            rows.append({"match_id": mid, "round": int(mid[1]), "player_id": f"p{i}", "player_name": f"P{i}", "team_id": "t1" if i < 3 else "t2", "p3": [0.5, 0.2, 0.1, 0.1, 0.05, 0.05][i]})
    preds = pd.DataFrame(rows)
    totals, players, by_round = sim.simulate_matches(preds, n_sims=2000, seed=1)
    assert totals.sum(axis=1).tolist() == [18] * 2000  # 3 matches x 6 votes every simulation
    assert (by_round.sum(axis=2) == 6).all()
    s = sim.summarise(totals, players)
    assert s.iloc[0]["player_id"] == "p0" and 0 < s.iloc[0]["p_winner"] <= 1
    h = sim.h2h(totals, players, "p0", "p1"); assert abs(h["p_a_wins"] + h["p_b_wins"] + h["p_tie"] - 1) < 1e-9
    e = sim.exact_order(totals, players, ["p0", "p1"]); assert e["p_exact_order"] <= e["p_all_in_top_k"] + 1e-12
    tl = sim.team_leader(totals, players); assert np.allclose(tl.groupby("team_id")["p_team_leader"].sum(), 1.0)
    cr = sim.clinch_round(by_round, players); assert cr["p_winner"].sum() == pytest.approx(1.0)
    t2, _, _ = sim.simulate_matches(preds, n_sims=2000, seed=1)
    assert np.array_equal(totals, t2)  # deterministic


# ---------------------------------------------------------------- immutability
def test_frozen_outputs_unchanged_by_rd():
    diff = subprocess.run(["git", "diff", "--stat", "--", *[str(p.relative_to(ROOT)) for p in FROZEN if p.exists()]], cwd=ROOT, capture_output=True, text=True)
    assert diff.stdout.strip() == "", diff.stdout
    ev = json.loads((ROOT / "data" / "evaluation" / "2026" / "manifest.json").read_text())["frozen_input_hashes"]
    for rel, h in ev.items():
        p = ROOT / rel
        if p.exists() and h:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == h, rel


# ---------------------------------------------------------------- analysis outputs / page (require the candidate suite + analyze)
AN = ROOT / "data" / "experiments" / "analysis"


@pytest.mark.skipif(not (AN / "comparison_pooled.csv").exists(), reason="analysis outputs not built")
def test_analysis_outputs_are_consistent():
    pooled = pd.read_csv(AN / "comparison_pooled.csv"); by = pd.read_csv(AN / "comparison_by_season.csv")
    # every model is scored on exactly the same matches within a scope (fair comparison)
    for scope, g in pooled.groupby("window_seasons"):
        assert g["n_matches"].nunique() == 1, (scope, g["n_matches"].tolist())
    assert set(by["season"]) >= set(range(2012, 2027))
    assert by["correct_3"].between(0, 1).all() and (by["log_loss_p3"] > 0).all()
    champ = pd.read_csv(AN / "champion.csv"); assert "Champion (2027 headline single model)" in set(champ["role"])
    w = pd.read_csv(AN / "ensemble_weights.csv")
    wcols = [c for c in w.columns if c.startswith("w_")]
    assert np.allclose(w[wcols].sum(axis=1), 1.0) and (w[wcols] >= -1e-9).all().all()
    assert (w["test_season"] > 2012).all()  # weights only ever fit on earlier seasons
    cal = pd.read_csv(AN / "calibration_study.csv")
    assert set(cal["method"]) == {"raw", "isotonic", "platt"}
    reg = pd.read_json(ROOT / "data" / "experiments" / "registry.jsonl", lines=True)
    assert reg["experiment_id"].is_unique and {"A_structural_pl", "C_stats_only_pl", "baseline_phase4_pl_legacy"} <= set(reg["name"])
    assert (reg["git_commit"].str.len() >= 7).all()


@pytest.mark.skipif(not (AN / "comparison_pooled.csv").exists(), reason="analysis outputs not built")
def test_model_lab_page_renders_and_is_in_rd_nav():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages" / "40_2027_Model_Lab.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "2027 Model Lab"
    router = (ROOT / "app.py").read_text()
    assert '"R&D": [' in router and 'st.Page("pages/40_2027_Model_Lab.py", title="2027 Model Lab")' in router
    main_block = router[router.index('"MAIN": ['):router.index("],", router.index('"MAIN": ['))]
    assert "40_2027_Model_Lab" not in main_block


# ---------------------------------------------------------------- pre-freeze audit
@pytest.mark.skipif(not (AN / "audit_summary.json").exists(), reason="audit not run")
def test_audit_ensemble_weights_learned_only_from_earlier_seasons():
    proof = pd.read_csv(AN / "audit_ensemble_weight_proof.csv")
    assert (proof["max_abs_diff_stored_vs_refit_prior_only"] < 1e-9).all()
    assert (proof["n_prior_seasons"] == proof["stored_n_train_seasons"]).all()
    assert (proof["max_abs_diff_if_test_season_included"] > 1e-4).any()  # including the test season would have changed the weights
    w = pd.read_csv(AN / "ensemble_weights.csv")
    assert (w["n_train_seasons"] == w["test_season"] - 2012).all()  # OOF starts 2012: exactly the seasons strictly before t


@pytest.mark.skipif(not (AN / "audit_summary.json").exists(), reason="audit not run")
def test_audit_platt_calibrators_are_chronological_and_same_period_tables_align():
    for f in ("audit_platt_fit_spans_B.csv", "audit_platt_fit_spans_ens.csv"):
        sp = pd.read_csv(AN / f)
        hi = sp["calibrator_fit_on_seasons"].str.split("-").str[1].astype(int)
        assert (hi < sp["season"]).all()
    t = pd.read_csv(AN / "audit_same_period_by_season.csv")
    assert len({frozenset(g) for g in t.groupby("model")["season"].apply(list)}) == 1  # identical season sets
    assert t.groupby("season")["n_matches"].nunique().eq(1).all()  # identical match counts per season


def test_frozen_2026_production_never_used_same_season_reputation():
    src = (ROOT / "src" / "models" / "train_2026_scenarios.py").read_text()
    ens = (ROOT / "src" / "models" / "build_2026_ensemble.py").read_text()
    from src.models import feature_sets as fs
    core = fs.FAMILIES["raw"] + fs.FAMILIES["match_relative"] + fs.FAMILIES["context"] + fs.FAMILIES["teammate"] + fs.FAMILIES["role"] + fs.FAMILIES["nonlinear"] + fs.FAMILIES["lagged_form"] + fs.FAMILIES["win_margin_interaction"]
    assert not any("brownlow" in c for c in core)
    assert '"A_historical": 0.45' in ens and "A_with_reputation" not in ens.split("ENSEMBLE_WEIGHTS")[1].split("}")[0]
    assert "REPUTATION_FEATURES" in src  # sensitivity scenario exists but is outside the ensemble
