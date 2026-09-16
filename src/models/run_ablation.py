"""
Phase 4, section F/G: feature-family ablation study.

Uses Model 1 (Plackett-Luce) as the primary structurally-aligned architecture
(per docs/MODELLING_PLAN.md, this is the model whose structure actually
matches how Brownlow votes are awarded, so it is the fairest lens for asking
"does feature family X help"). Progressively adds feature families in a fixed
order (raw stats always included as the base), retrains from scratch each
time, and reports the metric deltas -- exactly the brief's "I want to know
which ideas materially improve it" framing.

To keep compute bounded (this project's judgement call, documented rather
than silently done): evaluated on 3 held-out test seasons (2023, 2024, 2025)
with an EXPANDING training window (all prior available CORE seasons), rather
than the full 11-season sweep used in run_backtest.py's headline comparison.
"""
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import match_level_metrics
from src.models.plackett_luce import PlackettLuceModel
from src.models import feature_sets

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TEST_SEASONS = [2023, 2024, 2025]

ABLATION_STEPS = [
    ("1_raw_only", feature_sets.FAMILIES["raw"]),
    ("2_+match_relative", feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"]),
    ("3_+context", feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"]),
    ("4_+teammate", feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"]),
    ("5_+role", feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"] + feature_sets.FAMILIES["role"]),
    ("6_+nonlinear", feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"] + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"]),
    ("7_+lagged_form", feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"] + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"] + feature_sets.FAMILIES["lagged_form"]),
    ("8_+win_margin_interaction", feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["teammate"] + feature_sets.FAMILIES["role"] + feature_sets.FAMILIES["nonlinear"] + feature_sets.FAMILIES["lagged_form"] + feature_sets.FAMILIES["win_margin_interaction"]),
]

# Special hypothesis-test variants (section G), evaluated the same way
HYPOTHESIS_VARIANTS = {
    "H1_raw_only_no_relative": feature_sets.FAMILIES["raw"],
    "H1_relative_instead_of_raw": feature_sets.FAMILIES["match_relative"],
    "H4_flat_winner_no_interaction": [c for c in ABLATION_STEPS[-1][1] if c != "win_x_margin"],
    "H4_with_winner_margin_interaction": ABLATION_STEPS[-1][1],
    "H6_context_no_nonlinear": feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"],
    "H6_context_with_nonlinear": feature_sets.FAMILIES["raw"] + feature_sets.FAMILIES["match_relative"] + feature_sets.FAMILIES["context"] + feature_sets.FAMILIES["nonlinear"],
}


def load_core():
    df = pd.read_parquet(PROCESSED_DIR / "model_core.parquet")
    all_needed = sorted(set(sum([f for _, f in ABLATION_STEPS], []) + sum(HYPOTHESIS_VARIANTS.values(), [])))
    df = feature_sets.prepare_features(df, all_needed, dropna=False)
    return df


def evaluate_feature_set(df: pd.DataFrame, feats: list) -> dict:
    d = df.dropna(subset=[c for c in feats if c in df.columns])
    metrics_by_season = []
    for test_season in TEST_SEASONS:
        train = d[d["season"] < test_season]
        test = d[d["season"] == test_season]
        if len(train) == 0 or len(test) == 0:
            continue
        model = PlackettLuceModel(feature_names=feats).fit(train)
        preds = model.predict(test)
        metrics_by_season.append(match_level_metrics(preds))
    if not metrics_by_season:
        return {}
    avg = pd.DataFrame(metrics_by_season).mean(numeric_only=True).to_dict()
    return avg


def run():
    df = load_core()
    rows = []
    for name, feats in ABLATION_STEPS:
        print(f"Ablation step: {name} ({len(feats)} features)")
        metrics = evaluate_feature_set(df, feats)
        rows.append({"step": name, "n_features": len(feats), **metrics})
    ablation = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(REPORTS_DIR / "feature_ablation.csv", index=False)
    print(ablation.to_string(index=False))

    hyp_rows = []
    for name, feats in HYPOTHESIS_VARIANTS.items():
        print(f"Hypothesis variant: {name} ({len(feats)} features)")
        metrics = evaluate_feature_set(df, feats)
        hyp_rows.append({"variant": name, "n_features": len(feats), **metrics})
    hyp = pd.DataFrame(hyp_rows)
    hyp.to_csv(REPORTS_DIR / "hypothesis_variants.csv", index=False)
    print(hyp.to_string(index=False))
    return ablation, hyp


if __name__ == "__main__":
    run()
