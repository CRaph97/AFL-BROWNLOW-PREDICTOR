#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== [1/5] window comparison ==="
$PY -m src.models.run_window_comparison_2026

echo "=== [2/5] train 2026 scenarios ==="
$PY -m src.models.train_2026_scenarios

echo "=== [3/5] build ensemble + structural-break sensitivity ==="
$PY -m src.models.build_2026_ensemble

echo "=== [4/5] Monte Carlo simulation ==="
$PY -m src.models.run_2026_montecarlo

echo "=== [5/5] final outputs (leaderboard, contenders, quality checks) ==="
$PY -m src.models.build_2026_outputs

echo "=== PHASE5_PIPELINE_COMPLETE ==="
