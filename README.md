# AFL Brownlow Predictor

A quantitative research project modelling the probability distribution of AFL Brownlow Medal votes
(player × match), aggregated into season-level forecasts with uncertainty. Not a raw-stat heuristic — see
`docs/MODELLING_PLAN.md` for the research philosophy and `docs/DATA_SOURCE_AUDIT.md` for what data actually
underpins it.

**Status:** Phase 1 (data audit & planning) complete, pending review. No data collected, no model trained yet.
See `PROJECT_STATE.md` for current state, findings, and decisions needed before Phase 2.

## Documents

- [`PROJECT_STATE.md`](PROJECT_STATE.md) — current phase, findings, open decisions, next steps
- [`docs/DATA_SOURCE_AUDIT.md`](docs/DATA_SOURCE_AUDIT.md) — data source inventory and availability
- [`docs/MODELLING_PLAN.md`](docs/MODELLING_PLAN.md) — modelling approach, validation framework, 2026 regime-change treatment
- [`docs/FEATURE_CANDIDATES.md`](docs/FEATURE_CANDIDATES.md) — feature registry by category

## Repository layout

See the "Proposed architecture" section of `PROJECT_STATE.md`.

## 2026 Brownlow Review Dashboard

A local, read-only Streamlit dashboard for inspecting the final, audited 2026 forecast before
and during Brownlow count night. It never retrains the model, changes weights, or alters a
prediction — it only loads and re-derives presentational fields from the frozen production
outputs in `reports/2026_*.csv` and `data/processed/model_core_2026.parquet`.

**Install** (adds `streamlit` and `plotly` to the existing pinned environment):

```bash
pip install -r requirements.txt
```

**Run:**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Pages (left sidebar): Player Detail, Scenario Comparison,
Model Disagreement, Brownlow Night Tracker, Round View, Match Detail, Defender Bias Watchlist,
Projection Concentration, Uncertainty.

**Files that power it:**
- `app.py` — landing page (summary cards + Top 20 leaderboard)
- `pages/*.py` — one file per additional view (Streamlit's native multipage mechanism)
- `dashboard/data.py` — the only place that reads the underlying files; every loader is
  documented with the exact source column(s) it uses, and every derived field (e.g. a "win by
  12" result string, or a player's top-3 statistical drivers) is built from columns that
  already exist in the production outputs — nothing here is a new model or a new number
- `tests/test_dashboard_data.py` — integrity checks (leaderboard matches the production CSV,
  all 207 matches load, probabilities are never renormalised, round filters return correct
  counts) — run with `pytest tests/test_dashboard_data.py`

**Brownlow Night Tracker:** lets you enter each round's actual 3-2-1 votes as they're read out,
and compares the resulting actual cumulative total against the model's predicted cumulative
total (ahead/on/behind model). This is **local browser-session state only** — nothing is
written to disk or shared between sessions, and it never overwrites a model prediction; it
only adds a separate "actual" column alongside it.

Companion report: [`docs/2026_BROWNLOW_REVIEW_REPORT.md`](docs/2026_BROWNLOW_REVIEW_REPORT.md).
