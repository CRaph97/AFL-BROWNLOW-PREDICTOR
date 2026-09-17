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
Projection Concentration, Uncertainty, Betting Opportunities.

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

## Deployment

The dashboard can be deployed to [Streamlit Community Cloud](https://streamlit.io/cloud) as-is.

- **Entrypoint:** `app.py` (repo root) — set this as the "Main file path" when creating the
  Streamlit Cloud app.
- **Dependency file:** `requirements.txt`.
- **Betting Opportunities page online:** this page normally reads a finalized, already-audited
  output file live from a sibling checkout of the separate `AFL-BROWNLOW-MARKETS` repo — which
  a Streamlit Cloud deployment has no access to. When that live checkout isn't reachable, the
  page automatically falls back to a bundled, timestamped static snapshot committed at
  `data/deployment/sportsbet_verified_value_opportunities.csv`, and the page clearly labels
  itself as running from that snapshot (with the snapshot's own "last refreshed" timestamp)
  rather than silently going stale. No Sportsbet scraping ever runs from the deployed app —
  all ingestion/scraping/mapping code stays in the separate `AFL-BROWNLOW-MARKETS` repo.
- **Refreshing the snapshot before a redeploy:** after `AFL-BROWNLOW-MARKETS` has produced a
  fresh `reports/sportsbet_verified_value_opportunities.csv`, copy it into this repo and push:

  ```bash
  cp ~/code/AFL-BROWNLOW-MARKETS/reports/sportsbet_verified_value_opportunities.csv \
     data/deployment/sportsbet_verified_value_opportunities.csv
  git add data/deployment/sportsbet_verified_value_opportunities.csv
  git commit -m "Refresh bundled Sportsbet snapshot for deployment"
  git push
  ```

  Streamlit Community Cloud redeploys automatically on push to the connected branch.

**Betting Opportunities:** compares Sportsbet Brownlow prices against this model's probabilities
(edge, expected value, model disagreement, structural-break sensitivity). This page is
**read-only downstream of the Brownlow model** in two senses: it never changes a model
prediction, and it doesn't even compute the betting-market numbers itself — all scraping,
mapping, and value-audit logic lives in a separate repo, [AFL-BROWNLOW-MARKETS](../AFL-BROWNLOW-MARKETS),
which this page reads a single finalized CSV from.

- **Where the data comes from:** `AFL-BROWNLOW-MARKETS/reports/sportsbet_verified_value_opportunities.csv`,
  read via `dashboard/betting_data.py`. Path defaults to `~/code/AFL-BROWNLOW-MARKETS`; override with
  the `AFL_BROWNLOW_MARKETS_PATH` environment variable if that repo is checked out elsewhere.
- **Refresh instructions:** this page does not fetch anything itself. In the AFL-BROWNLOW-MARKETS repo,
  run `python -m src.ingestion.refresh_sportsbet` (see that repo's `docs/REFRESH_WORKFLOW.md`), then
  restart this Streamlit app (or just this page) to pick up the new file.
- **If the file is missing or that repo isn't checked out**, the page shows a clear message instead of
  failing — it never crashes the rest of the dashboard.
- **No arbitrage, no automated betting, no stake sizing** — this page shows a probability comparison
  only. It does not place bets and never will.

Companion report: [`docs/2026_BROWNLOW_REVIEW_REPORT.md`](docs/2026_BROWNLOW_REVIEW_REPORT.md).
