# data/deployment/

Git-tracked, deployment-only copies/derivations of local-only research
artefacts (`data/processed/*` is gitignored) and the Markets-repo betting
snapshot, so the Streamlit Cloud deployment is self-contained. Every loader
in `dashboard/data.py` / `dashboard/betting_data.py` prefers the real local
file when present and falls back to these only when it's missing (see
`dashboard/data._resolve_path`). No value here was recomputed or altered --
each file is either an exact copy, or an exact column/row subset, of the
same real production data.

| File | Built from | How |
|---|---|---|
| `model_core_2026_dashboard.parquet` | `data/processed/model_core_2026.parquet` | Same 2026 rows, only the ~40 columns the dashboard actually reads (identity/meta, the 8 z-scored stat families + their raw values, 5 team-rank/share stat pairs, 4 teammate-competition columns) -- column selection only, no value changes. |
| `scenario_predictions_2026.parquet` | `data/processed/scenario_predictions_2026.parquet` | Exact copy (already small). |
| `contender_probabilities_production.csv` | `data/processed/mc_totals_2026.npy` (100,000 real Monte Carlo draws) | Precomputed once via `src.models.order_scenarios.contender_probabilities` -- the identical function `pages/14_Order_Scenarios.py` calls locally. Avoids shipping a 113MB raw array; the deployed page reads the finished numbers instead of recomputing them from draws it doesn't have. |
| `contender_probabilities_objective.csv` | `data/processed/mc_totals_objective_2026.npy` (20,000 real draws) | Same method, Objective model. Avoids shipping a 30MB raw array. |
| `sportsbet_verified_value_opportunities.csv` | `~/code/AFL-BROWNLOW-MARKETS`'s finalized audited output | Real snapshot copy (see `dashboard/betting_data.py`'s own fallback logic, built in a prior task). |

To refresh any of these after new local data is generated, rerun the
selection/copy steps above (see git history / `dashboard/data.py` for the
exact column lists) -- there is no single script yet; each was built by hand
during the deployment-hardening pass documented in `PROJECT_STATE.md`.
