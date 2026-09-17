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
| `contender_probabilities_production.csv` | `data/processed/mc_totals_2026.npy` (100,000 real Monte Carlo draws) | Precomputed once via `src.models.order_scenarios.contender_probabilities` -- the identical function `pages/14_Order_Scenarios.py` calls locally. Avoids shipping a 113MB raw array; the deployed page reads the finished numbers instead of recomputing them from draws it doesn't have. Regenerated after a sort-key fix (was sorted by `prob_winner`, which is ~0.0 for nearly every real contender and so buried genuine top-EV players like Bontempelli behind arbitrary tie order under a `.head(N)` cutoff; now sorted by `mean_votes` -- see the function's docstring). |
| `contender_probabilities_objective.csv` | `data/processed/mc_totals_objective_2026.npy` (20,000 real draws) | Same method, Objective model. Avoids shipping a 30MB raw array. Regenerated for the same sort-key fix. |
| `sportsbet_verified_value_opportunities.csv` | `~/code/AFL-BROWNLOW-MARKETS`'s finalized audited output | Real snapshot copy (see `dashboard/betting_data.py`'s own fallback logic, built in a prior task). **Stale as of the hyphenated-surname join fix below** -- this snapshot's `model_probability` column was baked from Production probabilities computed before that fix, so any market row for one of the 12 fixed players or a teammate whose probabilities were legitimately renormalised (see `src/data/build_2026_extension.py`'s `_normalise_surname` fix) is now out of date. Refreshing it requires re-running ingestion in the Markets repo, which is out of scope for this repo alone -- flagged here rather than left silently inconsistent. |

`model_core_2026_dashboard.parquet` and `contender_probabilities_production.csv`
were regenerated (not just copied) after a Scenario C / footywire identity-join
fix for hyphenated surnames (12 players, e.g. "Wanganeen-Milera" -> footywire's
"W-Milera") -- `model_core_2026_dashboard.parquet` came out byte-identical
(CORE never used the broken join), `contender_probabilities_production.csv`
changed for the 12 fixed players plus their match-level competitors via
legitimate probability renormalisation. `contender_probabilities_objective.csv`
was NOT touched -- Objective doesn't use this join and its outputs are
confirmed byte-identical before/after.

To refresh any of these after new local data is generated, rerun the
selection/copy steps above (see git history / `dashboard/data.py` for the
exact column lists) -- there is no single script yet; each was built by hand
during the deployment-hardening pass documented in `PROJECT_STATE.md`.
