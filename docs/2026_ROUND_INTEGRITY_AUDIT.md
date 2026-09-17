# 2026 Round-Label Integrity Audit

Status: **Resolved.** Triggered by user-reported evidence that Geelong vs Collingwood
(2026-05-09) and Brisbane vs Geelong (2026-05-14) were labelled Round 10/11 instead of the
correct Round 9/10.

## 1. Root cause

afltables' raw `Round` column for the 2026 season labels the season's split Opening Round
(2026-03-05 to 03-08, 5 matches, only 10 of 18 teams) simply as Round `"1"`, then numbers
every subsequent round sequentially (`2, 3, 4, ... 25`) with no distinguishing marker. The
AFL's own official numbering treats the Opening Round as unnumbered and starts official
Round 1 the following week (2026-03-12 to 03-15, 9 matches, full round). This shifts every
raw round from 2 onward by **+1** relative to the official label: raw round `N` (N&ge;2) =
official round `N-1`; raw round 1 = Opening Round.

Confirmed directly against raw `data/raw/fitzroy_data/afldata.rda` and both user-supplied
examples:

| Match | Date | Raw (afltables) round | Official round |
|---|---|---|---|
| Geelong vs Collingwood | 2026-05-09 | 10 | **9** |
| Brisbane vs Geelong | 2026-05-14 | 11 | **10** |

This is a genuine gap in the Phase 1/2 finding recorded in `docs/DATA_SOURCE_AUDIT.md`
("afltables' and footywire's round numbers can differ by one ... never join across sources
on round number, always use date"). That finding addressed *cross-source join* risk only,
and was correctly enforced everywhere (grep-audited: no merge/join/sort-by-index anywhere
in `src/` or `dashboard/` uses `round` as a key). It did not check whether afltables' own
round *label* matches the AFL's official round label within a single source. It does not,
for any season with an Opening Round (2023 onward). This fix addresses 2026 only, per the
live audit's scope; 2023-2025 likely carry the same mislabeling but are untouched here
since they are validated Phase 2/4 training data -- **flagged as a follow-up item requiring
separate user approval before any change**, not fixed opportunistically.

## 2. Bug classification: A (display/label-only)

Proven, not assumed:

- **No join anywhere uses `round` as a key.** All merges use `match_id`, `player_id`, or a
  `(date, team, opponent, surname)` composite. All temporal/lagged features
  (`build_lagged_form_features.py`, `build_role_lagged.py`) sort by `date`, never `round`.
  Grep-verified across the full `src/` and `dashboard/` trees.
- **Shaun Mannagh's Brisbane-vs-Geelong stat line (30 disposals, 5 goals, 3 goal assists)
  was already correct** in the raw data and every processed table, attached to the right
  match, before this fix touched anything -- only its `round` field said 11 instead of 10.
- **The corrected leaderboard is numerically identical to the pre-fix leaderboard** (Nick
  Daicos 47.23048644657416, Bailey Smith 35.49251355832686, ... to 14 decimal places) after
  the full 2026 feature/scenario/ensemble/Monte Carlo pipeline was rebuilt from the
  round-corrected source data. This is direct proof, not inference: the round label change
  produced zero change in any probability, expected-vote, or season total.
- Therefore: **no metadata misalignment (B), no prediction misalignment (C), and no
  training contamination (D)** -- round was never a feature, so nothing downstream of the
  label itself was ever wrong.

## 3. Fix

Fixed at the earliest layer, not patched in the dashboard:

- `src/data/round_normalization_2026.py` (new): `official_round()`, `official_round_label()`,
  `fix_match_id()` -- the single source of truth for this transform.
- `src/data/build_2026_extension.py`: now computes the official round before constructing
  `match_id`, so both `player_match_core_1984_2026.parquet` and
  `player_match_advanced_2010_2026.parquet` carry the correct round/match_id for 2026 rows
  at the root. Historical (pre-2026) rows are untouched.
- Every downstream 2026 artifact was **regenerated from this corrected root** by rerunning
  the actual pipeline (`build_2026_features.py` &rarr; `train_2026_scenarios.py` &rarr;
  `build_2026_ensemble.py` &rarr; `build_2026_outputs.py` &rarr; `run_2026_montecarlo.py` &rarr;
  `scripts/audit_2026_forecast.py`), not hand-patched -- an earlier attempt at a faster
  hand-patch script had a real bug (accidentally applied twice due to a retry, corrupting
  round values by an extra -1) and was caught by comparing patched output against the clean
  root before it went anywhere near a deliverable; that script was deleted and the safer
  full-rebuild path used instead.
- `round` is descriptive metadata only (0 = Opening Round, 1-24 = Round 1-24); `match_id`
  remains the identity key and was never used for round-based lookups.

## 4. Verification

- `reports/2026_match_identity_audit.csv`: all 207 real 2026 home-and-away matches
  reconciled against the raw afltables fixture. **207/207** on every check: `identity_match`,
  `round_match`, `score_match`, `prediction_match`.
- `tests/test_2026_match_integrity.py` (10 new tests, all passing): pins both user-reported
  examples, Shaun Mannagh's exact stat line, the official-round mapping function, match_id
  rewriting, season-wide reconciliation, no-duplicate/no-missing-match invariants, and
  player/opponent/team consistency.
- Full project suite: **37/37 tests pass** (27 pre-existing + 10 new).
- Season total expected votes: exactly 1,242.0 (207 &times; 6), unchanged from before the fix.
- Dashboard smoke-tested against the corrected data: loads and serves without error.

## 5. What changed / what didn't

**Changed:** the `round` value and the round-embedded segment of `match_id` for 2026 rows
only, in every file that carried them (`model_core_2026.parquet`, `model_advanced_2026.parquet`,
`player_match_role_lagged_2026.parquet`, `scenario_predictions_2026.parquet`,
`all_2026_scenarios_and_ensemble.parquet`, `common_scenario_utilities_2026.parquet`,
`2026_match_probabilities.csv`, `2026_predicted_votes.csv`, `2026_match_explanations.csv`,
`2026_round_by_round_contenders.csv`, `2026_daicos_round_by_round.csv`,
`reports/2026_data_extension_validation.json`, `docs/2026_FINAL_AUDIT.md`).

**Did not change:** any P(3)/P(2)/P(1)/EV value, any season total, the leaderboard order or
values, the Monte Carlo simulation distribution, or any model coefficient. No season totals
changed for any player.

## 6. Follow-up: resolved

**Update:** the 2023-2025 follow-up audit is complete -- see `docs/ROUND_NORMALIZATION.md`.
Result: **2023 needed no correction at all** (confirmed unaffected, not assumed); **2024 and
2025 had the same mislabeling as 2026** and have been fixed the same way. As predicted here,
no Phase 4 model result changed (empirically reverified, not just argued from the grep audit).
