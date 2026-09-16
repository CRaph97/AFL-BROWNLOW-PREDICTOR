# 2026 Modelling Methodology — Phase 5

Status: **Complete.**
Last updated: 2026-09-17

## 1. Why no new architecture was introduced

Phase 4 backtested four model architectures across 22 walk-forward folds and found Plackett-Luce
(Model 1) wins on every single tracked metric (`docs/MODEL_BACKTEST.md`). Phase 5 does not
re-open that question. Every 2026 scenario below uses the SAME Plackett-Luce architecture; only the
**training window** and **feature set** vary across scenarios. Introducing a new, untested
architecture specifically for the highest-stakes deliverable of the project (the actual 2026
forecast) would trade validated performance for an unvalidated idea, which the brief's own
instructions ("Do not tune... build final ranking models" as new work, "avoid false precision")
argue against.

## 2. Data extension to 2026

`src/data/build_2026_extension.py` extends the validated CORE/ADVANCED pipelines with the complete,
real 2026 home-and-away season (no synthetic or placeholder rows) sourced from the same fitzRoy
mirror used throughout this project. See `docs/2026_DATA_VALIDATION.md` for full coverage checks.
Critically, this is an ADDITIVE extension: `player_match_core_1984_2025.parquet` and
`model_core.parquet` (the files every Phase 4 backtest and doc reference) are untouched. New,
separate files (`player_match_core_1984_2026.parquet`, `model_core_2026.parquet`,
`model_advanced_2026.parquet`) carry the 2026 rows.

## 3. Feature engineering for 2026 (no leakage)

`src/features/build_2026_features.py` reruns the exact same Phase 3/4 feature functions
(`build_relative_features`, `build_context_features`, `build_teammate_features`,
`build_composite_indices`, `build_lagged_form_features`, `build_role_lagged`'s lagged-proxy
classifier) over the 2026-extended table. Every one of these functions is a pure function of
box-score stats using only within-match/within-team groupings or `.shift(1)`-based lagged/expanding
windows — none of them can see a 2026 vote (there are none) or a future 2026 match. This is the
same leakage-safety property Phase 4 established for the historical backtest, applied unchanged to
2026 inference.

**Documented limitation**: the real-label position roster (`role_reference_2021_2025.parquet`,
sourced from torpdata) does not extend to 2026. Every 2026 row's `role` therefore comes from the
lagged, prior-games-only proxy classifier (Tier B of `build_role_lagged.py`'s hierarchy), never the
real label — see `docs/2026_DATA_VALIDATION.md` for the resulting UNKNOWN rate in early rounds.

## 4. Recent-history window comparison (section 2 of the brief)

`src/models/run_window_comparison_2026.py` tests training windows of 3, 5, 8 and expanding seasons,
plus a recency-weighted variant (exponential decay, 5-season half-life, full expanding history), for
Model 1 on CORE, restricted to test seasons 2021-2025 (bounded compute; this is also the most
relevant evidence for a near-term 2026 forecast). Results: see `reports/window_comparison_2026.csv`
and `reports/window_comparison_2026_summary.csv`, and the summary table below.

| Window | mean correct_3 | mean exact_321 | mean log_loss | mean brier | mean rank_corr |
|---|---|---|---|---|---|
| **recent5** | **0.5682** | 0.0792 | 0.1855 | 0.0884 | **0.4149** |
| recency_weighted (5-season half-life) | 0.5649 | 0.0783 | **0.1853** | 0.0888 | 0.4145 |
| recent8 (Phase 4's original headline choice) | 0.5605 | **0.0816** | 0.1867 | 0.0891 | 0.4141 |
| recent3 | 0.5585 | 0.0823 | 0.1860 | **0.0886** | 0.4147 |
| expanding | 0.5530 | 0.0718 | 0.1869 | 0.0897 | 0.4135 |

(Test seasons 2021-2025; full detail in `reports/window_comparison_2026.csv`.) **Recent windows beat
expanding for every metric except exact_321** (where recent8 is actually best) -- a second, independent
confirmation of Phase 4's own finding that recency helps this specific architecture. recent5 is the
single best window on 3 of 5 metrics and very close on the other two, so it is used, programmatically
selected, for Scenario B.

**Decision**: Scenario A (the historical-behaviour baseline) keeps Phase 4's validated recent-8
window. Scenario B's window is chosen programmatically as the narrowest window within 0.5
percentage points of the best-observed `correct_3` — see the code comment in
`train_2026_scenarios.py::_select_window_b` for the exact rule and the result actually selected.

## 5. The four 2026 scenarios

| Scenario | Feature set | Training window | Represents |
|---|---|---|---|
| A — Historical behaviour | CORE (raw + relative + context + teammate + role + nonlinear + lagged-form + win×margin) | recent-8 seasons (2018-2025) | "2026 votes like recent past seasons" |
| B — Recent era | Same CORE feature set | narrower window (selected above) | "2026 votes are shaped mostly by the very newest voting norms" |
| C — 2026 stats-assisted | CORE features + ADVANCED footywire stats (score involvements, metres gained, effective disposals, disposal efficiency, centre/stoppage clearances, intercepts-proxy, turnovers, tackles inside 50) | recent-8 seasons on ADVANCED (2018-2025) | "objective, umpire-visible-style stats carry more weight in 2026" |
| A_with_reputation | Scenario A features + lagged prior-vote-rate ("reputation") | recent-8 seasons | comparison arm only, not part of the default ensemble |

Scenario C's feature set is the best available proxy for "the statistics umpires are confirmed to
see" given what is actually buildable from public data for 2026 — see `docs/2026_STRUCTURAL_BREAK.md`
for the honest accounting of which of the 17 confirmed umpire stats this can and cannot cover (14 of
17 covered directly or as a documented proxy; kick-ins, intercept marks and spoils are not available
for 2026 without live-scraping the AFL's semi-public API, which is out of scope for this run).

## 6. Structural-break sensitivity (Scenario D) and the final ensemble

See `docs/2026_STRUCTURAL_BREAK.md` for the full rationale, the exact blending mechanism, and why it
is a documented judgement call rather than a fitted parameter.

## 6b. A genuine methodological finding: "season-to-date" features cannot exist for an unrevealed season

Two real, non-obvious issues surfaced while building the 2026 scenarios, both stemming from the same
underlying fact: **Brownlow votes are revealed once, after the whole season, never match-by-match** --
so any feature computed from "this player's votes earlier in the CURRENT season" is not knowable in
real time, for ANY season, not just 2026.

1. **Reputation features** (`brownlow_votes_prev5_mean`, `brownlow_votes_season_to_date_mean`) are
   built from rolling/expanding windows over the player's OWN vote history. Because 2026 votes are
   entirely null (correctly -- they are genuinely unrevealed), any window that comes to include even
   one 2026 match inherits NaN and stays NaN for the rest of the season: verified, EVERY 2026 row had
   NaN reputation features, and Scenario A_with_reputation's prediction step crashed with zero usable
   rows. **Fix**: for 2026 rows only, each player's reputation features are frozen at their last known
   REAL (pre-2026) value via a per-player forward-fill (`train_2026_scenarios.py::_freeze_reputation_for_2026`)
   -- pre-2026 training rows are left completely untouched (the fill is written back only where
   `season == 2026`), so Phase 4's validated training values are unaffected. This changes the 2026
   interpretation of "season-to-date" from "votes so far this season" (genuinely unknowable) to "voting
   rate as of the end of the player's last completed season" -- an honest, disclosed proxy.
2. **`disposals_season_to_date_mean` and its 3 CORE lagged-form siblings** (part of the default,
   non-reputation feature set) have the SAME structural property for ANY season's Round 1: there is no
   prior within-season game to average, so every player's round-1 row is NaN in these columns --
   **this is a pre-existing Phase 4 characteristic, present in every historical backtest too**, not a
   2026-specific bug, and was deliberately NOT patched (patching it would create an inconsistency
   between the validated training methodology and 2026 inference). Its 2026 consequence: **6 of 207
   matches (the 5 fixtures played in an actual Round 1, plus 1 Round 2 match for the 2 teams that had
   a Round 1 bye) cannot be scored by Scenario A/B and are excluded from every probability table and
   the leaderboard.** See `docs/2026_DATA_VALIDATION.md` §named exclusions and
   `reports/2026_excluded_matches.csv`. This causes a small, systematic, disclosed UNDER-count of
   season totals for any player who happened to have a big game in their team's actual first fixture.

## 7. Reputation down-weighting for 2026

Section 4 of the brief hypothesises that objective post-match statistics may reduce reliance on
memory/reputation in umpire deliberation. Scenario A_with_reputation is trained and compared against
Scenario A on 2026 predictions (`reports/2026_scenario_comparison.csv`, `reputation_effect` column) —
the DEFAULT 2026 ensemble uses Scenario A/B/C, none of which include the reputation feature, per this
hypothesis. This is stated as a hypothesis-consistent modelling choice, not a proven fact: the
reputation effect could equally persist unchanged in 2026 if umpires' use of the new statistics turns
out to be superficial. The comparison table lets a reader see exactly which contenders are most
affected by this choice.
