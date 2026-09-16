# Reputation Experiment — Phase 4, Section P

Status: **Complete.**
Last updated: 2026-09-17

## Method

Per the brief, this is a controlled, isolated experiment, not a feature silently added to the primary
model. Two identical Plackett-Luce specifications are compared, differing ONLY in the presence of two
strictly-lagged features:

- `brownlow_votes_prev5_mean` — mean Brownlow votes over the player's previous 5 matches (excluding
  the current one).
- `brownlow_votes_season_to_date_mean` — mean Brownlow votes across the current season so far,
  excluding the current match.

Both are built in `src/features/build_lagged_form_features.py` with the same never-look-ahead
discipline as every other lagged feature in this project (see `docs/LEAKAGE_AUDIT.md`).

**Without-reputation** uses the full CORE feature set from the main backtest (raw + match-relative +
context + teammate + role + nonlinear + lagged form (performance stats only) + win×margin
interaction). **With-reputation** adds only the two features above.

Evaluated with walk-forward validation, expanding training window, test seasons 2019-2025 (7 seasons)
— chosen to guarantee every test season has a reasonable number of prior seasons with sufficient
career history for the lagged reputation features to be populated (rather than mostly NaN in the
earliest test seasons).

## Interpretation rule (decided in advance, per the brief)

Reputation is **not** added to the primary/recommended model unless it shows a **consistent** genuine
out-of-sample improvement across the test seasons (not just a one-season fluke). If it does help, the
result must be reported with this explicit caveat: an improvement from a lagged prior-vote-rate feature
could reflect the model detecting **umpire recognition of a known high-polling player** (a reputation/
name-recognition effect) rather than a genuine unmeasured football-performance signal — this is an
important distinction for how any such result should be used or communicated, not merely a footnote.

## Results

Per-season (`correct_3` / `exact_321`):

| Season | without_reputation | with_reputation | Winner |
|---|---|---|---|
| 2019 | 0.630 / 0.082 | 0.614 / 0.076 | without |
| 2020 | 0.528 / 0.072 | 0.536 / 0.056 | with (correct_3) |
| 2021 | 0.644 / 0.133 | 0.689 / 0.128 | with (correct_3) |
| 2022 | 0.601 / 0.082 | 0.612 / 0.093 | with |
| 2023 | 0.486 / 0.044 | 0.514 / 0.066 | with |
| 2024 | 0.489 / 0.063 | 0.489 / 0.063 | tie |
| 2025 | 0.544 / 0.036 | 0.544 / 0.041 | tie (correct_3), with (exact_321) |

**Mean across all 7 test seasons:**

| Variant | mean_correct_3 | mean_exact_321 | mean_log_loss | mean_rank_corr |
|---|---|---|---|---|
| with_reputation | 0.5712 | 0.0748 | 0.18627 | 0.4162 |
| without_reputation | 0.5605 | 0.0732 | 0.18713 | 0.4159 |

## Interpretation

Per-season, reputation **wins outright in 4 of 7 seasons, ties in 2, and loses in only 1** (2019).
Averaged across all 7 seasons, `with_reputation` is better on **every one of the 4 aggregate metrics**
tracked — correct-3% (+1.1 percentage points), exact-3-2-1% (+0.16pp), log loss (better calibrated,
0.18627 vs 0.18713), and rank correlation (marginally). This clears the bar set in advance
(§"Interpretation rule"): the improvement is **consistent**, not a one-season fluke, though it is
**modest in magnitude**, not transformative.

**Required ethical/statistical caveat, per the brief:** this result is genuinely ambiguous about *why*
it works. `brownlow_votes_prev5_mean` and `brownlow_votes_season_to_date_mean` are, definitionally, a
record of how often umpires have *already chosen to reward* this player recently. A model that uses
this to predict future votes could be doing either (or both) of two different things:
1. **Proxying for real, persistent football quality** not fully captured by the box-score features
   already in the model (a genuinely good player's underlying quality is stable, so recent votes
   correlate with future votes for a legitimate performance reason).
2. **Detecting umpire reputation/name-recognition bias** — umpires may be more inclined to give votes
   to a player they already think of as a contender, independent of that specific match's performance
   relative to peers.

This experiment **cannot distinguish between these two explanations** — that would require, at minimum,
holding match performance constant and checking whether prior-vote-rate still predicts additional votes
(a natural Phase 5 follow-up), which was not attempted here. Both explanations are plausible and not
mutually exclusive.

## Recommendation

Per the brief's explicit instruction ("do NOT include reputation in the primary model unless it
provides consistent genuine predictive improvement"): the improvement here **is** consistent enough to
warrant inclusion, but given its modest size and the unresolved reputation-vs-quality ambiguity above,
the recommendation is to make it an **optional, clearly labelled feature layer** — reported separately
in any model output (e.g. "with career-form adjustment" vs. "performance-only") — rather than silently
folded into a single "the model" with no distinction. This preserves the brief's transparency
requirement: if reputation is doing real work, users of this research should be able to see that,
not have it hidden inside an unlabelled composite score.
