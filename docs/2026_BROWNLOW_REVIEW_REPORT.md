# 2026 Brownlow Review Report

Status: **Complete.** Companion to the interactive dashboard (`app.py`). Last updated: 2026-09-17.

This is a human-readable synthesis, not a data dump -- every number below traces to the
frozen, audited production files listed in section 12. Full detail (all 207 matches, every
player, every round) lives in the dashboard and the underlying CSVs; this report highlights
what matters, not what's already tabulated elsewhere.

## 1. Executive Summary

The 2026 forecast blends three real scenario models -- historical voting behaviour, a
recent-era-only model, and a model centred on the statistics umpires are confirmed to see
under the 2026 rule change -- into a single documented, non-uniform ensemble (0.45 / 0.20 /
0.35 respectively). All 207 home-and-away matches are scored (fixed from an initial 201/207
in a post-audit correction; see section 12). Nick Daicos is the clear #1 projection (47.2
expected votes) and it survived a dedicated over-concentration audit. The single most
scenario-sensitive player in the field is Zak Butters, whose projection swings materially
depending on how much weight the 2026 rule change is assumed to carry.

## 2. Final Top 20

| Rank | Player | Team | EV | Median | 80% Range | 95% Range | 3s | Disagreement | Struct-Break |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 47.2 | 47 | [43, 51] | [41, 53] | 17 | 0.92 | 0.23 |
| 2 | Bailey Smith | Geelong | 35.5 | 35 | [31, 40] | [28, 43] | 7 | 0.58 | 0.27 |
| 3 | Marcus Bontempelli | W. Bulldogs | 26.7 | 26 | [22, 31] | [20, 33] | 6 | 1.58 | 1.58 |
| 4 | Harry Sheezel | North Melbourne | 25.9 | 25 | [20, 30] | [18, 33] | 7 | 3.30 | 2.26 |
| 5 | Zak Butters | Port Adelaide | 25.4 | 23 | [19, 28] | [17, 30] | 6 | 6.43 | 4.50 |
| 6 | Patrick Cripps | Carlton | 25.2 | 24 | [20, 29] | [17, 32] | 6 | 1.42 | 0.11 |
| 7 | Will Ashcroft | Brisbane Lions | 24.9 | 25 | [21, 29] | [19, 31] | 5 | 0.96 | 0.62 |
| 8 | Isaac Heeney | Sydney | 24.0 | 24 | [19, 28] | [17, 30] | 6 | 1.46 | 1.36 |
| 9 | Lachie Neale | Brisbane Lions | 23.6 | 23 | [19, 28] | [17, 30] | 4 | 0.64 | 0.42 |
| 10 | Izak Rankine | Adelaide | 20.9 | 21 | [17, 24] | [16, 26] | 6 | 0.42 | 0.42 |
| 11 | Jordan Dawson | Adelaide | 20.7 | 21 | [17, 25] | [15, 27] | 4 | 0.65 | 0.65 |
| 12 | Max Gawn | Melbourne | 19.0 | 19 | [14, 24] | [12, 27] | 3 | 3.66 | 0.76 |
| 13 | Kysaiah Pickett | Melbourne | 19.0 | 19 | [16, 22] | [14, 24] | 5 | 0.08 | 0.04 |
| 14 | Jai Newcombe | Hawthorn | 18.6 | 19 | [14, 23] | [12, 26] | 5 | 0.18 | 0.11 |
| 15 | Ed Richards | W. Bulldogs | 18.4 | 18 | [14, 23] | [12, 25] | 4 | 0.43 | 0.18 |
| 16 | Chad Warner | Sydney | 15.0 | 15 | [11, 19] | [9, 22] | 1 | 0.96 | 0.96 |
| 17 | Caleb Serong | Fremantle | 14.5 | 14 | [10, 19] | [8, 22] | 3 | 0.71 | 0.45 |
| 18 | Clayton Oliver | GWS | 14.5 | 14 | [10, 20] | [7, 23] | 1 | 1.54 | 0.34 |
| 19 | Errol Gulden | Sydney | 14.4 | 14 | [12, 17] | [11, 18] | 5 | 0.07 | 0.06 |
| 20 | Max Holmes | Geelong | 14.2 | 14 | [10, 18] | [9, 20] | 4 | 0.16 | 0.15 |

(Full field, all scenario columns: `reports/2026_leaderboard.csv`.)

## 3. Major Contenders

**Nick Daicos** is not a single-scenario artefact -- his historical (46.8), recent-era (47.8)
and stats-assisted (47.1) EVs all agree within ~1 vote, and the audit (section 5 below) found
broad accumulation across 19 of 22 matches rather than a couple of outlier probabilities.
**Bailey Smith** is the clear #2 with the same low-disagreement profile. From rank 3 onward the
field tightens and scenario disagreement starts to matter more than rank order: Sheezel,
Butters, Cripps, Ashcroft, Heeney and Neale are separated by only ~5 votes of expected value
but differ sharply in how much of that value depends on the untested 2026 rule change.

## 4. Round-by-Round: Key Polling Games

**Highest-confidence predicted 3-vote games** (P(3) from `reports/2026_predicted_votes.csv`):

| Round | Player | Team | P(3) |
|---|---|---|---|
| 25 | Izak Rankine | Adelaide | 99% |
| 23 | Matt Rowell | Gold Coast | 98% |
| 8 | Kysaiah Pickett | Melbourne | 98% |
| 8 | Charlie Curnow | Sydney | 97% |
| 17 | Nick Daicos | Collingwood | 97% |
| 17 | Zak Butters | Port Adelaide | 97% |
| 25 | Errol Gulden | Sydney | 96% |
| 11 | Harley Reid | West Coast | 95% |

**Most uncertain matches** (largest round-level spread across the three real scenarios,
`data/processed/scenario_predictions_2026.parquet`, top-20 contenders only):

| Round | Player | Historical | Recent | Stats-Assisted | Spread |
|---|---|---|---|---|---|
| 6 | Zak Butters | 1.49 | 1.42 | 2.53 | 1.11 |
| 18 | Zak Butters | 0.57 | 0.63 | 1.43 | 0.86 |
| 25 | Max Gawn | 2.11 | 1.37 | 2.07 | 0.75 |
| 4 | Max Gawn | 1.74 | 1.11 | 1.62 | 0.62 |
| 16 | Max Gawn | 1.56 | 1.14 | 1.72 | 0.58 |

## 5. Structural-Break Analysis

Full methodology: `docs/2026_STRUCTURAL_BREAK.md`. In short: the 2026 rule change (umpires now
see approved player-performance statistics when deliberating) cannot be calibrated against any
real 2026 vote yet, so it is handled as an explicit sensitivity axis (Low/Medium/High) rather
than folded silently into one number. **Zak Butters is the field's clearest case study**: his
historical-behaviour EV (19.5) and stats-assisted EV (24.0) diverge by more than 4 votes --
larger than any other top-10 player -- because his statistical output (elite contested
numbers) is stronger relative to his historical polling reputation than most contenders. Max
Gawn shows the same pattern at a smaller scale (rucks are traditionally under-polled relative
to their statistical output).

## 6. Most Model-Sensitive Players

Ranked by `structural_break_sensitivity` (spread across historical/recent/stats-assisted
scenarios): **Zak Butters (4.50)** and **Harry Sheezel (2.26)** are far ahead of the rest of
the top 10; everyone else in the top 10 sits below 1.6. A player whose rank changes materially
between the "Historical" and "Stats-Assisted" columns of `reports/2026_scenario_comparison.csv`
should be treated with more caution than one whose position is stable across all three.

## 7. Largest Model Disagreements

By `model_disagreement_range` (scenario max-min): **Zak Butters (6.43)**, **Max Gawn (3.66)**,
**Harry Sheezel (3.30)**. Three additional high-disagreement rows in
`reports/2026_model_disagreement.csv` (Wanganeen-Milera, Horne-Francis, Davies-Uniacke) are a
**data artefact, not a real model conflict** -- a footywire join gap zeroes out their
stats-assisted EV entirely (documented in `docs/2026_FINAL_REPORT.md`). Treat those three as a
known data gap, not genuine scenario disagreement.

## 8. Defender-Bias Watchlist (Summary)

15 specific 2026 games are flagged where an elite defensive performance (top 5% league-wide by
disposals + 2x contested marks + 0.5x one-percenters) receives a model EV below 0.5 -- e.g.
Archie Roberts (R8, 42 disposals, EV 0.08), Harris Andrews (R20, 22 disposals/5 contested
marks/9 one-percenters, EV 0.27). This reflects a real, quantified, pre-existing model
weakness (key defenders are correctly identified as 3-vote winners only 12.5% of the time vs.
62% for midfielders) and is disclosed, not corrected. Full list: dashboard "Defender Bias
Watchlist" page or `docs/2026_FINAL_AUDIT.md` section 4.

## 9. Most Uncertain Matches

See section 4 above (round-level scenario spread) -- Zak Butters and Max Gawn account for 6 of
the 8 most scenario-sensitive individual matches in the top-20 field.

## 10. Methodology Summary

Four building blocks, each real and separately validated:

1. **Historical model** -- Plackett-Luce ranking model (recent-8-season window), the
   best-validated architecture from the Phase 4 backtest (57.5% correct-3, log loss 0.183
   across 22 walk-forward folds).
2. **Recent-era model** -- same architecture, trained on a shorter, more recent window.
3. **Stats-assisted model** -- centred on the confirmed umpire-visible statistics for 2026.
4. **Structural-break sensitivity bands** (Low/Medium/High) -- explicit, undetermined-weight
   variants, not a point estimate.

These are blended into `FINAL_ENSEMBLE` via a documented, non-uniform probability-space
combination (0.45/0.20/0.35) -- see `docs/2026_MODELLING_METHODOLOGY.md` for the full
derivation and the mid-build bug (a utility-blending error that mechanically flattened
probabilities) that was found and fixed before this was finalised.

## 11. Known Limitations

- **Defender underprediction** carries forward unchanged from Phase 4 (section 8 above) -- a
  real, quantified model weakness, not something this review layer corrects.
- **Small positive season-total bias**: the Phase 4 season-level backtest found the model
  slightly over-predicts total votes in every one of 8 historical test seasons (+0.01 to +0.05
  votes/player) -- worth keeping in mind when reading any single player's EV as a hard ceiling.
- **3 footywire-join players** (Wanganeen-Milera, Horne-Francis, Davies-Uniacke) have a
  zeroed-out stats-assisted scenario due to a data-join gap, not a real scenario disagreement
  -- flagged in section 7.
- **A small number of player-match rows remain excluded** at the individual level (true
  season debutants with no prior AFL history to carry forward) -- this affects specific
  players' rosters within a match, not whole-match coverage (which is 207/207).
- **The 2026 structural-break weighting is fundamentally unverifiable until real 2026
  Brownlow votes are announced.** Every number in this report that involves the
  stats-assisted scenario or the ensemble blend inherits that uncertainty; the dashboard's
  Uncertainty and Scenario Comparison pages are the right place to see how much any given
  player's ranking depends on that untested assumption.

## 12. Source Files

Dashboard: `app.py`, `dashboard/data.py`, `pages/*.py`. Underlying production data (all
frozen, unmodified by this dashboard build): `reports/2026_leaderboard.csv`,
`reports/2026_simulation_summary.csv`, `reports/2026_match_probabilities.csv`,
`reports/2026_predicted_votes.csv`, `reports/2026_scenario_comparison.csv`,
`reports/2026_round_by_round_contenders.csv`, `reports/2026_model_disagreement.csv`,
`reports/2026_daicos_round_by_round.csv`, `reports/2026_top10_probability_audit.csv`,
`reports/2026_quality_checks.json`, `data/processed/model_core_2026.parquet`,
`data/processed/scenario_predictions_2026.parquet`. Prior audit trail:
`docs/2026_FINAL_AUDIT.md`, `docs/2026_FINAL_REPORT.md`, `docs/2026_STRUCTURAL_BREAK.md`,
`docs/2026_CONTENDER_ANALYSIS.md`, `docs/2026_MODELLING_METHODOLOGY.md`,
`docs/2026_DATA_VALIDATION.md`.
