# 2026 Brownlow — post-event model and betting evaluation

Scores the **frozen, pre-count** 2026 outputs of the Production model, the Objective model and
Wheelo against the AFL's actual 3-2-1 votes, and settles the **pre-count** bookmaker markets.
Nothing was retrained, no simulation was rerun, no forecast was blended, and no post-result price
is used anywhere. The dashboard page is **2026 Evaluation** (MAIN section, after Guide & FAQs).

## Inputs (all read-only)

| Input | File | Captured / built | Role |
|---|---|---|---|
| Actual votes | `data/actual/2026_brownlow_match_votes.csv`, `…_leaderboard.csv` | 2026-09-23 from the AFL tracker, post-count | Ground truth |
| Production | `reports/2026_leaderboard.csv`, `2026_predicted_votes.csv`, `2026_match_probabilities.csv`, `2026_simulation_summary.csv`, `data/processed/mc_totals_2026.npy` | Phase 5, pre-count | EV, ranks, P3/P2/P1, 100k joint draws |
| Objective | `reports/2026_objective_leaderboard.csv`, `2026_objective_votes.csv`, `2026_objective_simulation_summary.csv`, `data/processed/mc_totals_objective_2026.npy` | pre-count | EV, ranks, P3/P2/P1, 20k joint draws |
| Wheelo | `data/external/processed/wheelo_season.csv`, `wheelo_match_level.csv` | scraped 2026-09-18 | Season EV/rank, match EV, match P3 %, match rank |
| Bookmakers | `data/betting/processed/priced_opportunities.csv`, `refresh_summary.json` | Neds + PointsBet, 2026-09-18 06:15 UTC | Odds, model probabilities, Bet Value / Wheelo labels |
| Roles | `data/processed/player_match_role_lagged_2026.parquet` | Phase 4 lagged proxy | Role / position groups |

The count was held on the evening of 2026-09-21 (AEST). Every bookmaker snapshot predates it by
three days; `manifest.json` records the timestamps and `all_bookmaker_snapshots_pre_count`.
Every frozen input is SHA-256 hashed before and after the build (`frozen_inputs_unchanged_during_build`)
and `tests/test_2026_evaluation.py` re-hashes them.

## Code and outputs

- `src/evaluation/settlement.py` — pure settlement / flat-stake rules (unit-tested).
- `src/evaluation/build_2026_evaluation.py` — builds every table below into `data/evaluation/2026/`.
- `dashboard/evaluation.py` + `pages/33_2026_Evaluation.py` — presentation only.

| Output | Content |
|---|---|
| `season_players.csv` | One row per player in any source (664): actual votes/rank/eligibility, modal role, each source's EV / rank / error, best & worst source, cross-model spread. |
| `scorecard.csv` | Season-level metrics per model, on each model's own coverage and on the common universe (576 players scored by all three). |
| `match_table.csv`, `match_scorecard.csv`, `round_table.csv` | 207 matches × 3 models: top-P3 pick, hit, 3-voter's P3 rank, exact 3-2-1, unordered top 3, EV error, P3 log loss / Brier. |
| `disagreement_summary.csv`, `disagreement_cases.csv` | Pre-event Prod-vs-Obj EV gap bins vs which model was closer; Wheelo tie-breaks; rank reversals. |
| `season_bias.csv`, `match_bias.csv` | Error by role and team per model; 3-voter hit rate by the 3-voter's role. |
| `final_order_top20.csv`, `simulation_coverage.json` | Actual top 20 vs each model, simulation P(win)/P(top-N), exact-order and set probabilities from the persisted draws. |
| `betting_settled.csv`, `betting_summary.csv`, `team_leaders.csv` | Every priced selection settled; summaries by market, bookmaker, Bet Value, Likelihood, Wheelo support, agreement, probability and edge bands. |
| `calibration_bins.csv`, `calibration_summary.csv` | 10-pp reliability bins with ECE and Brier per series. |
| `learnings.json` | Deterministic findings generated from the tables above. |
| `manifest.json` | Timestamps, counts, hashes, settlement counts. |

## Methodology

**Universes.** Production scores 578 players (its documented season-to-date exclusion drops a player's
first game and ~167 fringe players entirely), Objective 663 (after excluding `NOID2026_*` placeholder
rows) and Wheelo 661. Season metrics are reported on each source's own coverage *and* on the common
576-player universe; the common universe is the fair comparison. Players not on the AFL leaderboard
have 0 actual votes. Actual rank uses the min method over all players **including ineligible**
players, as the AFL leaderboard and the bookmakers' "Includes Ineligible" markets do; an eligible-only
rank is carried alongside.

**Season metrics.** Error = EV − actual. MAE, RMSE, mean bias, Spearman/Kendall, Top-N hit rate
(actual top-N including ties, so the actual top 5 has 7 members), winner prediction and rank error,
mean |rank error| over the actual top 30, plus a "relevant" MAE over players with actual ≥ 1 or EV ≥ 1.

**Match metrics.** For each match and model: the highest-P3 player (Wheelo: highest published P3 %);
hit if it is the actual 3-voter; the actual 3-voter's rank by P3; exact 3-2-1 using each model's own
vote assignment (Production `predicted_votes`, Objective `objective_pred_votes`, Wheelo match rank
1/2/3); unordered top-3 set match; EV error per roster player; P3 log loss and multi-class Brier only
when the 3-voter is in the roster. Wheelo publishes no P2/P1, so those series exist only for the two
internal models. One roster miss exists: Jagga Smith (Carlton, Round 1, his first game) is outside
Production's roster; log loss is not fabricated for that match.

**Disagreement.** |Production EV − Objective EV| binned 0-2 / 2-5 / 5+ votes; per bin, which model was
closer, both MAEs, the better and worse model's error, and the mean-of-two error. Wheelo tie-break:
where Wheelo's EV is nearer one internal model, was that model the closer one? Rank reversals: the
same players' rank gap of 10+.

**Bias.** Season error by modal lagged role and by team, per model, over all covered players and over
the relevant subset; over/under counted at ±0.5 votes; groups under 10 flagged. Match-level defender
test: 3-voter hit rate by the 3-voter's role and the role mix of each model's top picks.

**Simulations.** Read from the persisted summaries and joint draws; nothing rerun. Reports P(winner
finishes 1st), coverage of the actual top 3/5/10 by simulated mean rank, exact-order probability of
the actual top 3 and "all in top-k" set probabilities for k = 3 and 4 (a three-way tie at 5th makes an
exact top-5 order undefined), and whether any persisted order scenario carried the actual top-3 prefix.

**Betting settlement** (`src/evaluation/settlement.py`). Winner / Top-N / exact position by rank with
ties resolved as dead heats at the standard fraction (places remaining ÷ runners tied); O/U with
integer-line pushes; X+ and To-Poll-a-Vote as thresholds; H2H ties as pushes; team O/U on the sum of
every vote awarded to the team's players. Bookmaker-specific captured odds only; a selection with no
odds or with a pre-count-unresolved identity is "unsettleable", never inferred. 504 unmodelled
selections (Leader After Round N, exotics, group markets) had no pre-count probability and are
excluded; 14 selections for Chad Warner / Charlie Cameron were flagged `IDENTITY_AMBIGUOUS` pre-count
with no probability and are also excluded. Rows whose result would differ on eligible-only ranks are
flagged `settlement_sensitive_to_eligibility`. Flat 1-unit staking is a retrospective analytical
metric: win = odds − 1, loss = −1, push = 0, dead heat = odds × fraction − 1; hit rate excludes pushes;
ROI = P/L ÷ selections. Model accuracy (hit rate vs probability) is shown separately from ROI, and
no claim of future profitability is made. No team-top-poller price was captured, so the team-leader
table is model accuracy only.

**Calibration.** 10-pp bins, ECE (bin-weighted |actual − predicted|) and Brier for: match P3/P2/P1 and
any-vote (Production, Objective; Wheelo P3 only), every settled binary market (Production, Objective and
bookmaker implied), and season P(top 3)/P(top 10) from the simulations. Dead heats enter as fractional
wins; pushes are dropped. Denominators are in every row.

## Headline results

| | Production | Objective | Wheelo |
|---|---|---|---|
| Season MAE (common, n=576) | 0.757 | 0.986 | 0.707 |
| RMSE | 1.482 | 1.953 | 1.484 |
| Spearman ρ | 0.755 | 0.726 | 0.765 |
| Predicted leader | Daicos ✓ | Daicos ✓ | Daicos ✓ |
| Top-10 hit rate | 80% | 60% | 60% |
| Named the 3-voter (207 matches) | 70.0% | 50.2% | 72.5% |
| 3-voter inside top-3 by P3 | 94.2% | 83.1% | 92.8% |
| Exact 3-2-1 | 18.4% | 5.8% | 18.4% |
| Mean P3 log loss | 0.898 | 1.460 | 0.864 |
| P3 ECE | 0.005 | 0.005 | 0.004 |

Simulations: Production P(Daicos wins) 99.1%, exact top-3 order 31.0%, top-4 set 1.0%; Objective 88.9%,
16.1%, 2.7%. Production placed 8 of the actual top 10 inside its simulated top 10, Objective 7.

Betting (653 settled selections, flat 1u): backing everything lost −22.7% ROI as expected. By market:
X+ votes +25.7% (n=107), exact position +81% (n=16, driven by one winner: Gawn to finish 4th at 29 — small sample), To Poll a Vote
−22.4% (n=129), Top-N −47.0% (n=180), H2H −10.3% (n=48), player O/U −6.5% (n=38), team O/U −6.7% (n=70),
winner −96.9% (n=65). Signals: "both models see value" +8.4% (n=104) vs "neither" −46.7% (n=296);
Moderate Bet Value +9.7% (n=80), Strong + Wheelo +13.5% (n=22), Speculative −9.0% (n=119); Likelihood
bands were monotone in hit rate (Very High 83% → Very Low 21%). Wheelo "strong support" did **not**
help (−20.7%, n=117) while "partial support" did (+8.1%, n=136).

## Unresolved / caveats

- Jack Ross (Richmond) has no canonical id; his actual votes are excluded from player-level tables
  (match-level rows keep him by AFL id). See `data/actual/2026_brownlow_validation.json`.
- 111 Wheelo match-level rows are unresolved or ambiguous; they simply do not join.
- Defender and per-role samples are small (14 medium-defender 3-voters, fewer key defenders).
- 25 bookmaker rows are eligibility-sensitive (Top-N/exact-position results that would change on an
  eligible-only ranking); they are settled on the all-player ranking and flagged.
- Combination bets had no captured combined price and are not evaluated.

## Re-running

```
python -m src.evaluation.build_2026_evaluation
python -m pytest tests/test_2026_evaluation.py -q
```
