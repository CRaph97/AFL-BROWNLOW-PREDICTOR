# 2026 Brownlow Contender Analysis

Status: **Complete.**
Last updated: 2026-09-17
Full machine-readable detail: `reports/2026_leaderboard.csv` (569 players),
`reports/2026_round_by_round_contenders.csv` (top 20 contenders, every game),
`reports/2026_match_explanations.csv` (202 material games, P(3)>=0.30).

All figures are FINAL_ENSEMBLE (the documented 0.45/0.20/0.35 A/B/C blend) unless stated otherwise.
Ranges are from the 100,000-simulation Monte Carlo run (`reports/2026_simulation_summary.csv`).

## Top 20 leaderboard

| Rank | Player | Team | Expected votes | 80% range | 95% range | 3v/2v/1v games | Struct. sens. | Disagreement |
|---|---|---|---|---|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 44.9 | 41-49 | 39-50 | 16/2/0 | 0.04 | 0.91 |
| 2 | Bailey Smith | Geelong | 36.2 | 32-41 | 29-43 | 7/8/2 | 0.35 | 0.53 |
| 3 | Marcus Bontempelli | Western Bulldogs | 26.2 | 21-30 | 19-32 | 6/4/2 | 1.40 | 1.40 |
| 4 | Patrick Cripps | Carlton | 25.5 | 20-30 | 18-32 | 6/4/1 | 0.19 | 1.44 |
| 5 | Zak Butters | Port Adelaide | 25.5 | 19-28 | 17-30 | 6/4/1 | **4.49** | **6.53** |
| 6 | Will Ashcroft | Brisbane Lions | 24.7 | 21-29 | 19-31 | 5/5/2 | 0.59 | 0.91 |
| 7 | Harry Sheezel | North Melbourne | 24.0 | 19-28 | 16-31 | 6/3/6 | 2.30 | 3.21 |
| 8 | Isaac Heeney | Sydney | 23.4 | 19-27 | 17-29 | 6/2/4 | 1.33 | 1.33 |
| 9 | Lachie Neale | Brisbane Lions | 22.4 | 18-26 | 16-29 | 4/6/2 | 0.02 | 0.18 |
| 10 | Izak Rankine | Adelaide | 20.9 | 17-24 | 16-26 | 6/3/1 | 0.42 | 0.42 |
| 11 | Jordan Dawson | Adelaide | 20.6 | 17-25 | 15-27 | 4/3/4 | 0.65 | 0.65 |
| 12 | Kysaiah Pickett | Melbourne | 19.0 | 16-22 | 14-24 | 5/2/0 | 0.04 | 0.08 |
| 13 | Jai Newcombe | Hawthorn | 18.6 | 14-23 | 12-26 | 5/3/3 | 0.10 | 0.18 |
| 14 | Max Gawn | Melbourne | 17.8 | 13-23 | 11-25 | 3/5/4 | 0.60 | 3.09 |
| 15 | Ed Richards | Western Bulldogs | 15.9 | 12-20 | 10-23 | 3/4/2 | 0.01 | 0.42 |
| 16 | Max Holmes | Geelong | 14.8 | 11-19 | 9-21 | 4/2/3 | 0.22 | 0.26 |
| 17 | Chad Warner | Sydney | 14.7 | 11-19 | 9-22 | 1/5/4 | 1.02 | 1.02 |
| 18 | Shai Bolton | Fremantle | 13.7 | 10-18 | 8-20 | 3/1/4 | 0.32 | 0.93 |
| 19 | Caleb Serong | Fremantle | 13.6 | 9-18 | 7-20 | 3/1/5 | 0.20 | 0.75 |
| 20 | Errol Gulden | Sydney | 13.5 | 11-15 | 10-17 | 5/0/0 | 0.03 | 0.09 |

## Contender-by-contender notes (top ~10)

**#1 Nick Daicos (Collingwood)** — the clearest, most stable projection in the field: virtually zero
scenario disagreement (0.04 structural-break sensitivity, 0.91 model disagreement — smallest of any
top-10 player) and a Monte Carlo 95% range of 39-50, the tightest relative spread of any contender.
11 games are HIGH-CONFIDENCE 3-vote games (P(3) >= 0.5, e.g. round 17: 37 disposals, 11 contested
possessions, 7 clearances, 3 goals, P(3)=0.97). Two rounds (10, 25) are confidently projected at 0
votes despite his overall dominance — every projection has real, disclosed downside games, not just
upside.

**#2 Bailey Smith (Geelong)** — second on raw expected votes, but shows the largest positive
reputation effect of any top-10 player (+2.84 votes with reputation included vs without) — worth
knowing this projection leans partly on the reputation feature's own caveat (docs/REPUTATION_EXPERIMENT.md):
some of the model's confidence here could reflect prior-season polling recognition, not purely 2026
form.

**#3 Marcus Bontempelli (Western Bulldogs)** — the largest positive reputation effect in the entire
top-10 tier is not his (+3.03, largest overall in fact) — same caveat as above, doubly so here.

**#5 Zak Butters (Port Adelaide)** — **by far the single most scenario-sensitive top-10 player**:
structural_break_sensitivity 4.49 and model_disagreement_range 6.53, both roughly 3-10x any other
top-10 contender. Scenario A (historical) has him at 19.5 EV; Scenario C (stats-assisted) has him at
24.0 — a genuinely different picture depending on whether 2026's rule change meaningfully favours his
statistical profile. **This is the single player in the top 10 whose rank most depends on an
assumption this project cannot verify** (see docs/2026_STRUCTURAL_BREAK.md).

**#7 Harry Sheezel (North Melbourne)** — second-most scenario-sensitive top-10 player (struct. sens.
2.30). Notably has the most 1-vote games of any top-10 player (6) alongside 6 three-vote games — a
"gets on the board often, dominates less often" profile relative to Daicos or Cripps.

**#14 Max Gawn (Melbourne)** — highest model_disagreement_range (3.09) outside Butters/Sheezel, driven
by Scenario B (recent-era, 15.6) sitting notably below Scenario A/C (18.1/18.7) — the ruckman-specific
signal appears more sensitive to which training window is used than for most outfield contenders.

## Section 12 categories (from `reports/2026_round_by_round_contenders.csv`, top 20 contenders)

- **HIGH-CONFIDENCE 3-vote games** (most_likely=3, confidence=HIGH, i.e. P(3) >= 0.5): 11 for Daicos,
  7 for Smith, 6 for Rankine — full list in the CSV.
- **BORDERLINE POLLING GAMES** (0.30 <= P(3) <= 0.55, genuinely uncertain who takes the 3): 37 such
  games among the top 20 alone, e.g. Bontempelli round 2 (P3=0.54 vs P2=0.42 — almost a coin flip),
  Daicos round 8 (P3=0.46 vs P2=0.50 — the model actually leans 2-votes that round despite Daicos being
  the projected season leader).
- **SIGNIFICANT GAMES WHERE THE MODEL EXPECTS NO VOTES**: 210 HIGH-confidence zero-vote games across
  the top 20 contenders combined (average ~10-11 per player across a 23-round season) — every
  contender, including the projected #1, has confidently-projected quiet games. Shai Bolton (16),
  Ed Richards and Kysaiah Pickett (15 each) have the most among the top 20.
- **MOST IMPORTANT ROUNDS DRIVING THE SEASON TOTAL**: for Daicos specifically, rounds 12, 16, 17, 21,
  23 each carry P(3) > 0.85 and are collectively worth ~14-15 of his ~45 projected votes — roughly a
  third of his season total concentrated in 5 of 23 games.

## Known data-completeness caveat affecting the disagreement metric

3 of 569 players (Nasiah Wanganeen-Milera, Jason Horne-Francis, Luke Davies-Uniacke) have
`C_stats_assisted = 0.0` because their footywire (ADVANCED) join failed for their entire 2026 season —
a known identity-resolution limitation for hyphenated surnames (`docs/DATA_COVERAGE.md`), not a real
model disagreement. This inflates their apparent `model_disagreement_range`; none of the three rank in
the top 20 (their FINAL_ENSEMBLE score, appropriately, is 0 since only 2 of 3 scenarios contributed —
they should be read as "not covered by the ensemble" rather than "zero votes", and are flagged here
rather than silently left in a ranked table). The same artifact appears in `reputation_effect` for a
handful of 2026 debutants with no pre-2026 vote history at all (e.g. "Jagga Smith" shows an apparent
-5.5 effect, which is really "no reputation feature exists for this player" rendered as 0, not a
genuine finding that reputation hurts them).
