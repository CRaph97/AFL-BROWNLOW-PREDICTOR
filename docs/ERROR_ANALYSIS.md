# Error Analysis — Phase 4, Section N

Status: **Complete.**
Last updated: 2026-09-17

## Method (and a bug fixed along the way)

Uses the pooled Model 1 (Plackett-Luce, expanding window) out-of-sample predictions. The first version
of this analysis segmented by PLAYER-level attributes (role, disposals≥25) by filtering the prediction
table before recomputing match-level metrics — this silently breaks match completeness (a match doesn't
have one "role", its 22-44 players do) and crashed outright when a segment produced zero valid matches.

**Fixed approach**: match-level correctness (was the 3-vote pick correct? the full 3-2-1?) is computed
ONCE on the full, unfiltered match, then attached back onto the row of the ACTUAL 3-vote winner and
segmented by THAT row's attributes. This answers a well-posed question: "when the actual 3-vote winner
had attribute X, how often did the model correctly predict it was them."

## Results — role segment

| Role | n (times this role actually won 3 votes) | Accuracy predicting them |
|---|---|---|
| MIDFIELDER | 1,276 | 0.620 |
| UNKNOWN | 109 | 0.642 |
| KEY_FORWARD | 156 | 0.564 |
| MIDFIELDER_FORWARD | 43 | 0.558 |
| RUCK | 95 | 0.505 |
| MEDIUM_FORWARD | 123 | 0.398 |
| **MEDIUM_DEFENDER** | 124 | **0.339** |
| **KEY_DEFENDER** | 32 | **0.125** |

## Results — margin bucket segment

| Absolute margin | n | Accuracy predicting the 3-vote winner |
|---|---|---|
| 0-12 (very close) | 487 | 0.538 |
| 12-24 | 396 | 0.591 |
| 24-48 | 601 | 0.567 |
| 48-100 | 423 | 0.572 |
| 100+ (blowout) | 34 | **0.794** |

## Interpretation

**The model's single biggest, most consistent blind spot is key and medium defenders.** When a key
defender actually wins the 3 votes (a rare event — only 32 of 1,958 matches — but a real one), the
model correctly picks them only 12.5% of the time, far below every other role and less than a third of
the midfielder rate (62.0%). Medium defenders fare only somewhat better (33.9%).

This is genuinely coherent with — not contradicted by — the stability finding that `role_KEY_DEFENDER`
has the single largest positive coefficient in the model (`docs/PHASE4_DECISIONS.md`): the model *does*
give defenders a real, learned boost relative to their raw stat line, but that boost is usually not
enough to overcome key defenders' typically modest showing on the disposal/goal-heavy features that
dominate the score. In other words: the model has correctly learned that "being a key defender is worth
extra" but the *size* of that extra is not enough to fully close the gap on the rare occasions a
defender genuinely does have the best game on the ground. This is exactly the kind of real-world
Brownlow bias (defenders historically underrewarded — see Phase 3's `docs/ROLE_ANALYSIS.md`) that the
model has partially, but not fully, learned to compensate for.

**Blowouts are the model's easiest cases.** Accuracy in 100+-point-margin matches (79.4%) is far above
any other margin band (all clustered 53.8-59.1%) — makes clear football sense: in a genuine blowout, the
best player is usually unambiguous (little competing signal from the losing team, per the strong
winner-effect finding across this whole project), while genuinely close, competitive games are hardest
to call, which is itself a sensible, expected pattern rather than a flaw.

## Other outputs

- `reports/surprising_3vote_misses.csv` (254 rows) — every actual 3-vote winner the model ranked outside
  its own match's top 3, for qualitative follow-up.
- `reports/player_level_bias.csv` (931 players, min. 20 OOS matches) — pooled expected-vs-actual bias
  per player, usable to check whether specific well-known players are systematically over/under-served
  by the model (a natural Phase 5 follow-up, not pursued further here per the brief's caution against
  adjusting weights from anecdotes).

Per the brief's explicit instruction, none of these findings have been used to hand-adjust feature
weights — they are recorded here as hypotheses for structured testing in a later phase (e.g., an
explicit role × margin interaction term, or a defender-specific feature such as a proxy for intercepts/
spoils once Phase 2's 2026-stats-mirror gap is closed).
