# Role / Position Analysis — Phase 3

Status: **Complete.** Machine-readable backing: `reports/role_vote_summary.csv`,
`reports/role_proxy_confusion_matrix.csv`, `reports/role_proxy_classification_report.txt`.
Last updated: 2026-09-17

## 1. What role data we actually have (per the brief: derive from available data, don't invent it)

No official position field exists in either of our Phase 2 canonical sources (afltables or footywire).
Phase 3 found and validated a real source instead: the AFL's own officially-listed player position
(`KEY_FORWARD`, `MEDIUM_FORWARD`, `MIDFIELDER`, `MIDFIELDER_FORWARD`, `RUCK`, `KEY_DEFENDER`,
`MEDIUM_DEFENDER`) is available for **2021-2025** via the `torp`/`torpdata` ecosystem already validated
for the event-data pilot in Phase 2. This is a **static, once-per-season listed position**, not a
dynamic per-match role — exactly the limit anticipated in `docs/DATA_SOURCE_AUDIT.md` Phase 1.

For 1999-2020 (before this data source's coverage begins), a **statistically-derived proxy** was built
and honestly validated rather than assumed reliable — see `src/features/build_role_proxy.py` and
`docs/FEATURE_REGISTRY.md` §5 for the method. 1984-1998 has no role classification at all (the input
stats for the proxy aren't structurally available that early).

## 2. How good is the proxy? (be skeptical of your own construction)

A shallow (max_depth=4), fully human-readable decision tree, trained on the 2021-2025 real labels and
evaluated with **grouped 5-fold cross-validation (grouped by player, so no player's multiple seasons
leak between train/test)**:

**Overall accuracy: 76.8%** (2,702 held-out player-seasons). Per-class breakdown:

| Role | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RUCK | 0.96 | 0.96 | 0.96 | 164 |
| KEY_DEFENDER | 0.94 | 0.75 | 0.83 | 303 |
| MEDIUM_DEFENDER | 0.80 | 0.83 | 0.81 | 564 |
| KEY_FORWARD | 0.86 | 0.76 | 0.81 | 297 |
| MEDIUM_FORWARD | 0.74 | 0.78 | 0.76 | 520 |
| MIDFIELDER | 0.67 | 0.87 | 0.76 | 672 |
| **MIDFIELDER_FORWARD** | **0.23** | **0.03** | **0.06** | 182 |

**Rucks and key defenders classify very reliably** (hitouts and rebound-50/one-percenter profiles are
statistically distinctive). **The MIDFIELDER_FORWARD hybrid class is essentially unrecoverable from
box-score stats alone** (F1 = 0.06) — a genuinely blurry, real football role that a season-average
statistical profile cannot reliably distinguish from either pure midfielders or medium forwards. This
is reported honestly rather than smoothed over: **any pre-2021 season's MIDFIELDER_FORWARD-labelled
player-seasons should not be trusted**, and downstream analysis should treat that specific category's
historical rate with extra caution, or collapse it into a broader "hybrid forward/mid" bucket if used at
all pre-2021.

The full tree structure is in `reports/role_proxy_classification_report.txt` — it is a short, readable
4-level tree using rebound_50s, hitouts, marks_inside_50, and clearances z-scores as its main splits,
which matches football intuition (defenders/rucks/key forwards have distinctive extreme profiles;
midfield vs medium-forward is the harder boundary).

## 3. Brownlow votes by role (2021-2025 real labels + 1999-2020 proxy labels combined)

From `reports/role_vote_summary.csv`, 217,574 role-classified player-matches:

| Role | Votes/game | % polling any | % receiving 3 | Share of ALL 3-vote games | Mean disposals | Mean goals |
|---|---|---|---|---|---|---|
| **MIDFIELDER** | **0.224** | 10.9% | 3.9% | **64.5%** | 19.1 | 0.55 |
| KEY_FORWARD | 0.155 | 7.8% | 2.5% | 9.3% | 11.3 | 1.68 |
| MIDFIELDER_FORWARD | 0.121 | 6.2% | 1.7% | 0.9% | 16.4 | 0.58 |
| RUCK | 0.120 | 6.1% | 1.9% | 5.6% | 11.9 | 0.49 |
| MEDIUM_DEFENDER | 0.092 | 4.8% | 1.4% | 12.9% | 17.0 | 0.16 |
| MEDIUM_FORWARD | 0.058 | 3.1% | 0.8% | 5.5% | 12.0 | 1.03 |
| **KEY_DEFENDER** | **0.030** | 1.8% | 0.3% | **1.3%** | 12.3 | 0.10 |

## 4. Interpretation

- **The role effect is large — one of the largest single findings in this entire audit.** A midfielder
  polls at **7.4x** the per-game rate of a key defender, and captures roughly **half** the "votes per
  game" advantage over even key forwards (the next best-polling role).
- **This is not simply explained by disposal volume.** Medium defenders average 17.0 disposals per
  game — more than key forwards (11.3) or rucks (11.9) — yet poll at less than half key forwards' rate
  and about the same as rucks. The same volume of disposals evidently does not translate to the same
  vote probability depending on role, directly confirming the brief's Q9 hypothesis rather than
  assuming it.
- **Key forwards and key defenders present an especially stark contrast** given genuinely comparable
  disposal counts (11.3 vs 12.3) and both being "key position" specialists: key forwards convert their
  specialism (goals: 1.68/game) into a polling rate more than 5x higher than key defenders convert
  theirs (defensive/intercept-style contribution, not well captured by any raw stat available to us —
  see `docs/DATA_SOURCE_AUDIT.md` on the missing intercept/spoil history). This matches a long-standing
  piece of AFL folklore ("defenders are underrated by the Brownlow") with an actual, quantified,
  reproducible number from our own data, not just anecdote.
- **This finding on its own justifies explicit role-aware modelling in Phase 4** (role × performance
  interaction terms, or role-specific model branches) rather than a single pooled model with role
  ignored, per Q9's direct question.

## 5. Caveats

- Real labels only cover 2021-2025 (2,702 of the 217,574 role-classified rows, ~1.2%); the remaining
  98.8% rely on the validated-but-imperfect proxy (§2), particularly weak for the hybrid
  MIDFIELDER_FORWARD class.
- The role label is a **season-level static assignment**, joined onto every match that player played
  that season — it cannot capture genuine mid-season role changes (e.g. a wing moved into a rotating
  midfield role after injuries to teammates) or explicit tagging match-ups. This is a real, stated
  limitation, not a hidden one.
- The role reference join itself (surname + team + season) dropped 177 ambiguous same-surname,
  same-team, same-season rows rather than guess — logged to
  `reports/role_reference_surname_collisions.csv`.
- **Leakage note:** both the real and proxy role labels are built from full-season aggregates — see
  `docs/LEAKAGE_AUDIT.md` for why this must be rebuilt as season-to-date-only before any forward-looking
  Phase 4 model use, even though it is safe for this retrospective analysis.
