"""
Technical Summary -- a read-only explainer of how the modelling actually
works, for technically-minded readers who want more depth than the Guide &
FAQs page without reading raw docs/*.md research notes.

Every claim on this page is sourced from an existing validated doc or the
model source code itself (cited inline). Nothing here recomputes anything --
it is presentation only, exactly like app.py's Guide & FAQs page. Numeric
group weights are shown ONLY where they are genuinely fixed, documented
constants (the Objective model's hand-set weights, the production ensemble's
scenario weights) -- the Plackett-Luce production model's per-feature
coefficients are fitted per fold, not a single stable number, so its feature
importance is described qualitatively, sourced from the stability-analysis
finding in docs/PHASE4_DECISIONS.md / docs/ERROR_ANALYSIS.md.
"""
import streamlit as st

st.set_page_config(page_title="Technical Summary", layout="wide")
st.title("Technical Summary")
st.caption(
    "How the modelling actually works -- grounded in this project's own validated docs and code, "
    "not a marketing summary. For plain-English definitions of terms, see Guide & FAQs."
)

# --------------------------------------------------------------------------
# 1. How the modelling works
# --------------------------------------------------------------------------
st.header("1. How the Modelling Works")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🏈 Production Model")
    st.caption("Historically validated, adjusted for the 2026 rule change")
with c2:
    st.markdown("### 📊 Objective Stats Model")
    st.caption("2026-only, zero historical Brownlow voting data")
with c3:
    st.markdown("### 🔍 Wheelo / External")
    st.caption("Independent corroboration, never blindly blended")
st.markdown(
    "Three separate evidence layers, kept structurally apart. The Production and Objective "
    "models are each built and validated independently; Wheelo and other external sources are "
    "consulted afterward, as corroborating (or contradicting) evidence -- never as inputs that "
    "get mixed into either model's own probabilities."
)

st.divider()

# --------------------------------------------------------------------------
# 2. Production Model
# --------------------------------------------------------------------------
st.header("2. Production Model")

st.markdown(
    """
The Production model is a **Plackett-Luce rank-ordered choice model** ("exploded logit") --
it treats a Brownlow match literally as a ranking problem: exactly one 3-vote, one 2-vote, one
1-vote drawn from the same pool of players on the ground. Each player gets a latent utility
`u = β·x` from a linear combination of features, and P(3), P(2), P(1) are computed by an
**exact marginalisation** over all possible orderings of the other players -- not an approximation,
and not independently-estimated probabilities forced to sum to 1 after the fact.
"""
)

with st.expander("Data and history it uses"):
    st.markdown(
        """
- Trained on `model_core.parquet` (2003-2025, ~193K player-match rows) plus `model_advanced.parquet`
  (2015-2025, adds footywire's extended stats) for the stats-assisted 2026 scenario.
- **Recent-8-season training window** beats the alternative (using all history) on 8 of 8
  tracked backtest metrics -- recency helps this specific model, though not every model tested.
- For 2026 specifically, four scenarios are blended (see the 2026 structural-break section below)
  rather than a single model trained naively on old data.
"""
    )

with st.expander("Major feature groups"):
    st.markdown(
        """
Raw box-score stats, **match-relative** features (how a player compares to everyone else in the
same game), **team context** (win/loss, margin), **teammate competition**, **role** (validated
classifier, 2021-2025 real labels + a lagged proxy for earlier seasons), **nonlinear** performance
thresholds (e.g. disposal/goal "hinge" terms), and **lagged form** (prior-games rolling averages,
strictly `.shift(1)`-based so no future information ever leaks in).

Feature ablation found **match/team context is the single biggest lever** (+5.7 percentage points
on correct-3-vote accuracy -- the best single addition tested). Hand-built match-relative features
turned out to be **largely redundant** with what the ranking model already learns implicitly from
comparing players within a match. The win×margin interaction term is **confirmed redundant** --
identical metrics with or without it.
"""
    )

with st.expander("How recent form and context are handled"):
    st.markdown(
        """
Lagged-form features (disposals, contested possessions, clearances, etc.) use only *prior* games,
computed with `.shift(1)` before any rolling/expanding window -- this is a hard leakage-safety rule
applied identically in historical training and 2026 inference.

A genuine methodological finding: "season-to-date" features (a player's own average *so far this
season*) cannot exist for a season's Round 1 -- there's no prior within-season game to average. This
affects every historical season equally (not a 2026-specific bug) and, for 2026, means **6 of 207
matches cannot be scored** by the historical-behaviour scenarios and are excluded from the
leaderboard -- a small, disclosed, systematic under-count for anyone whose team's first fixture was
one of their best games.
"""
    )

with st.expander("How match-level 3/2/1 probabilities are produced"):
    st.markdown(
        """
Given each player's utility, the model computes:

    P(3-vote = i)  = exp(uᵢ) / Σⱼ exp(uⱼ)
    P(2-vote = k | 3=i)  = exp(uₖ) / Σ_{j≠i} exp(uⱼ)
    P(1-vote = m | 3=i, 2=k)  = exp(uₘ) / Σ_{j≠i,k} exp(uⱼ)

then marginalises exactly over who actually won 3rd/2nd/1st, rather than treating each vote level
as three separate classification problems. This guarantees coherent probabilities (every player's
P(3) across a match sums to exactly 1) by construction, not by post-hoc renormalisation.

A real numerical-stability bug was found and fixed here mid-project: unstandardised, mixed-scale
features (e.g. `metres_gained`, 0-600+) caused catastrophic cancellation in an earlier non-log-space
version of this likelihood. Fixed with train-only feature standardisation and a fully log-space
computation -- 9 regression tests cover this exact failure mode.
"""
    )

with st.expander("Calibration and validation approach"):
    st.markdown(
        """
Validated exclusively with **walk-forward (rolling-origin)** backtesting -- training uses only
seasons strictly *before* the test season, across 22 folds spanning 11 test seasons (2015-2025).
Never a random train/test split across seasons, which would leak future information.

Four architectures were compared this way; Plackett-Luce with the recent-8 window won outright --
first place on **every one of 8 tracked metrics simultaneously** (correct-3-vote rate, exact 3-2-1,
rank correlation, log loss, Brier score, and more), not a narrow or ambiguous result.

Calibration (Expected Calibration Error) confirms this isn't just good ranking: predicted P(3) is
within about half a percentage point of observed frequency on average (ECE ≈ 0.004) -- well
calibrated enough that no post-hoc correction (e.g. isotonic regression) is needed, unlike some
alternative architectures tested (one gradient-boosted variant needed mandatory recalibration,
being ~8-14x worse calibrated despite reasonable ranking accuracy).

A full historical pseudo-live simulation (predict every match of a season using only prior seasons'
training data, then compare season totals to the real result) found a **small, consistent positive
bias in every one of 8 test seasons** (+0.01 to +0.05 votes) -- the model slightly over-predicts
season totals on average. Reported as a genuine, disclosed limitation, not corrected for.
"""
    )

with st.expander("Known strengths and limitations"):
    st.markdown(
        """
**Strength:** the single most validated, best-performing model built in this project, on real
out-of-sample data, not just in-sample fit.

**Biggest limitation -- defenders.** When a key defender actually wins the 3 votes (rare: 32 of
1,958 matches in the backtest), the model correctly picks them only **12.5%** of the time, versus
**62%** for midfielders in the same situation. This is coherent with (not contradicted by) a
separate finding that a player's key-defender role status is the single largest, most stable
learned coefficient in the model -- it *does* give defenders real credit, just not always enough to
overcome their typically lower disposal/goal numbers on the rare occasion they have the best game
on the ground.
"""
    )

with st.expander("2026 structural considerations"):
    st.markdown(
        """
From 2026, umpires are given approved player statistics after each match before voting -- a
genuine process change with **no precedent in any historical data this project has**. No model
trained purely on pre-2026 votes can claim to have "learned" this effect. Rather than assume old
patterns transfer unchanged, four scenarios are built on the same validated architecture and
blended (see the comparison table and weightings section below), with the blend's sensitivity to
this assumption tracked and shown per player, not hidden inside one number.
"""
    )

st.divider()

# --------------------------------------------------------------------------
# 3. Objective Stats Model
# --------------------------------------------------------------------------
st.header("3. Objective Stats Model")
st.markdown(
    """
A deliberately **separate, independent** model built to test one specific hypothesis: because 2026
umpires now see objective post-match statistics, votes might be driven more by "who was objectively
best today" than by historical voting tendencies. It shares no code, no fitted parameters, and no
training data with the Production model.
"""
)

with st.expander("Deliberately independent from historical Brownlow voting"):
    st.markdown(
        """
Uses **zero** historical Brownlow vote data, no reputation effects, no player-identity history --
only a player's own current-match statistics and within-match context. It is a hand-specified
scoring system, not a model fitted against any past Brownlow outcome. There is deliberately **no
historical backtest** for it: testing a stats-only scoring philosophy against votes cast *before*
umpires ever saw such statistics would test a different hypothesis entirely. Its credibility rests
on the transparency of its inputs and weights, not on out-of-sample accuracy -- disclosed plainly,
not implied away.
"""
    )

with st.expander("Within-match normalisation and feature groups"):
    st.markdown(
        """
Every raw stat is converted to a **within-match z-score** before weighting, so a big number only
counts for as much as it's genuinely unusual in that specific match's context -- not an arbitrary
fixed scale. Ten conceptual feature groups feed a 100-point weight budget:
"""
    )
    st.table(
        {
            "Group": ["Possession Quality", "Contest", "Clearance", "Scoring", "Score Creation",
                      "Territory", "Defence", "Pressure", "Ruck", "Team Result"],
            "Weight": [16, 14, 12, 16, 10, 8, 8, 8, 4, 4],
        }
    )
    st.caption(
        "Never fitted against historical votes -- every weight has a stated, documented rationale "
        "(e.g. Defence partially offsets the Production model's documented tendency to under-credit "
        "defenders; Ruck is kept small because hitouts are near-zero for ~95% of players)."
    )

with st.expander("How performance becomes vote probabilities"):
    st.markdown(
        """
The resulting per-player Objective Performance Score is fed through the **same exact Plackett-Luce
marginalisation math** used by the Production model (a legitimate reuse of shared, validated
*probability machinery* -- not of the trained model or its fitted coefficients) to produce coherent
within-match P(3)/P(2)/P(1)/P(0).
"""
    )

with st.expander("Purpose and limitations"):
    st.markdown(
        """
Its main value is as an **independent second opinion** to compare against Production -- large
disagreement between the two flags a player whose projection is genuinely sensitive to modelling
philosophy, not model error. A sensitivity analysis (±25% perturbation on every group weight, 20
scenarios) found the top-20 list overlaps 19-20 of 20 players in every scenario (mean 19.5/20) --
the leaderboard is not an artefact of any single coefficient. The weights remain a judgement call,
though: a wholesale re-weighting philosophy (not just a perturbation) wasn't tested and could move
results more than the tested range.
"""
    )

st.divider()

# --------------------------------------------------------------------------
# 4. External / Wheelo
# --------------------------------------------------------------------------
st.header("4. External / Wheelo")
st.markdown(
    """
**Wheelo is the primary external benchmark** because it publishes genuine match-level predicted
votes and probabilities, not just a season-end ranking. Two exact fields are used: **`Votes`**
(a per-match expected-votes value, 0.0-3.0) and **`Votes3_Probability`** (a genuine 0-100 percent
P(3)-equivalent -- confirmed directly against the raw data, not assumed). Wheelo does not publish
P(2)/P(1), so a Wheelo "P(any vote)" is never synthesised or estimated.

ESPN and Betfair are included only where they exposed genuinely comparable data: ESPN contributes a
28-player top-N snapshot with its own round-by-round columns; Betfair contributes 54 players across
22 of 207 matches, season-total only (its round labelling was found to be non-monotonic in its own
source text, so round-level attribution was judged unreliable and dropped rather than guessed).
AFL.com.au's predictor page returned a JS-rendered shell with zero real prediction data in the
static HTML and was excluded entirely -- documented as unavailable, not faked.

**Why not blindly blended:** confidence classification explicitly checks whether Wheelo *confirms*,
is *neutral on*, or *materially contradicts* what the two internal models already agree on --
Wheelo can upgrade a result to "confirmed" or downgrade it when it disagrees, but it never
overrides Production or Objective's own probabilities, and a result built from Wheelo alone (no
ESPN/Betfair corroboration) is explicitly labelled "Wheelo benchmark" rather than implying a
broader consensus.
"""
)

st.divider()

# --------------------------------------------------------------------------
# 5. Model Comparison
# --------------------------------------------------------------------------
st.header("5. Model Comparison")
st.table(
    {
        "": ["Data basis", "Role", "Strength", "Main limitation"],
        "Production": [
            "Historical Brownlow votes (2003-2025) + 2026 match stats",
            "Primary forecasting model",
            "Best validated out-of-sample accuracy of any architecture tested (wins on 8/8 backtest metrics)",
            "Systematically under-picks key defenders (12.5% vs 62% for midfielders)",
        ],
        "Objective": [
            "2026 match statistics only -- zero historical votes",
            "Independent second opinion / structural-break stress-test",
            "Tests a genuinely different hypothesis with full transparency; not sensitive to any single weight",
            "No historical validation is possible for it by design",
        ],
        "Wheelo": [
            "Independent third-party model (2026 match-level predictions)",
            "Corroborating evidence, never blended into probabilities",
            "Genuine, real match-level EV and P(3) data from an outside source",
            "No P(2)/P(1) published; not built or validated by this project",
        ],
    }
)

st.divider()

# --------------------------------------------------------------------------
# 6. Validation / Integrity
# --------------------------------------------------------------------------
st.header("6. Validation / Integrity")
st.markdown(
    """
- **Walk-forward validation only** -- 22 folds, 11 test seasons, training strictly on prior seasons; never a random split.
- **Calibration checked directly**, not assumed: Production's predicted probabilities match observed frequency to within ~0.4 percentage points (ECE ≈ 0.004).
- **Season simulation**: Monte Carlo draws (100,000 simulated seasons for Production) respect the real match-level constraint that exactly one player gets 3, one gets 2, one gets 1 in every match -- never independently-sampled season totals that could imply an impossible match outcome.
- **Identity safeguards**: player identity is resolved via a stable composite key (hyphen-aware surname + first-initial + team + date); a genuinely ambiguous case (e.g. two real teammates who'd both plausibly match one external row) is always left unresolved rather than guessed -- confirmed and tested across several real, hard identity cases found during development.
- **Reproducibility**: all historical and 2026 data traces back to the same validated fitzRoy data mirror used throughout the project, with an explicit 2026 data-coverage audit before any prediction was produced.
- **Known caveats, disclosed rather than hidden**: the defender blind spot above; 3 of the 17 statistics confirmed shown to umpires in 2026 (kick-ins, intercept marks, spoils) aren't available in this project's public data sources; a small, consistent positive bias in predicted season totals (every one of 8 backtest seasons, same direction).
"""
)

st.divider()

# --------------------------------------------------------------------------
# 7. Weightings / Feature Importance
# --------------------------------------------------------------------------
st.header("7. Weightings / Feature Importance")
st.markdown("**Explicitly fixed, documented weights** (shown exactly, not approximated):")
w1, w2 = st.columns(2)
with w1:
    st.markdown("**2026 production ensemble** (Historical / Recent-era / Stats-assisted scenarios):")
    st.table({"Scenario": ["Historical (A)", "Recent-era (B)", "Stats-assisted (C)"], "Weight": ["45%", "20%", "35%"]})
    st.caption("A documented judgement call, not a fitted statistical result -- there are no 2026 votes yet to fit against.")
with w2:
    st.markdown("**Objective model's group weights** (of a 100-point budget):")
    st.table(
        {
            "Group": ["Possession Quality", "Contest", "Clearance", "Scoring", "Score Creation",
                      "Territory", "Defence", "Pressure", "Ruck", "Team Result"],
            "Weight": [16, 14, 12, 16, 10, 8, 8, 8, 4, 4],
        }
    )

st.markdown(
    """
**Production model feature importance is described qualitatively, not as fixed numbers** -- its
coefficients are *fitted* per training fold (recent-8 seasons, refit as new seasons arrive), so
there is no single stable "the weight is X" to display honestly. What repeated stability analysis
across folds does show: **role -- especially key-defender status -- is the single largest and most
consistent coefficient in the model**, match/team context is the single biggest lever found in
feature ablation (+5.7pp accuracy), and hand-built match-relative features and the win×margin
interaction were both found to add little to nothing once role, context, and teammate-competition
features are already present.
"""
)
