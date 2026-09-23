"""
Guide & FAQs -- plain-English explanation of the model for AFL-literate,
non-technical readers. Relocated here from app.py so app.py can be a pure
st.navigation() router (Streamlit requires this once st.navigation is used).
No longer the default landing page -- see app.py for the current default
(Brownlow Betting Opportunities).

Purely explanatory: no data loading, no model logic, no charts. Every claim
here is grounded in the existing validated docs (docs/2026_MODELLING_METHODOLOGY.md,
docs/2026_STRUCTURAL_BREAK.md, docs/2026_FINAL_AUDIT.md, docs/MODEL_BACKTEST.md,
docs/CALIBRATION.md, docs/ERROR_ANALYSIS.md, dashboard/data.py) -- nothing below
describes model behaviour that isn't documented there.

Run with: streamlit run app.py
"""
import streamlit as st

from dashboard import data as d

st.set_page_config(page_title="Guide & FAQs", layout="wide")
d.highlight_objective_stats_nav()
st.title("Guide & FAQs")
st.caption("A plain-English explanation of the model -- no stats background required.")

# --------------------------------------------------------------------------
# 1. 30-second overview
# --------------------------------------------------------------------------
st.header("30-Second Overview")
st.markdown(
    """
The model predicts the **chance of each player receiving 3, 2, 1, or 0 Brownlow votes**
in each match, using historical Brownlow voting patterns plus 2026 player-match statistics.

It doesn't just look at raw stats -- it also weighs:

- how dominant a player was **relative to everyone else in that match**
- whether the **team won or lost**, and by how much
- the player's **role** (midfielder, forward, defender, ruck)
- **teammate competition** (a big game counts for less if a teammate had an even bigger one)
- **nonlinear performance thresholds** (e.g. going from 25 to 30 disposals matters more than
  going from 10 to 15)

For 2026 specifically, the model separately accounts for the fact that **umpires now have
access to approved player statistics after each match** -- a genuine rule change with no
precedent in the historical data (see §6 below).

Match-level probabilities are then combined across the whole season to produce each
player's **expected vote total** and an **uncertainty range**, rather than a single number
pretending to be certain.
"""
)

st.divider()

# --------------------------------------------------------------------------
# 2. Glossary
# --------------------------------------------------------------------------
st.header("Glossary -- Plain English")

with st.expander("**Expected Votes (EV)**", expanded=True):
    st.markdown(
        """
The average number of Brownlow votes the model expects a player to receive **if the
season were replayed many times** under the same assumptions.

> Example: EV 2.4 in a match means the player is expected to earn about 2.4 votes on
> average -- not that 2.4 actual votes can be awarded. A real umpire panel can only ever
> award 3, 2, 1, or 0.
"""
    )

with st.expander("**P(3), P(2), P(1), P(0)**"):
    st.markdown(
        """
- **P(3)** -- probability the player receives 3 votes (best-afield) in that match.
- **P(2)** -- probability of 2 votes.
- **P(1)** -- probability of 1 vote.
- **P(0)** -- probability of no votes.

These four always add up to 100% for a given player in a given match.
"""
    )

with st.expander("**Most Likely Votes**"):
    st.markdown("The single most probable 3/2/1/0 outcome for that player in that match "
                "-- whichever of P(3)/P(2)/P(1)/P(0) is largest.")

with st.expander("**Median**"):
    st.markdown(
        "The middle simulated season result: across all simulated seasons, half finish "
        "above this number and half finish below it."
    )

with st.expander("**80% Low / 80% High**"):
    st.markdown(
        """
The middle 80% of simulated season outcomes fall between these two values.

> Example: an 80% range of 28-36 means roughly 8 in 10 simulated seasons produced a
> vote total somewhere between 28 and 36.
"""
    )

with st.expander("**95% Low / 95% High**"):
    st.markdown(
        "Same idea as the 80% range, but wider: roughly 95 in 100 simulated seasons fall "
        "inside this range. Useful for seeing genuinely unlikely-but-possible outcomes."
    )

with st.expander("**Model Disagreement**"):
    st.markdown(
        """
How differently the underlying scenario models (historical / recent-era / stats-assisted)
rate the same player, holding the data fixed.

- **High disagreement** -- the projection depends heavily on which modelling assumptions
  you trust.
- **Low disagreement** -- the different approaches broadly agree, so the projection is
  more robust to modelling choices.
"""
    )

with st.expander("**Structural-Break Sensitivity**"):
    st.markdown(
        """
How much a player's projected total changes depending on how strongly we assume the
2026 umpire-statistics rule change actually affects voting.

- **High sensitivity** -- this player benefits or suffers meaningfully if 2026 voting
  turns out to be more statistics-driven than in the past.
- **Low sensitivity** -- their projection stays similar either way, which is itself
  useful information (a more robust projection).
"""
    )

with st.expander("**Historical Scenario (Scenario A)**"):
    st.markdown("What the model expects if 2026 voting behaves like recent past seasons.")

with st.expander("**Recent-Era Scenario (Scenario B)**"):
    st.markdown(
        "Weights the newest seasons more heavily and reduces the influence of older-era "
        "voting norms."
    )

with st.expander("**Stats-Assisted Scenario (Scenario C)**"):
    st.markdown(
        "Places more emphasis on the objective statistics now available to umpires in "
        "2026 (see §6). It is the model's best available approximation of \"an umpiring "
        "process that weighs displayed stats more heavily\" -- not a literal reconstruction "
        "of what umpires see."
    )

with st.expander("**Final Ensemble**"):
    st.markdown(
        "The production forecast, combining the Historical, Recent-Era, and Stats-Assisted "
        "scenarios (weighted 45% / 20% / 35% respectively -- a documented judgement call, "
        "not a statistically fitted number, since there are no 2026 votes yet to fit against) "
        "rather than relying on a single model."
    )

with st.expander("**Reputation Effect**"):
    st.markdown(
        """
A controlled experiment testing whether players who have historically polled well tend to
receive more votes than their current-game stats alone would suggest.

**Important:** the model does not blindly reward famous players. Reputation/history is
only used where it was shown to genuinely improve forward-looking prediction on held-out
historical seasons -- and the default 2026 ensemble does **not** include it at all, on the
hypothesis that visible statistics may reduce umpires' reliance on memory/reputation in
2026. This is flagged as a hypothesis, not a proven fact.
"""
    )

with st.expander("**Projected 3s / 2s / 1s**"):
    st.markdown(
        "The number of matches in the season where that vote value (3, 2, or 1) is the "
        "model's single most likely outcome for that player."
    )

with st.expander("**Confidence**"):
    st.markdown(
        """
How concentrated the match-level probability distribution is.

> Example: P(3)=80% is much more confident than P(3)=34%, P(2)=31%, P(1)=25% -- even
> though 3 votes is still the "most likely" outcome in both cases.
"""
    )

st.divider()

# --------------------------------------------------------------------------
# 2b. Two models, and the terms for comparing them
# --------------------------------------------------------------------------
st.header("Two Models: Production vs. Objective")
st.markdown(
    """
This dashboard shows **two separate, independently-built models** -- deliberately not blended
into one number, so you can see where they agree and where they don't.

**PRODUCTION MODEL** -- the main forecast throughout this dashboard (leaderboard, round view,
match detail, uncertainty). Historically validated against real past Brownlow counts, and
adjusted for the 2026 rule change via the Historical / Recent-Era / Stats-Assisted scenario
blend described above -- this is the final ensemble forecast.

**OBJECTIVE STATS MODEL** -- a completely separate, experimental, **2026-only** model (see the
*Objective Stats Model* page). It uses only each player's actual 2026 match statistics and
context (relative performance, team result, nonlinear performance credit) -- it has **no
historical Brownlow voting data and no reputation effect of any kind**. Its purpose is
comparison, not replacement: where it agrees with the Production model, that's a useful
robustness signal; where it disagrees, that disagreement itself is informative.
"""
)

with st.expander("**Objective EV**"):
    st.markdown("The Objective Stats Model's own expected-votes number for a player -- same "
                "concept as Expected Votes above, computed from a model that never sees "
                "historical Brownlow votes.")

with st.expander("**Production EV**"):
    st.markdown("The Production model's expected-votes number -- what's shown everywhere else "
                "in this dashboard by default.")

with st.expander("**EV Difference**"):
    st.markdown("Objective EV minus Production EV for a player. A large gap means the two "
                "models, built on very different information, reach very different views of "
                "that player's season.")

with st.expander("**Rank Difference**"):
    st.markdown("How many leaderboard positions apart a player sits between the two models' "
                "own rankings. Small = the models broadly agree on where this player belongs; "
                "large = they don't.")

with st.expander("**Model Agreement**"):
    st.markdown(
        "Places where the Production and Objective models -- built from different information "
        "and different assumptions -- **independently reach broadly similar conclusions** "
        "(similar rank, similar EV, the same 3-2-1 pick in a match, or the same betting value "
        "call). See the *Model Agreement* page. Agreement is a useful robustness signal, but "
        "**it does not guarantee either model is correct** -- both could share the same blind spot."
    )

st.divider()

# --------------------------------------------------------------------------
# 2c. Betting terms
# --------------------------------------------------------------------------
st.header("Betting Terms")
st.caption(
    "These appear on the *Betting Opportunities* page, which compares Sportsbet's prices "
    "against this project's model probabilities (read-only; sourced from a separate, "
    "manually-refreshed markets project -- see that page for details)."
)

with st.expander("**Implied Probability**"):
    st.markdown("What a bookmaker's price implies the true probability is, ignoring their "
                "margin: `1 / odds`. E.g. odds of 4.00 imply a 25% probability.")

with st.expander("**Model Probability**"):
    st.markdown("This project's own estimated probability for that same outcome, from Monte "
                "Carlo simulation.")

with st.expander("**Probability Edge**"):
    st.markdown("Model Probability minus Implied Probability, in percentage points. A positive "
                "edge means the model thinks the outcome is more likely than the price suggests.")

with st.expander("**Expected Value / EV**"):
    st.markdown("The average return per $1 staked if this exact edge were genuinely correct and "
                "the bet were repeated many times: `model probability x odds - 1`. Positive EV "
                "does not guarantee a win on any single bet.")

with st.expander("**High-Confidence Value**"):
    st.markdown("Passed every mapping/settlement/wording check, has a meaningful positive edge "
                "and EV, and the player's projection itself has low/moderate model disagreement "
                "and structural-break sensitivity.")

with st.expander("**Model-Sensitive Value**"):
    st.markdown("Same edge/EV bar as High-Confidence Value, but this player's projection has "
                "elevated model disagreement or structural-break sensitivity -- the edge is real "
                "relative to the model, but the model itself is less sure here.")

with st.expander("**Speculative Upside**"):
    st.markdown("Longer-priced (odds 15+), positive expected value, lower absolute probability -- "
                "still model-supported, but a longshot by nature.")

with st.expander("**Review Required**"):
    st.markdown("The mechanical audit (market wording, player/team mapping, settlement rules) "
                "couldn't fully verify this row -- held out of headline value sections until a "
                "human checks it, not deleted.")

with st.expander("**Settlement Uncertain**"):
    st.markdown("A head-to-head or similar market where the two named outcomes have a real, "
                "non-trivial chance of tying (a dead heat / push) -- excluded from headline value "
                "because the simulated edge doesn't account for how a push actually settles.")

with st.expander("**Price Suspect**"):
    st.markdown("The audit found a pricing pattern that looks stale or templated (e.g. every "
                "team priced at an identical 1.87/1.87) rather than a genuine, currently-accurate "
                "market -- excluded from headline value pending a fresher price.")

st.divider()

# --------------------------------------------------------------------------
# 2d. Evaluation metrics (2026 Evaluation page)
# --------------------------------------------------------------------------
st.header("Evaluation Metrics")
st.caption(
    "These appear on the *2026 Evaluation* page, which scores the frozen pre-count forecasts "
    "against the AFL's actual votes. In one line: MAE / RMSE = vote-total accuracy; Spearman = "
    "ranking accuracy; 3-vote / exact 3-2-1 = match prediction accuracy; Brier / Log Loss / ECE = "
    "probability quality; ROI = retrospective betting performance."
)

with st.expander("**MAE -- Mean Absolute Error**"):
    st.markdown("Average absolute difference between predicted and actual votes. Lower is better.")

with st.expander("**RMSE -- Root Mean Squared Error**"):
    st.markdown("Similar to MAE but penalises large misses more heavily. Lower is better.")

with st.expander("**Spearman Correlation**"):
    st.markdown("Measures how well predicted player rankings match actual rankings. "
                "Range -1 to +1; higher is better.")

with st.expander("**3-Vote Winner Accuracy**"):
    st.markdown("% of matches where the model correctly identified the actual 3-vote player.")

with st.expander("**Exact 3-2-1 Accuracy**"):
    st.markdown("% of matches where the exact ordered 3-, 2- and 1-vote players were all correct.")

with st.expander("**Brier Score**"):
    st.markdown("Probability accuracy metric for binary outcomes. Lower is better; 0 is perfect.")

with st.expander("**Log Loss**"):
    st.markdown("Probability metric that heavily penalises confident wrong predictions. "
                "Lower is better.")

with st.expander("**ECE -- Expected Calibration Error**"):
    st.markdown("Measures whether predicted probabilities match observed frequencies. "
                "Lower is better; near 0 = well calibrated.")

with st.expander("**Rank Error**"):
    st.markdown("Absolute difference between predicted finishing rank and actual rank. "
                "Lower is better.")

with st.expander("**Top-N Hit Rate**"):
    st.markdown("% of actual Top-N players also contained in the model's predicted Top-N.")

with st.expander("**ROI -- Return on Investment**"):
    st.markdown("Profit/loss divided by stake. Positive = profit; negative = loss. In our "
                "evaluation this uses retrospective flat 1-unit staking only -- it is not a "
                "staking plan or a claim about future results.")

st.divider()

# --------------------------------------------------------------------------
# 3. How the model reaches a match prediction
# --------------------------------------------------------------------------
st.header("How the Model Reaches a Match Prediction")
st.markdown(
    """
1. Collect the player's performance statistics for that match.
2. Compare the player with everyone else on the ground in the same game.
3. Account for team result, margin, role, and teammate competition.
4. Apply relationships learned from historical Brownlow voting.
5. Adjust for the 2026 statistics-assisted voting environment (blended across scenarios).
6. Estimate P(3), P(2), P(1), P(0) for every player in the match -- these always sum
   correctly within the match (e.g. all players' P(3) values add to 1).
7. Combine every match across the season into a season-long expected vote total and
   uncertainty range via simulation.
"""
)

st.divider()

# --------------------------------------------------------------------------
# 4. Why EV isn't the same as a predicted vote total
# --------------------------------------------------------------------------
st.header("Why EV Is Not the Same As a Predicted Vote Total")
st.info(
    "A player with **30 EV is not being predicted to receive exactly 30 votes**. It means "
    "30 is the *average* across many possible simulated vote counts for that player's season. "
    "Actual Brownlow voting is discrete (3, 2, 1, or 0 per match) and subjective -- which is "
    "exactly why the uncertainty ranges (80%/95%) matter as much as the headline EV number."
)

st.divider()

# --------------------------------------------------------------------------
# 5. How to read a player line
# --------------------------------------------------------------------------
st.header("How to Read a Player Line")
st.code(
    "P(3)  60%\nP(2)  25%\nP(1)  10%\nP(0)   5%\nEV = 2.40",
    language=None,
)
st.markdown(
    """
This player is the **clear favourite for 3 votes** in this match (60% chance) but there's a
real 40% chance they get fewer -- or none. The expected value, 2.40, blends all four
outcomes into one number: `3×0.60 + 2×0.25 + 1×0.10 + 0×0.05 = 2.40`. It's a useful summary,
but the full P(3)/P(2)/P(1)/P(0) breakdown tells you more about *how confident* the model
actually is.
"""
)

st.divider()

# --------------------------------------------------------------------------
# 6. 2026 rule change
# --------------------------------------------------------------------------
st.header("The 2026 Rule Change")
st.warning(
    """
**From 2026, the four field umpires are given approved player-performance statistics
after each match and before voting** -- a genuine, confirmed process change.

Because **no prior Brownlow season was run under this exact system**, its real effect on
voting behaviour cannot yet be measured from historical data. Any model claiming to have
"learned" the 2026 effect from pre-2026 votes would be overstating what's actually
estimable.

That's why this dashboard shows **structural-break sensitivity** and **scenario
comparisons** rather than a single number pretending the effect is known. See the
*Scenario Comparison* and *Model Disagreement* pages to see exactly which players'
projections are most exposed to this uncertainty.
"""
)

st.divider()

# --------------------------------------------------------------------------
# 7. What the model does not know
# --------------------------------------------------------------------------
st.header("What the Model Does Not Know")
st.markdown(
    """
- Brownlow votes are still **subjective human decisions**, not a deterministic function of stats.
- Umpires may value on-field things (leadership, momentum, "the eye test") that are hard to
  capture in public box-score data.
- **Defensive performances are historically harder for the model to predict.** When a key
  defender genuinely wins the 3 votes, the model correctly picks them only about **12.5%**
  of the time, versus **62%** for midfielders in the same situation (see the *Defender Bias
  Watchlist* page).
- **Three of the 17 statistics confirmed to be shown to umpires in 2026** (kick-ins,
  intercept marks, spoils) are not available in this project's public data sources, so
  Scenario C is a close approximation, not a literal reconstruction, of what umpires see.
- All probabilities are **estimates**, not certainties -- even a well-calibrated model is
  regularly "wrong" on any single match by design.
"""
)

st.divider()

# --------------------------------------------------------------------------
# 8. FAQ
# --------------------------------------------------------------------------
st.header("FAQ")

faqs = [
    (
        "Why can EV be a decimal?",
        "Because it's an average across many possible outcomes (0, 1, 2, or 3 votes), not a "
        "single predicted vote count. Real Brownlow votes are always whole numbers; EV isn't "
        "trying to be one.",
    ),
    (
        "Why can a player have high EV but not be the most likely 3-vote player in every game?",
        "EV accumulates gradually across many matches, each with modest probabilities. A "
        "player can build a big season total from lots of 1.0-2.0 EV games without ever being "
        "the outright 3-vote favourite -- consistency adds up.",
    ),
    (
        "Why is the 95% range so wide?",
        "Brownlow voting has real match-to-match randomness (close games, other players having "
        "career nights) that compounds across a ~22-match season. A wide 95% range is an honest "
        "reflection of that variance, not a modelling weakness.",
    ),
    (
        "What does high model disagreement mean?",
        "The historical, recent-era, and stats-assisted versions of the model rate that player "
        "quite differently. Treat their final number as less certain than a player where all "
        "versions roughly agree.",
    ),
    (
        "What does high structural-break sensitivity mean?",
        "This player's projection would change a lot if the 2026 umpire-statistics rule change "
        "turns out to matter more (or less) than the model's default assumption. It's a flag "
        "for \"this projection depends on an assumption we can't yet verify,\" not an error.",
    ),
    (
        "Does the model know who the umpires are?",
        "No. It has no umpire-specific data at all -- only player, team, and match statistics, "
        "plus the documented list of statistics umpires are now shown post-match.",
    ),
    (
        "Does reputation automatically give a player more votes?",
        "No. Reputation is tested as a separate, clearly-labelled experiment and is not included "
        "in the default 2026 ensemble at all -- see the Reputation Effect glossary entry above.",
    ),
    (
        "Can the model predict the exact Brownlow count?",
        "No, and it isn't designed to. It estimates probabilities and expected values with "
        "explicit uncertainty ranges -- treat the leaderboard as a ranked set of plausible "
        "outcomes, not a guaranteed final tally.",
    ),
    (
        "Why are defenders sometimes flagged?",
        "The model has a documented, validated blind spot: it under-predicts elite defensive "
        "performances relative to midfield ones (12.5% vs 62% correct-pick rate when they "
        "actually win the votes). The Defender Bias Watchlist page lists specific 2026 games "
        "where this may be underweighting a real contender.",
    ),
    (
        "Why might the model differ from betting markets or media predictions?",
        "Betting markets and media tips can incorporate information this model doesn't use "
        "(injury news, team-of-the-week buzz, subjective form reads) and can also be influenced "
        "by public sentiment. This model is built only from historical voting patterns and "
        "box-score statistics, validated by walk-forward backtesting -- it will sometimes "
        "disagree with market consensus, and neither is guaranteed to be closer to the truth.",
    ),
]

for question, answer in faqs:
    with st.expander(question):
        st.markdown(answer)

st.divider()
st.caption(
    "Sources: docs/2026_MODELLING_METHODOLOGY.md, docs/2026_STRUCTURAL_BREAK.md, "
    "docs/2026_FINAL_AUDIT.md, docs/MODEL_BACKTEST.md, docs/CALIBRATION.md, "
    "docs/ERROR_ANALYSIS.md, dashboard/data.py. See the sidebar's Overview page for the "
    "live leaderboard."
)
