"""
Phase 3, section N: propose CORE / ADVANCED / EXPERIMENTAL feature set
membership per feature, combining availability (reports/feature_availability.csv),
univariate signal (reports/univariate_vote_relationships.csv), and redundancy
(reports/correlation_matrix.csv). This is a PROPOSAL for Phase 4 to start from,
not a final feature-selection decision -- Phase 4's own regularisation/
importance analysis on real models is what actually decides inclusion.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"

# Features that are inputs to a composite index and therefore redundant with it if both were
# included naively -- not dropped, but flagged, since the brief wants exclusions documented.
REDUNDANT_WITH_COMPOSITE = {
    "disposals_match_z": "possession_impact_index",
    "disposals_team_share": "possession_impact_index / teammate competition features",
    "contested_possessions_match_z": "possession_impact_index / contest_index",
    "clearances_match_z": "contest_index / clearance_index",
    "stoppage_clearances": "clearances (r=0.91, see correlation_matrix.csv)",
    "effective_disposals": "disposals (r=0.92) -- keep only if efficiency framing specifically matters",
    "afl_fantasy_points": "supercoach_points (r=0.86) -- keep one, not both, as an external composite benchmark",
}

LEAKAGE_EXCLUDE = set()  # nothing in this Phase 3 table touches future information -- see docs/LEAKAGE_AUDIT.md


def build() -> pd.DataFrame:
    avail = pd.read_csv(REPORTS_DIR / "feature_availability.csv")
    uni = pd.read_csv(REPORTS_DIR / "univariate_vote_relationships.csv")

    df = avail.merge(uni[["feature", "spearman_corr_votes", "mutual_info_votes", "n_obs"]], on="feature", how="left")

    def classify(row):
        if row["family"] in ("composite_metadata",):
            return "exclude_metadata_only"
        if pd.isna(row["first_reliable_season"]):
            return "exclude_insufficient_coverage"
        core = row["first_reliable_season"] <= 2003
        advanced = row["first_reliable_season"] <= 2015
        if core:
            return "CORE + ADVANCED"
        if advanced:
            return "ADVANCED only"
        return "ADVANCED only (recent-era, first available after 2015)"

    df["proposed_set"] = df.apply(classify, axis=1)
    df["redundancy_note"] = df["feature"].map(REDUNDANT_WITH_COMPOSITE).fillna("")
    df["leakage_flag"] = df["feature"].apply(lambda f: "EXCLUDE" if f in LEAKAGE_EXCLUDE else "SAFE")

    df = df.sort_values(["proposed_set", "spearman_corr_votes"], ascending=[True, False], key=lambda s: s.abs() if s.name == "spearman_corr_votes" else s)
    df.to_csv(REPORTS_DIR / "candidate_feature_sets.csv", index=False)
    print(f"Wrote {len(df)} rows -> reports/candidate_feature_sets.csv")
    print(df["proposed_set"].value_counts())
    return df


if __name__ == "__main__":
    build()
