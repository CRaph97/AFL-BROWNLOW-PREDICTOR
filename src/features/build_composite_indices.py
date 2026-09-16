"""
Phase 3, section K: simple, transparent composite performance indices, for
EXPLORATORY use only (concentration / teammate-competition / relative-dominance
analysis) -- NOT final Brownlow predictors, and NOT built with any Brownlow-vote
information whatsoever (they are pure functions of box-score stats).

Each index is the sum of its components' WITHIN-MATCH z-scores (mean 0, std 1
across all players in that match). Using within-match z-scores rather than raw
sums keeps every index comparable across eras and match paces automatically,
and is exactly the "normalise components and document the construction"
instruction from the brief. Missing components (structurally unavailable in a
given era) are dropped from that row's sum rather than treated as zero, and the
number of components actually summed is recorded so a thin (era-limited) index
value is distinguishable from a full one.

Index definitions:
  possession_impact_index = z(disposals) + z(contested_possessions) + z(uncontested_possessions)
  contest_index           = z(contested_possessions) + z(tackles) + z(clearances)
  clearance_index         = z(clearances)  [+ z(centre_clearances) where available, ADVANCED only]
  scoring_index           = z(goals) + 0.5*z(behinds) + z(goal_assists)
  territory_index         = z(inside_50s) + z(rebound_50s)  [+ z(metres_gained) where available, ADVANCED only]
  defensive_index         = z(one_percenters) + z(rebound_50s) + z(tackles)
"""
import numpy as np
import pandas as pd

INDEX_COMPONENTS = {
    "possession_impact_index": ["disposals", "contested_possessions", "uncontested_possessions"],
    "contest_index": ["contested_possessions", "tackles", "clearances"],
    "clearance_index": ["clearances"],
    "scoring_index": [("goals", 1.0), ("behinds", 0.5), ("goal_assists", 1.0)],
    "territory_index": ["inside_50s", "rebound_50s"],
    "defensive_index": ["one_percenters", "rebound_50s", "tackles"],
}

ADVANCED_EXTRA_COMPONENTS = {
    "clearance_index": ["centre_clearances"],
    "territory_index": ["metres_gained"],
}


def _match_z(df: pd.DataFrame, stat: str) -> pd.Series:
    g = df.groupby("match_id")[stat]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    return (df[stat] - mean) / std


def build(df: pd.DataFrame, extra_components: dict | None = None) -> pd.DataFrame:
    """df must contain match_id plus every stat referenced in INDEX_COMPONENTS
    (and extra_components, if given -- pass the ADVANCED table's extra columns
    here to get the richer clearance/territory index for 2010-2025 rows)."""
    out = pd.DataFrame(index=df.index)
    all_components = {k: list(v) for k, v in INDEX_COMPONENTS.items()}
    if extra_components:
        for idx, extra_stats in extra_components.items():
            all_components[idx] = all_components[idx] + [s for s in extra_stats if s in df.columns]

    for index_name, components in all_components.items():
        z_sum = pd.Series(0.0, index=df.index)
        n_components = pd.Series(0, index=df.index)
        for comp in components:
            weight = 1.0
            if isinstance(comp, tuple):
                comp, weight = comp
            if comp not in df.columns:
                continue
            z = _match_z(df, comp)
            valid = z.notna()
            z_sum = z_sum + weight * z.fillna(0)
            n_components = n_components + valid.astype(int)
        out[index_name] = np.where(n_components > 0, z_sum, np.nan)
        out[f"{index_name}_n_components"] = n_components

    return out


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    core = pd.read_parquet(ROOT / "data" / "processed" / "player_match_core_1984_2025.parquet")
    feats = build(core)
    print(feats.describe().T)
