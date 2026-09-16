"""
Phase 3, section D: winning / margin context features.

Kept deliberately simple and interpretable -- these are inputs to an EXPLORATORY
question ("does the winner effect depend on margin?"), not a final model, so no
weighting or combination is baked in here.
"""
import pandas as pd

CLOSE_GAME_THRESHOLD = 12   # <= 2 scoring shots at match end, a common exploratory cut in AFL analysis
BLOWOUT_THRESHOLD = 50      # exploratory cut; both thresholds are revisited in docs/EXPLORATORY_ANALYSIS.md,
                            # not treated as sacred per the Phase 3 instruction


def build(core: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=core.index)
    out["is_win"] = (core["win_loss_draw"] == "win").astype(int)
    out["is_loss"] = (core["win_loss_draw"] == "loss").astype(int)
    out["is_draw"] = (core["win_loss_draw"] == "draw").astype(int)
    out["is_close_game"] = (core["absolute_margin"] <= CLOSE_GAME_THRESHOLD).astype(int)
    out["is_blowout"] = (core["absolute_margin"] >= BLOWOUT_THRESHOLD).astype(int)
    # margin / absolute_margin are already present on the CORE table -- not duplicated here.
    return out


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    core = pd.read_parquet(ROOT / "data" / "processed" / "player_match_core_1984_2025.parquet")
    feats = build(core)
    print(feats.describe().T)
