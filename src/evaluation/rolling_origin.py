"""
Phase 4, section D/E: walk-forward (rolling-origin) train/test split generator.

No random splitting across seasons anywhere in this project -- every split
here trains only on seasons strictly before the test season.
"""
from dataclasses import dataclass


@dataclass
class Fold:
    train_seasons: list
    test_season: int
    window_name: str


def generate_folds(all_seasons: list, test_seasons: list, window_strategies: dict) -> list:
    """window_strategies: {name: window_size_or_None}. None means expanding
    (all available prior seasons); an int N means the most recent N seasons
    strictly before the test season."""
    folds = []
    for test_season in test_seasons:
        available_prior = sorted(s for s in all_seasons if s < test_season)
        for name, window in window_strategies.items():
            train = available_prior if window is None else available_prior[-window:]
            if len(train) == 0:
                continue
            folds.append(Fold(train_seasons=train, test_season=test_season, window_name=name))
    return folds
