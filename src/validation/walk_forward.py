"""
Walk-forward (chronological) folds for 2027 R&D. train seasons < test season,
never random. Two window policies: expanding (all prior) and recent-k.
Nested selection: inner_holdout = last training season, for hyperparameter /
temperature / calibration fitting without touching the test season.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fold:
    test_season: int
    train_seasons: tuple
    window: str

    @property
    def inner_train(self) -> tuple:
        return self.train_seasons[:-1]

    @property
    def inner_holdout(self) -> int:
        return self.train_seasons[-1]


def make_folds(all_seasons: list[int], test_seasons: list[int], window: int | None = None, min_train: int = 5) -> list[Fold]:
    name = "expanding" if window is None else f"recent{window}"
    folds = []
    for t in test_seasons:
        prior = sorted(s for s in all_seasons if s < t)
        if window is not None:
            prior = prior[-window:]
        if len(prior) >= min_train:
            folds.append(Fold(test_season=int(t), train_seasons=tuple(int(s) for s in prior), window=name))
    return folds


def assert_no_overlap(fold: Fold) -> None:
    assert fold.test_season not in fold.train_seasons
    assert all(s < fold.test_season for s in fold.train_seasons)
