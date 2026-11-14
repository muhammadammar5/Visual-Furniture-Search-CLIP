"""Ranking metrics. `relevant` is a list of booleans in rank order (True = correct category)."""
from __future__ import annotations

from typing import Sequence


def hit_at_k(relevant: Sequence[bool], k: int) -> float:
    """1 if at least one of the top-k results is relevant, else 0."""
    return float(any(relevant[:k]))


def precision_at_k(relevant: Sequence[bool], k: int) -> float:
    top = list(relevant[:k])
    return sum(top) / k if top else 0.0


def reciprocal_rank(relevant: Sequence[bool], k: int = 10) -> float:
    for rank, ok in enumerate(relevant[:k], start=1):
        if ok:
            return 1.0 / rank
    return 0.0
