from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (item_id, _score) in enumerate(ranking):
            scores[item_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: -item[1])
