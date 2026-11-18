"""Reciprocal Rank Fusion: combina rankings de fontes diferentes usando só as posições.

score(d) = soma, em cada ranking onde d aparece, de 1 / (k + posição). Quem aparece bem colocado
em mais de uma fonte sobe; os scores originais (cosseno, BM25) não são comparados entre si.
"""

from __future__ import annotations

RRF_K = 60


def rrf_scores(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + position)
    return scores


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Ids ordenados pelo score RRF. Empates mantêm a ordem em que o id apareceu primeiro."""
    scores = rrf_scores(rankings, k)
    first_seen = {item: index for index, item in enumerate(dict.fromkeys(i for r in rankings for i in r))}
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
