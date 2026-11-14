"""Métricas de retrieval: recall@k, MRR e nDCG@k.

Um id relevante pode ser de chunk ("doc-x#chunk-3") ou de documento ("doc-x"). No segundo caso,
qualquer chunk do documento conta como acerto.
"""

from __future__ import annotations

import math


def matches(retrieved_id: str, relevant_id: str) -> bool:
    if "#" in relevant_id:
        return retrieved_id == relevant_id
    return retrieved_id.split("#")[0] == relevant_id


def _hits(retrieved: list[str], relevant: list[str]) -> list[int | None]:
    """Para cada posição recuperada, o índice do item relevante que ela acerta (ou None)."""
    found: set[int] = set()
    out: list[int | None] = []
    for rid in retrieved:
        hit = next((i for i, rel in enumerate(relevant) if i not in found and matches(rid, rel)), None)
        if hit is not None:
            found.add(hit)
        out.append(hit)
    return out


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fração dos itens relevantes que aparecem nas k primeiras posições."""
    if not relevant:
        raise ValueError("recall indefinido sem itens relevantes")
    hits = [h for h in _hits(retrieved[:k], relevant) if h is not None]
    return len(hits) / len(relevant)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """Inverso da posição do primeiro item relevante (0 se nenhum aparece)."""
    if not relevant:
        raise ValueError("MRR indefinido sem itens relevantes")
    for position, hit in enumerate(_hits(retrieved, relevant), start=1):
        if hit is not None:
            return 1 / position
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """nDCG com relevância binária: acertos no topo valem mais."""
    if not relevant:
        raise ValueError("nDCG indefinido sem itens relevantes")
    dcg = sum(
        1 / math.log2(position + 1)
        for position, hit in enumerate(_hits(retrieved[:k], relevant), start=1)
        if hit is not None
    )
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def doc_level(ids: list[str]) -> list[str]:
    """Reduz ids de chunk a ids de documento, sem repetir e mantendo a ordem."""
    out: list[str] = []
    for i in ids:
        doc = i.split("#")[0]
        if doc not in out:
            out.append(doc)
    return out
