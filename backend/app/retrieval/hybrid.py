"""Retrieval híbrido: funde a busca densa e a lexical por RRF, guardando a posição em cada fonte."""

from __future__ import annotations

from app.graph.state import ScoredChunk
from app.retrieval.rrf import RRF_K, rrf_fuse, rrf_scores


def fuse(dense: list[ScoredChunk], lexical: list[ScoredChunk], k: int = RRF_K) -> list[ScoredChunk]:
    """Devolve os chunks das duas listas ordenados por RRF, com dense_rank, bm25_rank e rrf_rank."""
    rankings = [[c.id for c in dense], [c.id for c in lexical]]
    scores = rrf_scores(rankings, k)
    by_id = {c.id: c for c in lexical} | {c.id: c for c in dense}
    dense_rank = {c.id: c.dense_rank for c in dense}
    lexical_rank = {c.id: c.bm25_rank for c in lexical}
    return [
        by_id[chunk_id].model_copy(
            update={
                "score": scores[chunk_id],
                "dense_rank": dense_rank.get(chunk_id),
                "bm25_rank": lexical_rank.get(chunk_id),
                "rrf_rank": position,
            }
        )
        for position, chunk_id in enumerate(rrf_fuse(rankings, k), start=1)
    ]
