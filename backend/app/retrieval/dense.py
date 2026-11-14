"""Busca densa: vizinhos mais próximos por similaridade de cosseno no pgvector."""

from __future__ import annotations

import numpy as np

from app.db import get_pool
from app.graph.state import ScoredChunk
from app.llm.factory import get_embeddings

COLUMNS = "id, doc_id, title, section, url, text, token_count"


def embed_query(query: str) -> list[float]:
    return get_embeddings().embed_query(query)


def dense_search(query_vector: list[float], k: int) -> list[ScoredChunk]:
    vector = np.array(query_vector, dtype=np.float32)
    with get_pool().connection() as conn:
        rows = conn.execute(
            f"SELECT {COLUMNS}, 1 - (embedding <=> %s) AS score FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (vector, vector, k),
        ).fetchall()
    return [
        ScoredChunk(
            id=r[0],
            doc_id=r[1],
            title=r[2],
            section=r[3],
            url=r[4],
            text=r[5],
            token_count=r[6],
            score=float(r[7]),
            dense_rank=rank,
        )
        for rank, r in enumerate(rows, start=1)
    ]
