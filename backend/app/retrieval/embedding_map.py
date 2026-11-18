"""Posição da pergunta no mapa 2D do corpus (gerado por ingest/corpus_map.py).

O UMAP não roda no servidor. A pergunta é colocada na média das posições dos seus vizinhos mais próximos na
busca densa, com peso 1/rank. É uma aproximação: o `transform` do UMAP daria um ponto um pouco diferente,
mas exigiria carregar o modelo (e o numba) em cada request. Para mostrar "a pergunta caiu nesta região do
corpus", os vizinhos bastam, e o ponto fica sempre no meio dos trechos que de fato foram recuperados.
"""

from __future__ import annotations

from functools import lru_cache

from app.graph.state import ScoredChunk

NEIGHBORS = 10


@lru_cache
def _coords() -> dict[str, tuple[float, float]]:
    from app.db import get_pool

    try:
        with get_pool().connection() as conn:
            rows = conn.execute("SELECT id, x, y FROM chunk_map").fetchall()
    except Exception:  # mapa ainda não gerado: o evento só não é emitido
        return {}
    return {r[0]: (float(r[1]), float(r[2])) for r in rows}


def query_position(hits: list[tuple[str, int]], coords: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """`hits` são pares (chunk_id, rank denso, começando em 1). Devolve None se nenhum vizinho estiver no mapa."""
    weighted = [(coords[cid], 1 / rank) for cid, rank in sorted(hits, key=lambda h: h[1])[:NEIGHBORS] if cid in coords]
    if not weighted:
        return None
    total = sum(w for _, w in weighted)
    x = sum(p[0] * w for p, w in weighted) / total
    y = sum(p[1] * w for p, w in weighted) / total
    return round(x, 4), round(y, 4)


def map_payload(
    chunks: list[ScoredChunk] | list[dict], coords: dict[str, tuple[float, float]] | None = None
) -> dict | None:
    """Payload do evento `embedding_map` a partir dos candidatos do retrieve (objetos ou dicts do evento)."""
    coords = _coords() if coords is None else coords
    if not coords:
        return None
    items = [c if isinstance(c, dict) else c.model_dump(include={"id", "dense_rank"}) for c in chunks]
    position = query_position([(c["id"], c["dense_rank"]) for c in items if c.get("dense_rank")], coords)
    if position is None:
        return None
    return {"query_xy": position, "hit_ids": [c["id"] for c in items]}
