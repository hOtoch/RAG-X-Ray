"""Rerank com cross-encoder local (ONNX via fastembed, roda em CPU).

O cross-encoder lê a pergunta e o trecho juntos e dá uma nota de relevância. É mais preciso que
comparar embeddings, mas custa uma inferência por trecho, então só roda sobre os candidatos.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import ROOT, get_settings
from app.graph.state import ScoredChunk

_lock = threading.Lock()


@lru_cache
def get_reranker(model: str) -> TextCrossEncoder:
    # Os modelos ficam em data/models (fora do git), e não no diretório temporário do sistema.
    return TextCrossEncoder(model_name=model, cache_dir=str(ROOT / "data" / "models"))


def rerank(query: str, chunks: list[ScoredChunk], top_k: int, model: str | None = None) -> list[ScoredChunk]:
    """Reordena os candidatos pela nota do cross-encoder e corta em top_k."""
    if not chunks:
        return []
    model = model or get_settings().reranker_model
    documents = [f"{c.title} > {c.section}\n{c.text}" if c.section else f"{c.title}\n{c.text}" for c in chunks]
    with _lock:  # a sessão ONNX usa todos os núcleos; duas inferências ao mesmo tempo só disputam CPU
        scores = list(get_reranker(model).rerank(query, documents))
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: -pair[1])[:top_k]
    return [
        chunk.model_copy(update={"score": float(score), "rerank_rank": position})
        for position, (chunk, score) in enumerate(ranked, start=1)
    ]
