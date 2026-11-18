"""Nó compress_context: remove trechos quase iguais e respeita o orçamento de tokens da rota."""

from __future__ import annotations

from app.graph.instrument import emit_data
from app.graph.state import AgentState, ScoredChunk
from app.pipeline import PipelineConfig
from app.retrieval.lexical import TOKEN

DUPLICATE_THRESHOLD = 0.9


def _words(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Jaccard sobre o conjunto de palavras: 1.0 para textos com as mesmas palavras."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def dedupe(chunks: list[ScoredChunk], threshold: float = DUPLICATE_THRESHOLD) -> list[ScoredChunk]:
    """Mantém o primeiro (mais bem colocado) de cada grupo de trechos quase iguais."""
    kept: list[ScoredChunk] = []
    for chunk in chunks:
        if all(similarity(chunk.text, other.text) < threshold for other in kept):
            kept.append(chunk)
    return kept


def fit_budget(chunks: list[ScoredChunk], budget: int | None) -> list[ScoredChunk]:
    """Mantém os trechos, na ordem, enquanto couberem no orçamento. O primeiro sempre entra."""
    if budget is None:
        return chunks
    kept: list[ScoredChunk] = []
    used = 0
    for chunk in chunks:
        if kept and used + chunk.token_count > budget:
            break
        kept.append(chunk)
        used += chunk.token_count
    return kept


def compress(chunks: list[ScoredChunk], budget: int | None) -> list[ScoredChunk]:
    return fit_budget(dedupe(chunks), budget)


def make_compress(pipeline: PipelineConfig):
    def compress_context(state: AgentState) -> dict:
        before = state.get("context", [])
        after = compress(before, pipeline.context_token_budget)
        emit_data(
            "compress_context",
            {
                "chunks_before": len(before),
                "chunks_after": len(after),
                "tokens_before": sum(c.token_count for c in before),
                "tokens_after": sum(c.token_count for c in after),
                "budget": pipeline.context_token_budget,
            },
        )
        return {"context": after}

    return compress_context
