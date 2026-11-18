"""Estado do grafo e tipos compartilhados entre os nós."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from app.llm.usage import NodeUsage


class ScoredChunk(BaseModel):
    """Chunk recuperado, com a posição e o score de cada etapa do retrieval."""

    id: str
    doc_id: str
    title: str
    section: str
    url: str
    text: str
    token_count: int
    score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    rerank_rank: int | None = None


class Citation(BaseModel):
    n: int
    chunk_id: str
    title: str
    section: str
    url: str


class AgentState(TypedDict, total=False):
    request_id: str
    query: str
    query_vector: list[float]
    cache_similarity: float | None

    # Router
    route: Literal["trivial", "out_of_scope", "simple_rag", "multi_hop"] | None
    route_reason: str | None
    route_confidence: float | None

    # Retrieval
    sub_queries: list[str]
    # Cada retrieve paralelo (Send) devolve uma lista de resultados; o reducer junta todas.
    sub_results: Annotated[list[list[ScoredChunk]], operator.add]
    candidates: list[ScoredChunk]
    reranked: list[ScoredChunk]
    context: list[ScoredChunk]
    docs_sufficient: bool | None
    rewritten_query: str
    retries: int

    # Geração e verificação
    answer: str | None
    citations: list[Citation]
    model_tier_used: Literal["small", "medium", "large"] | None
    next_tier: Literal["small", "medium", "large"] | None
    escalations: int
    grounded: bool | None
    unsupported_claims: list[str]
    gave_up: bool
    # Decisão tomada pelo nó (e lida pela aresta condicional seguinte)
    next_action: str

    cache_hit: Literal["exact", "semantic"] | None
    # Cada nó devolve uma lista com o próprio uso; o reducer concatena.
    usage: Annotated[list[NodeUsage], operator.add]


def spent_usd(state: AgentState) -> float:
    """Custo acumulado do request até aqui."""
    return sum(u.cost_usd for u in state.get("usage", []))
