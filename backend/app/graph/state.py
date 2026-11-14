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
    candidates: list[ScoredChunk]
    context: list[ScoredChunk]
    answer: str | None
    citations: list[Citation]
    model_tier_used: Literal["small", "medium", "large"] | None
    # Cada nó devolve uma lista com o próprio uso; o reducer concatena.
    usage: Annotated[list[NodeUsage], operator.add]
