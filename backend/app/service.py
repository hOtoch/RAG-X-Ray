"""Execução de uma pergunta no grafo. Usado pela API e pelo eval."""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel

from app.graph.build import get_graph
from app.graph.state import Citation, ScoredChunk
from app.llm.usage import NodeUsage


class QueryResult(BaseModel):
    request_id: str
    pipeline: str
    question: str
    answer: str
    citations: list[Citation]
    context: list[ScoredChunk]
    candidates: list[ScoredChunk]
    model_tier_used: str | None
    usage: list[NodeUsage]
    total_cost_usd: float
    total_latency_ms: float


def run_query(question: str, pipeline: str = "v0") -> QueryResult:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    state = get_graph(pipeline).invoke({"request_id": request_id, "query": question, "usage": []})
    usage = state.get("usage", [])
    return QueryResult(
        request_id=request_id,
        pipeline=pipeline,
        question=question,
        answer=state.get("answer") or "",
        citations=state.get("citations", []),
        context=state.get("context", []),
        candidates=state.get("candidates", []),
        model_tier_used=state.get("model_tier_used"),
        usage=usage,
        total_cost_usd=sum(u.cost_usd for u in usage),
        total_latency_ms=(time.perf_counter() - start) * 1000,
    )
