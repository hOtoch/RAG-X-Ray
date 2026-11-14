"""Nó retrieve: busca os chunks candidatos para a pergunta."""

from __future__ import annotations

import time

from app.config import get_settings
from app.graph.state import AgentState
from app.llm.usage import NodeUsage, cost_usd
from app.pipeline import PipelineConfig
from app.retrieval import dense
from ingest.chunk import count_tokens


def make_retrieve(pipeline: PipelineConfig):
    def retrieve(state: AgentState) -> dict:
        start = time.perf_counter()
        s = get_settings()
        query = state["query"]
        vector = dense.embed_query(query)
        candidates = dense.dense_search(vector, pipeline.top_k)
        tokens = count_tokens(query)
        usage = NodeUsage(
            node="retrieve",
            model=s.embedding_model,
            tokens_in=tokens,
            cost_usd=cost_usd(s.embedding_model, tokens),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return {"candidates": candidates, "context": candidates, "usage": [usage]}

    return retrieve
