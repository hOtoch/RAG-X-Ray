"""Nó rerank: reordena os candidatos com o cross-encoder e corta em top_k."""

from __future__ import annotations

from app.config import get_settings
from app.graph.instrument import emit_data
from app.graph.state import AgentState
from app.llm.usage import NodeUsage
from app.pipeline import PipelineConfig
from app.retrieval import reranker


def make_rerank(pipeline: PipelineConfig):
    def rerank(state: AgentState) -> dict:
        top_k = pipeline.multi_hop_top_k if state.get("route") == "multi_hop" else pipeline.top_k
        reranked = reranker.rerank(state["query"], state.get("candidates", []), top_k)
        emit_data(
            "rerank",
            {"chunks": [{"id": c.id, "rerank_rank": c.rerank_rank, "score": c.score} for c in reranked]},
        )
        # Modelo local: sem custo por token, mas o nome aparece no trace e na interface.
        return {
            "reranked": reranked,
            "context": reranked,
            "usage": [NodeUsage(node="rerank", model=get_settings().reranker_model)],
        }

    return rerank
