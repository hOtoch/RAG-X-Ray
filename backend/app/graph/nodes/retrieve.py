"""Nó retrieve: busca os chunks candidatos para a pergunta."""

from __future__ import annotations

from app.config import get_settings
from app.graph.instrument import emit_data
from app.graph.state import AgentState, ScoredChunk
from app.llm.usage import NodeUsage, cost_usd
from app.pipeline import PipelineConfig
from app.retrieval import dense
from ingest.chunk import count_tokens


def retrieve_payload(chunks: list[ScoredChunk]) -> dict:
    fields = {"id", "title", "section", "url", "score", "dense_rank", "bm25_rank", "rrf_rank"}
    return {"chunks": [c.model_dump(include=fields) for c in chunks]}


def embedding_usage(query: str) -> NodeUsage:
    model = get_settings().embedding_model
    tokens = count_tokens(query)
    return NodeUsage(node="retrieve", model=model, tokens_in=tokens, cost_usd=cost_usd(model, tokens))


def make_retrieve(pipeline: PipelineConfig):
    def retrieve(state: AgentState) -> dict:
        query = state["query"]
        vector = dense.embed_query(query)
        candidates = dense.dense_search(vector, pipeline.top_k)
        emit_data("retrieve", retrieve_payload(candidates))
        return {"candidates": candidates, "context": candidates, "usage": [embedding_usage(query)]}

    return retrieve
