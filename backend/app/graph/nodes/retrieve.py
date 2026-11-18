"""Nó retrieve: busca os chunks candidatos para a pergunta (denso ou híbrido)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.config import get_settings
from app.graph.instrument import emit_data
from app.graph.state import AgentState, ScoredChunk
from app.llm.usage import NodeUsage, cost_usd
from app.pipeline import PipelineConfig
from app.retrieval import dense, lexical
from app.retrieval.embedding_map import map_payload
from app.retrieval.hybrid import fuse
from ingest.chunk import count_tokens

_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieve")


def retrieve_payload(chunks: list[ScoredChunk]) -> dict:
    fields = {"id", "title", "section", "url", "score", "dense_rank", "bm25_rank", "rrf_rank"}
    return {"chunks": [c.model_dump(include=fields) for c in chunks]}


def embedding_usage(query: str) -> NodeUsage:
    model = get_settings().embedding_model
    tokens = count_tokens(query)
    return NodeUsage(node="retrieve", model=model, tokens_in=tokens, cost_usd=cost_usd(model, tokens))


def search(query: str, pipeline: PipelineConfig, vector: list[float] | None = None) -> list[ScoredChunk]:
    """Candidatos em ordem. No modo híbrido, a busca densa e a BM25 rodam em paralelo e são fundidas por RRF.
    `vector` reaproveita o embedding da pergunta já calculado (pelo cache semântico, por exemplo)."""
    vector = vector or dense.embed_query(query)
    if pipeline.retrieval_mode == "dense":
        return dense.dense_search(vector, pipeline.top_k)
    lexical_future = _pool.submit(lexical.bm25_search, query, pipeline.candidates_k)
    dense_hits = dense.dense_search(vector, pipeline.candidates_k)
    return fuse(dense_hits, lexical_future.result())[: pipeline.candidates_k]


def make_retrieve(pipeline: PipelineConfig):
    def retrieve(state: AgentState) -> dict:
        query = state["query"]
        cached_vector = state.get("query_vector")
        candidates = search(query, pipeline, cached_vector)
        emit_data("retrieve", retrieve_payload(candidates))
        if position := map_payload(candidates):
            emit_data("embedding_map", position)
        # Sem rerank, os primeiros candidatos já são o contexto da geração.
        usage = [] if cached_vector else [embedding_usage(query)]
        return {"candidates": candidates, "context": candidates[: pipeline.top_k], "usage": usage}

    return retrieve
