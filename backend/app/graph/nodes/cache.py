"""Nós de cache: consulta exata, consulta semântica e gravação (só de respostas sustentadas)."""

from __future__ import annotations

from app import cache
from app.graph.instrument import emit_data
from app.graph.nodes.retrieve import embedding_usage
from app.graph.state import AgentState, Citation
from app.pipeline import PipelineConfig
from app.retrieval import dense


def _hit_update(entry: dict, kind: str) -> dict:
    return {
        "answer": entry["answer"],
        "citations": [Citation(**c) for c in entry.get("citations", [])],
        "model_tier_used": None,
        "route": entry.get("route"),
        "cache_hit": kind,
        "next_action": "hit",
    }


def make_cache_exact(pipeline: PipelineConfig):
    def cache_exact(state: AgentState) -> dict:
        entry = cache.lookup_exact(pipeline.name, state["query"])
        emit_data("cache_exact", {"hit": entry is not None})
        return _hit_update(entry, "exact") if entry else {"next_action": "miss"}

    return cache_exact


def make_cache_semantic(pipeline: PipelineConfig):
    def cache_semantic(state: AgentState) -> dict:
        query = state["query"]
        vector = dense.embed_query(query)
        threshold = pipeline.cache_threshold
        entry, similarity = cache.lookup_semantic(pipeline.name, vector, threshold)
        emit_data("cache_semantic", {"hit": entry is not None, "similarity": similarity, "threshold": threshold})
        # O vetor da pergunta fica no estado: o retrieve e o cache_write reaproveitam sem nova chamada à API.
        update = {"query_vector": vector, "usage": [embedding_usage(query)]}
        if entry:
            return update | _hit_update(entry, "semantic") | {"cache_similarity": similarity}
        return update | {"next_action": "miss", "cache_similarity": similarity}

    return cache_semantic


def should_write(state: AgentState) -> bool:
    """Só entra no cache resposta sustentada pelos trechos e que não é um 'não sei'."""
    if state.get("cache_hit") or state.get("gave_up"):
        return False
    if state.get("route") in ("trivial", "out_of_scope"):
        return state.get("route") == "out_of_scope"  # recusa por template é segura; conversa fiada não
    return state.get("grounded") is True


def make_cache_write(pipeline: PipelineConfig):
    def cache_write(state: AgentState) -> dict:
        write = should_write(state)
        if write:
            vector = state.get("query_vector") or dense.embed_query(state["query"])
            cache.put_entry(
                pipeline.name,
                state["query"],
                vector,
                {
                    "answer": state.get("answer", ""),
                    "citations": [c.model_dump() for c in state.get("citations", [])],
                    "route": state.get("route"),
                },
            )
        emit_data("cache_write", {"written": write})
        return {}

    return cache_write
