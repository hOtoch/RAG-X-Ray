"""Montagem do grafo a partir de uma config de pipeline.

Sem router (v0, v1): cadeia retrieve -> [rerank] -> [compress_context] -> generate -> [groundedness_check].
Com router (v2 em diante):

    router ─ trivial ──────► direct_answer ─► fim
           ├ out_of_scope ─► refuse ─► fim
           ├ simple_rag ───► retrieve ─► rerank ─► compress_context ─► generate
           └ multi_hop ────► decompose ─► retrieve_sub (N em paralelo, Send) ─► merge_results ─► rerank
                               ─► grade_docs ─(insuficiente, até max_retries)─► retry_retrieve ─► rerank ...
                               ─► compress_context ─► generate
    generate ─► groundedness_check ─ ok ─► fim
                                   ├ falhou ─► escalate ─► generate (modelo um nível acima, uma vez)
                                   └ falhou de novo ─► give_up ("não sei") ─► fim
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.instrument import instrumented
from app.graph.nodes.cache import make_cache_exact, make_cache_semantic, make_cache_write
from app.graph.nodes.compress import make_compress
from app.graph.nodes.direct import make_direct_answer, make_refuse
from app.graph.nodes.generate import make_generate
from app.graph.nodes.groundedness import make_escalate, make_give_up, make_groundedness_check
from app.graph.nodes.multihop import (
    fan_out,
    make_decompose,
    make_grade_docs,
    make_merge_results,
    make_retrieve_sub,
    make_retry_retrieve,
)
from app.graph.nodes.rerank import make_rerank, make_select_context
from app.graph.nodes.retrieve import make_retrieve
from app.graph.nodes.router import make_router
from app.graph.state import AgentState
from app.pipeline import PipelineConfig, load_pipeline


def _next_action(state: AgentState) -> str:
    return state.get("next_action", "")


def _route(state: AgentState) -> str:
    return state.get("route") or "simple_rag"


def build_graph(pipeline: PipelineConfig):
    builder = StateGraph(AgentState)

    def node(name: str, factory) -> str:
        builder.add_node(name, instrumented(name, factory(pipeline)))
        return name

    # Com cache, toda resposta passa por cache_write antes do fim (ele decide se grava).
    final = END
    if pipeline.cache:
        final = node("cache_write", make_cache_write)
        builder.add_edge("cache_write", END)

    def entry(first: str) -> None:
        """START -> [cache_exact -> cache_semantic] -> first."""
        if not pipeline.cache:
            builder.add_edge(START, first)
            return
        node("cache_exact", make_cache_exact)
        node("cache_semantic", make_cache_semantic)
        builder.add_edge(START, "cache_exact")
        builder.add_conditional_edges("cache_exact", _next_action, {"hit": END, "miss": "cache_semantic"})
        builder.add_conditional_edges("cache_semantic", _next_action, {"hit": END, "miss": first})

    def after_generate(source: str) -> None:
        """Liga generate ao fim, passando pela checagem de groundedness quando ela está ligada."""
        if not pipeline.groundedness:
            builder.add_edge(source, final)
            return
        node("groundedness_check", make_groundedness_check)
        node("escalate", make_escalate)
        node("give_up", make_give_up)
        builder.add_edge(source, "groundedness_check")
        builder.add_conditional_edges(
            "groundedness_check", _next_action, {"finish": final, "escalate": "escalate", "give_up": "give_up"}
        )
        builder.add_edge("escalate", source)
        builder.add_edge("give_up", END)

    if not pipeline.router:
        steps = ["retrieve"]
        node("retrieve", make_retrieve)
        if pipeline.rerank:
            steps.append(node("rerank", make_rerank))
        if pipeline.compress:
            steps.append(node("compress_context", make_compress))
        steps.append(node("generate", make_generate))
        entry(steps[0])
        for a, b in zip(steps, steps[1:], strict=False):
            builder.add_edge(a, b)
        after_generate("generate")
        return builder.compile()

    # Ponto onde os candidatos viram contexto: rerank (se ligado) ou só o corte no top-k.
    select = "rerank" if pipeline.rerank else "select_context"
    for name, factory in [
        ("router", make_router),
        ("direct_answer", make_direct_answer),
        ("refuse", make_refuse),
        ("retrieve", make_retrieve),
        ("decompose", make_decompose),
        ("retrieve_sub", make_retrieve_sub),  # recebe só a sub-pergunta (via Send)
        ("merge_results", make_merge_results),
        (select, make_rerank if pipeline.rerank else make_select_context),
        ("grade_docs", make_grade_docs),
        ("retry_retrieve", make_retry_retrieve),
        ("compress_context", make_compress),
        ("generate", make_generate),
    ]:
        node(name, factory)

    entry("router")
    builder.add_conditional_edges(
        "router",
        _route,
        {"trivial": "direct_answer", "out_of_scope": "refuse", "simple_rag": "retrieve", "multi_hop": "decompose"},
    )
    builder.add_edge("direct_answer", final)
    builder.add_edge("refuse", final)

    builder.add_edge("retrieve", select)
    builder.add_conditional_edges("decompose", fan_out, ["retrieve_sub"])
    builder.add_edge("retrieve_sub", "merge_results")
    builder.add_edge("merge_results", select)
    builder.add_edge("retry_retrieve", select)
    # Depois da seleção: simple_rag segue direto; multi_hop passa pela avaliação dos trechos.
    builder.add_conditional_edges(
        select,
        lambda s: "grade_docs" if _route(s) == "multi_hop" else "compress_context",
        ["grade_docs", "compress_context"],
    )
    builder.add_conditional_edges(
        "grade_docs", _next_action, {"retry": "retry_retrieve", "generate": "compress_context"}
    )
    builder.add_edge("compress_context", "generate")
    after_generate("generate")
    return builder.compile()


def warmup(pipeline: PipelineConfig) -> None:
    """Carrega o que é lento na primeira chamada (índice BM25 e modelo de rerank)."""
    if pipeline.retrieval_mode == "hybrid":
        from app.retrieval import lexical

        lexical.get_bm25_index()
    if pipeline.rerank:
        from app.config import get_settings
        from app.retrieval import reranker

        reranker.get_reranker(get_settings().reranker_model)


@lru_cache
def get_graph(pipeline_name: str):
    return build_graph(load_pipeline(pipeline_name))
