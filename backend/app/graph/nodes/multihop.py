"""Caminho multi-hop: decompor a pergunta, buscar cada parte em paralelo, juntar, avaliar os trechos
e, se não forem suficientes, reescrever a busca e tentar de novo (Corrective RAG)."""

from __future__ import annotations

from langgraph.types import Send
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.instrument import emit_data
from app.graph.nodes.generate import format_context
from app.graph.nodes.retrieve import embedding_usage, retrieve_payload, search
from app.graph.state import AgentState, ScoredChunk, spent_usd
from app.llm.structured import call_structured
from app.pipeline import PipelineConfig, budget_usd
from app.retrieval.rrf import rrf_fuse

MAX_SUB_QUERIES = 3

DECOMPOSE_PROMPT = """You split a question about the LangChain ecosystem documentation into search queries.

Write 2 or 3 short, self-contained search queries, one per piece of information needed to answer the \
question. Each query must name the library or feature it is about. Write the queries in English, even if \
the question is in another language. Do not answer the question."""

GRADE_PROMPT = """You check whether retrieved documentation passages are enough to answer a question.

- sufficient: true if the passages together contain the information needed to answer the whole question.
- If not sufficient, write in `rewritten_query` ONE new search query (in English, naming the library or \
feature) aimed at the missing information. Otherwise leave it empty.
- reason: at most 12 words."""


class SubQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=MAX_SUB_QUERIES)


class DocsGrade(BaseModel):
    sufficient: bool
    reason: str = Field(description="At most 12 words")
    rewritten_query: str = ""


def make_decompose(pipeline: PipelineConfig):
    def decompose(state: AgentState) -> dict:
        result, usage = call_structured("small", SubQueries, DECOMPOSE_PROMPT, state["query"], "decompose")
        sub_queries = [q.strip() for q in result.queries if q.strip()][:MAX_SUB_QUERIES]
        emit_data("decompose", {"sub_queries": sub_queries})
        return {"sub_queries": sub_queries, "usage": [usage]}

    return decompose


def fan_out(state: AgentState) -> list[Send]:
    """Aresta condicional: um retrieve por sub-pergunta, em paralelo. A pergunta original também é buscada."""
    queries = [state["query"], *state.get("sub_queries", [])]
    return [Send("retrieve_sub", {"sub_query": q}) for q in queries]


def make_retrieve_sub(pipeline: PipelineConfig):
    def retrieve_sub(state: dict) -> dict:
        query = state["sub_query"]
        hits = search(query, pipeline)
        emit_data("retrieve_sub", {"sub_query": query, **retrieve_payload(hits[: pipeline.top_k])})
        return {"sub_results": [hits], "usage": [embedding_usage(query)]}

    return retrieve_sub


def merge(results: list[list[ScoredChunk]], limit: int) -> list[ScoredChunk]:
    """Junta as listas por RRF: um trecho bem colocado em mais de uma busca sobe."""
    by_id: dict[str, ScoredChunk] = {}
    for hits in results:
        for chunk in hits:
            by_id.setdefault(chunk.id, chunk)
    order = rrf_fuse([[c.id for c in hits] for hits in results])
    return [by_id[i].model_copy(update={"rrf_rank": n}) for n, i in enumerate(order[:limit], start=1)]


def make_merge_results(pipeline: PipelineConfig):
    def merge_results(state: AgentState) -> dict:
        # Cada sub-busca traz até candidates_k; o rerank recebe um pouco mais do que numa busca simples.
        candidates = merge(state.get("sub_results", []), pipeline.candidates_k * 2)
        emit_data("merge_results", {"chunks": len(candidates), "searches": len(state.get("sub_results", []))})
        return {"candidates": candidates, "context": candidates[: pipeline.top_k]}

    return merge_results


def make_grade_docs(pipeline: PipelineConfig):
    def grade_docs(state: AgentState) -> dict:
        human = f"Passages:\n\n{format_context(state.get('context', []))}\n\nQuestion: {state['query']}"
        grade, usage = call_structured("small", DocsGrade, GRADE_PROMPT, human, "grade_docs")
        rewritten = grade.rewritten_query.strip()
        # Guardrails do laço: número máximo de tentativas e orçamento do request.
        can_retry = (
            not grade.sufficient
            and bool(rewritten)
            and state.get("retries", 0) < get_settings().max_retries
            and spent_usd(state) < budget_usd(pipeline)
        )
        action = "retry" if can_retry else "generate"
        emit_data("grade_docs", {**grade.model_dump(), "next_action": action, "retries": state.get("retries", 0)})
        return {
            "docs_sufficient": grade.sufficient,
            "rewritten_query": rewritten,
            "next_action": action,
            "usage": [usage],
        }

    return grade_docs


def make_retry_retrieve(pipeline: PipelineConfig):
    """Nova busca com a query reescrita. Os trechos novos entram junto dos anteriores para o rerank."""

    def retry_retrieve(state: AgentState) -> dict:
        query = state.get("rewritten_query") or state["query"]
        hits = search(query, pipeline)
        previous = state.get("candidates", [])
        candidates = merge([hits, previous], pipeline.candidates_k * 2)
        emit_data("retry_retrieve", {"sub_query": query, **retrieve_payload(hits[: pipeline.top_k])})
        return {
            "candidates": candidates,
            "context": candidates[: pipeline.top_k],
            "retries": state.get("retries", 0) + 1,
            "usage": [embedding_usage(query)],
        }

    return retry_retrieve
