"""Execução de uma pergunta no grafo. A API (JSON e SSE) e o eval usam o mesmo caminho."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

from pydantic import BaseModel

from app import tracing
from app.config import get_settings
from app.events.schema import Event, EventFactory
from app.graph.build import get_graph
from app.graph.nodes.generate import FINAL_ANSWER_TAG
from app.graph.state import Citation, ScoredChunk
from app.llm.usage import NodeUsage, cost_usd
from app.pipeline import load_pipeline


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
    counterfactual_cost_usd: float
    total_latency_ms: float
    route: str | None = None
    route_reason: str | None = None
    grounded: bool | None = None
    gave_up: bool = False
    escalations: int = 0
    retries: int = 0
    cache_hit: str | None = None


def counterfactual_cost(usage: list[NodeUsage]) -> float:
    """Quanto o request custaria se toda chamada de LLM usasse o modelo large, com os mesmos tokens.
    Chamadas que não são de LLM (embedding, rerank) entram com o custo real."""
    s = get_settings()
    chat_models = {s.llm_model_small, s.llm_model_medium, s.llm_model_large}
    total = 0.0
    for u in usage:
        if u.model in chat_models:
            total += cost_usd(s.llm_model_large, u.tokens_in, u.tokens_out, u.tokens_cached)
        else:
            total += u.cost_usd
    return total


class QueryRun:
    """Uma execução do grafo. Itere `events()` até o fim e leia `result`."""

    def __init__(self, question: str, pipeline: str = "v0", request_id: str | None = None):
        load_pipeline(pipeline)  # falha cedo (KeyError) se a pipeline não existe
        self.question = question
        self.pipeline = pipeline
        self.request_id = request_id or str(uuid.uuid4())
        self.result: QueryResult | None = None

    def events(self) -> Iterator[Event]:
        factory = EventFactory(self.request_id)
        start = time.perf_counter()
        state: dict = {}
        graph = get_graph(self.pipeline)
        inputs = {"request_id": self.request_id, "query": self.question, "usage": []}
        config = tracing.trace_config(self.request_id, self.pipeline, self.question)
        try:
            for mode, chunk in graph.stream(inputs, config=config, stream_mode=["custom", "messages", "values"]):
                if mode == "custom":
                    yield factory.make(chunk["type"], chunk.get("node"), chunk.get("payload"))
                elif mode == "messages":
                    message, metadata = chunk
                    text = message.text if isinstance(getattr(message, "text", None), str) else ""
                    if text and FINAL_ANSWER_TAG in (metadata.get("tags") or []):
                        yield factory.make("token", metadata.get("langgraph_node"), {"text": text})
                elif mode == "values":
                    state = chunk
        except Exception as error:
            yield factory.make("error", None, {"message": f"{type(error).__name__}: {error}"})
            raise

        usage = state.get("usage", [])
        self.result = QueryResult(
            request_id=self.request_id,
            pipeline=self.pipeline,
            question=self.question,
            answer=state.get("answer") or "",
            citations=state.get("citations", []),
            context=state.get("context", []),
            candidates=state.get("candidates", []),
            model_tier_used=state.get("model_tier_used"),
            usage=usage,
            total_cost_usd=sum(u.cost_usd for u in usage),
            counterfactual_cost_usd=counterfactual_cost(usage),
            route=state.get("route"),
            route_reason=state.get("route_reason"),
            grounded=state.get("grounded"),
            gave_up=bool(state.get("gave_up")),
            escalations=state.get("escalations", 0),
            retries=state.get("retries", 0),
            cache_hit=state.get("cache_hit"),
            total_latency_ms=(time.perf_counter() - start) * 1000,
        )
        yield factory.make(
            "done",
            None,
            {
                "answer": self.result.answer,
                "citations": [c.model_dump() for c in self.result.citations],
                "total_cost_usd": self.result.total_cost_usd,
                "total_latency_ms": self.result.total_latency_ms,
                "counterfactual_cost_usd": self.result.counterfactual_cost_usd,
            },
        )


def run_query(question: str, pipeline: str = "v0") -> QueryResult:
    run = QueryRun(question, pipeline)
    for _ in run.events():
        pass
    assert run.result is not None
    return run.result
