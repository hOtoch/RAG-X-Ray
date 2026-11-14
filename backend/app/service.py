"""Execução de uma pergunta no grafo. A API (JSON e SSE) e o eval usam o mesmo caminho."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

from pydantic import BaseModel

from app import tracing
from app.events.schema import Event, EventFactory
from app.graph.build import get_graph
from app.graph.nodes.generate import FINAL_ANSWER_TAG
from app.graph.state import Citation, ScoredChunk
from app.llm.usage import NodeUsage
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
    total_latency_ms: float


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
            },
        )


def run_query(question: str, pipeline: str = "v0") -> QueryResult:
    run = QueryRun(question, pipeline)
    for _ in run.events():
        pass
    assert run.result is not None
    return run.result
