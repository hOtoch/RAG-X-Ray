"""API do RAG X-Ray."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import __version__, guard, store, tracing
from app.config import get_settings
from app.events.schema import Event
from app.graph.build import get_graph
from app.pipeline import load_pipelines
from app.service import QueryResult, QueryRun

log = logging.getLogger("ragxray")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Carrega índice BM25 e reranker antes da primeira pergunta (se o banco estiver no ar)."""
    from app.graph.build import warmup

    for pipeline in load_pipelines().values():
        try:
            warmup(pipeline)
        except Exception:
            log.warning("warmup da pipeline %s falhou", pipeline.name)
    yield


app = FastAPI(title="RAG X-Ray", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    pipeline: str = "v0"


def start_run(request: QueryRequest, http: Request) -> QueryRun:
    guard.check_daily_budget()
    guard.check_rate_limit(http)
    try:
        return QueryRun(request.question, request.pipeline)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def recorded(run: QueryRun) -> Iterator[Event]:
    """Repassa os eventos e, ao final (mesmo com erro), grava o stream e envia o trace."""
    events: list[Event] = []
    try:
        for event in run.events():
            events.append(event)
            yield event
    finally:
        try:
            if run.result is not None:
                guard.record_spend(run.result.total_cost_usd)
            store.save_events(run.request_id, run.question, run.pipeline, events)
            tracing.flush()
            if run.result is not None:
                tracing.record_totals(run.request_id, run.result.total_cost_usd, run.result.total_latency_ms)
        except Exception:  # falha ao gravar não pode derrubar a resposta
            log.exception("falha ao gravar eventos do request %s", run.request_id)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/budget")
def budget() -> dict:
    """Gasto do dia, para a interface avisar quando o modo ao vivo vai ser desligado."""
    s = get_settings()
    return {
        "spent_today_usd": round(guard.spent_today_usd(), 4),
        "daily_budget_usd": s.daily_budget_usd,
        "live_mode_enabled": s.live_mode_enabled,
    }


class FeedbackRequest(BaseModel):
    request_id: str
    value: int = Field(ge=-1, le=1, description="1 = positivo, -1 = negativo")
    comment: str | None = Field(default=None, max_length=1000)


@app.post("/feedback")
def feedback(body: FeedbackRequest) -> dict:
    """Feedback do usuário, gravado como score no trace do Langfuse (eval online)."""
    if not store.load_events(body.request_id):
        raise HTTPException(status_code=404, detail="request não encontrado")
    try:
        tracing.create_score(body.request_id, "user_feedback", float(body.value), body.comment)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"falha ao gravar no Langfuse: {error}") from error
    store.save_feedback(body.request_id, body.value, body.comment)
    return {"ok": True}


@app.post("/query")
def query(request: QueryRequest, http: Request) -> QueryResult:
    """Versão síncrona: devolve a resposta completa em JSON."""
    run = start_run(request, http)
    for _ in recorded(run):
        pass
    assert run.result is not None
    return run.result


@app.post("/query/stream")
def query_stream(request: QueryRequest, http: Request) -> StreamingResponse:
    """Versão em streaming (SSE): um evento por etapa, conforme docs/events.schema.json."""
    run = start_run(request, http)

    def body() -> Iterator[str]:
        try:
            for event in recorded(run):
                yield event.sse()
        except Exception:
            log.exception("erro no request %s", run.request_id)  # o evento `error` já foi enviado

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-Id": run.request_id}
    return StreamingResponse(body(), media_type="text/event-stream", headers=headers)


@app.get("/traces/{request_id}")
def trace(request_id: str) -> list[dict]:
    """Eventos gravados de um request, em ordem."""
    events = store.load_events(request_id)
    if not events:
        raise HTTPException(status_code=404, detail="request não encontrado")
    return events


@app.get("/pipelines")
def pipelines() -> list[dict]:
    """Versões do pipeline disponíveis, com as features ligadas em cada uma."""
    return [p.model_dump() for p in load_pipelines().values()]


@app.get("/graph/{pipeline}")
def graph(pipeline: str) -> dict:
    """Topologia real do grafo compilado. Os ids dos nós são os mesmos do campo `node` dos eventos."""
    try:
        drawable = get_graph(pipeline).get_graph()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    nodes = [{"id": n, "kind": "terminal" if n.startswith("__") else "node"} for n in drawable.nodes]
    edges = [
        {"source": e.source, "target": e.target, "conditional": bool(e.conditional), "label": e.data or None}
        for e in drawable.edges
    ]
    return {"pipeline": pipeline, "nodes": nodes, "edges": edges}


@app.get("/chunks/{chunk_id:path}")
def chunk(chunk_id: str) -> dict:
    """Texto completo de um chunk, para abrir a citação na interface."""
    from app.db import get_pool

    with get_pool().connection() as conn:
        row = conn.execute(
            "SELECT id, doc_id, title, section, url, text FROM chunks WHERE id = %s", (chunk_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="chunk não encontrado")
    return dict(zip(("id", "doc_id", "title", "section", "url", "text"), row, strict=True))
