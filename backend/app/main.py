"""API do RAG X-Ray."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import __version__, store, tracing
from app.events.schema import Event
from app.service import QueryResult, QueryRun

log = logging.getLogger("ragxray")
app = FastAPI(title="RAG X-Ray", version=__version__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    pipeline: str = "v0"


def start_run(request: QueryRequest) -> QueryRun:
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
            store.save_events(run.request_id, run.question, run.pipeline, events)
            tracing.flush()
            if run.result is not None:
                tracing.record_totals(run.request_id, run.result.total_cost_usd, run.result.total_latency_ms)
        except Exception:  # falha ao gravar não pode derrubar a resposta
            log.exception("falha ao gravar eventos do request %s", run.request_id)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/query")
def query(request: QueryRequest) -> QueryResult:
    """Versão síncrona: devolve a resposta completa em JSON."""
    run = start_run(request)
    for _ in recorded(run):
        pass
    assert run.result is not None
    return run.result


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """Versão em streaming (SSE): um evento por etapa, conforme docs/events.schema.json."""
    run = start_run(request)

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
