"""API do RAG X-Ray."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.service import QueryResult, run_query

app = FastAPI(title="RAG X-Ray", version=__version__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    pipeline: str = "v0"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/query")
def query(request: QueryRequest) -> QueryResult:
    """Versão síncrona: devolve a resposta completa. O streaming SSE entra na F2."""
    try:
        return run_query(request.question, request.pipeline)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
