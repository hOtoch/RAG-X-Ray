"""API do RAG X-Ray."""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__

app = FastAPI(title="RAG X-Ray", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
