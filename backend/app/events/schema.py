"""Contrato de eventos SSE (seção 6 do context.md).

O mesmo stream alimenta a interface e é gravado para o replay. Todo evento tem o mesmo envelope;
o `payload` muda conforme o `type`.

Para exportar o JSON Schema usado pelo frontend:
    uv run python -m app.events.schema
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.config import ROOT

EventType = Literal["node_start", "node_end", "token", "data", "error", "done"]


class NodeEndPayload(BaseModel):
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    cost_usd: float = 0.0
    model: str | None = None


class TokenPayload(BaseModel):
    text: str


class ErrorPayload(BaseModel):
    message: str


class CitationPayload(BaseModel):
    n: int
    chunk_id: str
    title: str
    section: str
    url: str


class DonePayload(BaseModel):
    answer: str
    citations: list[CitationPayload] = Field(default_factory=list)
    total_cost_usd: float
    total_latency_ms: float
    counterfactual_cost_usd: float | None = None


# --- payloads de `data`, um por nó ---


class RetrievedChunk(BaseModel):
    id: str
    title: str
    section: str
    url: str
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_rank: int | None = None


class RetrieveData(BaseModel):
    chunks: list[RetrievedChunk]


class RerankedChunk(BaseModel):
    id: str
    rerank_rank: int
    score: float


class RerankData(BaseModel):
    chunks: list[RerankedChunk]


class CompressData(BaseModel):
    chunks_before: int
    chunks_after: int
    tokens_before: int
    tokens_after: int
    budget: int | None = None


class RouterData(BaseModel):
    route: Literal["trivial", "out_of_scope", "simple_rag", "multi_hop"]
    reason: str
    confidence: float


class CacheSemanticData(BaseModel):
    hit: bool
    similarity: float | None = None
    threshold: float


class CacheExactData(BaseModel):
    hit: bool


class EmbeddingMapData(BaseModel):
    query_xy: tuple[float, float]
    hit_ids: list[str]


class GroundednessData(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class Event(BaseModel):
    """Envelope comum a todos os eventos."""

    request_id: str
    seq: int = Field(ge=0, description="Ordem monotônica dentro do request")
    ts: datetime
    node: str | None = Field(default=None, description="Id do nó no grafo; None em eventos do request inteiro")
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)

    def sse(self) -> str:
        """Formato text/event-stream."""
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"


PAYLOADS: dict[str, type[BaseModel]] = {
    "node_end": NodeEndPayload,
    "token": TokenPayload,
    "error": ErrorPayload,
    "done": DonePayload,
}

DATA_PAYLOADS: dict[str, type[BaseModel]] = {
    "retrieve": RetrieveData,
    "rerank": RerankData,
    "compress_context": CompressData,
    "router": RouterData,
    "cache_exact": CacheExactData,
    "cache_semantic": CacheSemanticData,
    "embedding_map": EmbeddingMapData,
    "groundedness_check": GroundednessData,
}


def validate_payload(event_type: str, node: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Valida o payload contra o modelo do tipo (ou do nó, em eventos `data`) e devolve o dict normalizado."""
    model = DATA_PAYLOADS.get(node or "") if event_type == "data" else PAYLOADS.get(event_type)
    return model(**payload).model_dump(mode="json") if model else payload


class EventFactory:
    """Cria eventos de um request com `seq` crescente."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._seq = 0

    def make(self, event_type: EventType, node: str | None = None, payload: dict[str, Any] | None = None) -> Event:
        event = Event(
            request_id=self.request_id,
            seq=self._seq,
            ts=datetime.now(UTC),
            node=node,
            type=event_type,
            payload=validate_payload(event_type, node, payload or {}),
        )
        self._seq += 1
        return event


def json_schema() -> dict[str, Any]:
    models: list[type[BaseModel]] = [Event, *PAYLOADS.values(), *DATA_PAYLOADS.values()]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RAG X-Ray SSE events",
        "event": TypeAdapter(Event).json_schema(),
        "payloads": {name: model.model_json_schema() for name, model in PAYLOADS.items()},
        "data_payloads": {name: model.model_json_schema() for name, model in DATA_PAYLOADS.items()},
        "models": sorted(m.__name__ for m in models),
    }


if __name__ == "__main__":
    out = ROOT / "docs" / "events.schema.json"
    out.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8")
    print(f"escrito {out}")
