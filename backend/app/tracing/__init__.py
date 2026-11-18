"""Tracing com Langfuse. Fica desligado se LANGFUSE_ENABLED for falso ou se as chaves não existirem.

Cada request vira um trace cujo id é derivado do `request_id`, o mesmo id do stream SSE. Assim dá
para ir do evento na interface ao trace, e anexar feedback ao trace só com o `request_id`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx

from app.config import get_settings


@lru_cache
def _client() -> Any | None:
    s = get_settings()
    if not (s.langfuse_enabled and s.langfuse_public_key and s.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)


def enabled() -> bool:
    return _client() is not None


def trace_id_for(request_id: str) -> str | None:
    client = _client()
    return client.create_trace_id(seed=request_id) if client else None


def trace_config(request_id: str, pipeline: str, question: str) -> dict[str, Any]:
    """Config do LangGraph para um request."""
    config: dict[str, Any] = {
        "run_name": "rag-xray-query",
        "metadata": {"request_id": request_id, "pipeline": pipeline},
    }
    if enabled():
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler(trace_context={"trace_id": trace_id_for(request_id)})]
        # chaves reconhecidas pelo handler do Langfuse
        config["metadata"] |= {"langfuse_session_id": request_id, "langfuse_tags": [f"pipeline:{pipeline}"]}
    return config


def genai_attributes(usage: dict[str, Any]) -> dict[str, Any]:
    """Atributos do nó nas convenções semânticas GenAI do OpenTelemetry (gen_ai.*), mais os nossos (ragxray.*).

    Nós sem chamada de modelo (rerank local, compressão, cache) só levam os atributos ragxray.*. O retrieve
    chama o modelo de embeddings, então a operação é `embeddings`; os demais nós com modelo são `chat`.
    """
    s = get_settings()
    attrs: dict[str, Any] = {
        "ragxray.latency_ms": round(float(usage.get("latency_ms", 0.0)), 1),
        "ragxray.cost_usd": float(usage.get("cost_usd", 0.0)),
        "ragxray.tokens_cached": int(usage.get("tokens_cached", 0)),
    }
    model = usage.get("model")
    if model:
        embeddings = model == s.embedding_model
        attrs |= {
            "gen_ai.operation.name": "embeddings" if embeddings else "chat",
            "gen_ai.provider.name": s.embedding_provider if embeddings else s.llm_provider,
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": int(usage.get("tokens_in", 0)),
        }
        if not embeddings:
            attrs["gen_ai.usage.output_tokens"] = int(usage.get("tokens_out", 0))
    return attrs


def annotate_node(usage: dict[str, Any]) -> None:
    """Grava tokens, custo e modelo do nó no span atual (o span do nó, criado pelo handler): como metadata
    no Langfuse e como atributos OpenTelemetry, para qualquer backend OTel ler sem conhecer o projeto."""
    client = _client()
    if client is None:
        return
    attrs = genai_attributes(usage)
    try:
        client.update_current_span(metadata={"usage": usage, "otel": attrs})
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attributes(attrs)
    except Exception:  # tracing nunca pode derrubar um request
        pass


def create_score(request_id: str, name: str, value: float, comment: str | None = None) -> None:
    """Anexa um score ao trace do request, pela API REST do Langfuse."""
    if not enabled():
        return
    s = get_settings()
    body = {"traceId": trace_id_for(request_id), "name": name, "value": value, "dataType": "NUMERIC"}
    if comment:
        body["comment"] = comment
    response = httpx.post(
        f"{s.langfuse_host}/api/public/scores",
        json=body,
        auth=(s.langfuse_public_key, s.langfuse_secret_key),
        timeout=10,
    )
    response.raise_for_status()


def record_totals(request_id: str, total_cost_usd: float, total_latency_ms: float) -> None:
    """Custo e latência medidos por nós, como scores do trace."""
    try:
        create_score(request_id, "total_cost_usd", total_cost_usd)
        create_score(request_id, "total_latency_ms", total_latency_ms)
    except Exception:  # tracing nunca pode derrubar um request
        pass


def flush() -> None:
    client = _client()
    if client is not None:
        client.flush()
