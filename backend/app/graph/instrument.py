"""Instrumentação dos nós: cada execução emite node_start/node_end e registra um NodeUsage."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer

from app import tracing
from app.llm.usage import NodeUsage


def _write(event: dict[str, Any]) -> None:
    """Envia um evento para o stream `custom` do LangGraph. Fora de uma execução do grafo, não faz nada."""
    try:
        writer = get_stream_writer()
    except Exception:
        return
    writer(event)


def emit_data(node: str, payload: dict[str, Any]) -> None:
    """Evento `data`: dados específicos do nó para a interface (ranks, rota, similaridade...)."""
    _write({"type": "data", "node": node, "payload": payload})


def instrumented(name: str, fn: Callable[[dict], dict | None]) -> Callable[[dict], dict]:
    """Envolve um nó. O nó devolve os usos das chamadas de modelo que fez; aqui eles viram um
    único NodeUsage com a latência total do nó."""

    def node(state: dict) -> dict:
        _write({"type": "node_start", "node": name})
        start = time.perf_counter()
        result = dict(fn(state) or {})
        calls: list[NodeUsage] = result.get("usage", [])
        usage = NodeUsage(
            node=name,
            model=next((u.model for u in reversed(calls) if u.model), None),
            tokens_in=sum(u.tokens_in for u in calls),
            tokens_cached=sum(u.tokens_cached for u in calls),
            tokens_out=sum(u.tokens_out for u in calls),
            cost_usd=sum(u.cost_usd for u in calls),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        result["usage"] = [usage]
        tracing.annotate_node(usage.model_dump(exclude={"node"}))
        _write({"type": "node_end", "node": name, "payload": usage.model_dump(exclude={"node"})})
        return result

    node.__name__ = name
    return node
