"""Contagem de tokens e custo por chamada de modelo."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.config import ModelPrice, get_prices


class NodeUsage(BaseModel):
    """Uso de um nó do grafo: tokens, custo e latência."""

    node: str
    model: str | None = None
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


def price_of(model: str, prices: dict[str, ModelPrice] | None = None) -> ModelPrice:
    prices = prices if prices is not None else get_prices()
    if model not in prices:
        raise KeyError(f"Modelo '{model}' sem preço em configs/prices.yaml")
    return prices[model]


def cost_usd(
    model: str,
    tokens_in: int,
    tokens_out: int = 0,
    tokens_cached: int = 0,
    prices: dict[str, ModelPrice] | None = None,
) -> float:
    """Custo em USD. `tokens_in` inclui os tokens lidos do cache do provedor."""
    p = price_of(model, prices)
    cached_price = p.cached_input if p.cached_input is not None else p.input
    fresh = max(tokens_in - tokens_cached, 0)
    return (fresh * p.input + tokens_cached * cached_price + tokens_out * p.output) / 1_000_000


def usage_from_message(node: str, model: str, message: Any, latency_ms: float = 0.0) -> NodeUsage:
    """Extrai o uso de uma AIMessage do LangChain (campo `usage_metadata`)."""
    meta = getattr(message, "usage_metadata", None) or {}
    tokens_in = int(meta.get("input_tokens", 0))
    tokens_out = int(meta.get("output_tokens", 0))
    cached = int((meta.get("input_token_details") or {}).get("cache_read", 0) or 0)
    return NodeUsage(
        node=node,
        model=model,
        tokens_in=tokens_in,
        tokens_cached=cached,
        tokens_out=tokens_out,
        cost_usd=cost_usd(model, tokens_in, tokens_out, cached),
        latency_ms=latency_ms,
    )
