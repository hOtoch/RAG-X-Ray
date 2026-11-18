"""Proteções do demo público: rate limit por IP e teto de gasto diário em USD (contadores no Redis)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import HTTPException, Request

from app.cache import get_redis
from app.config import get_settings


def client_ip(request: Request) -> str:
    # Atrás de proxy (Fly, Render, Railway), o IP real vem no primeiro valor de X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    """Janela fixa de 60 s por IP."""
    limit = get_settings().rate_limit_per_minute
    if not limit:
        return
    window = int(time.time() // 60)
    key = f"ragxray:rate:{client_ip(request)}:{window}"
    client = get_redis()
    count = client.incr(key)
    if count == 1:
        client.expire(key, 70)
    if count > limit:
        raise HTTPException(
            status_code=429,
            detail=f"Limite de {limit} perguntas por minuto atingido. Tente de novo em instantes ou use o modo replay.",
        )


def _spend_key() -> str:
    return f"ragxray:spend:{datetime.now(UTC).date().isoformat()}"


def spent_today_usd() -> float:
    value = get_redis().get(_spend_key())
    return float(value) if value else 0.0


def check_daily_budget() -> None:
    budget = get_settings().daily_budget_usd
    if not get_settings().live_mode_enabled:
        raise HTTPException(status_code=503, detail="O modo ao vivo está desligado neste deploy. Use o modo replay.")
    if budget and spent_today_usd() >= budget:
        raise HTTPException(
            status_code=429,
            detail="O teto de gasto diário do demo foi atingido. O modo replay continua disponível.",
        )


def record_spend(cost_usd: float) -> None:
    if cost_usd <= 0:
        return
    client = get_redis()
    key = _spend_key()
    client.incrbyfloat(key, cost_usd)
    client.expire(key, 3 * 24 * 3600)
