"""Cache em duas camadas no Redis.

- Exato: hash da pergunta normalizada. Só acerta perguntas iguais (a menos de maiúsculas, espaços e pontuação).
- Semântico: embedding da pergunta e busca do vizinho mais próximo entre as perguntas já respondidas.
  Acerta paráfrases, com o risco de servir a resposta de uma pergunta parecida mas diferente
  (falso positivo). O limiar θ é escolhido com dados (eval/cache_calibration.py).

As chaves incluem um "namespace" com a pipeline, o modelo de embedding e o commit do corpus, para nunca
servir uma resposta gerada com outra configuração.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache

import numpy as np
import redis

from app.config import get_settings

CORPUS_VERSION = "4db8ed56"
PUNCTUATION = re.compile(r"[^\w\s]")


def normalize(question: str) -> str:
    """Minúsculas, sem acento, sem pontuação e com espaços simples."""
    text = unicodedata.normalize("NFKD", question.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = PUNCTUATION.sub(" ", text)
    return " ".join(text.split())


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url)


def namespace(pipeline: str) -> str:
    s = get_settings()
    return f"ragxray:cache:{pipeline}:{s.embedding_model}:{CORPUS_VERSION}"


def exact_key(pipeline: str, question: str) -> str:
    digest = hashlib.sha256(normalize(question).encode()).hexdigest()[:32]
    return f"{namespace(pipeline)}:exact:{digest}"


class SemanticCache:
    """Vetores das perguntas respondidas, guardados num hash do Redis e buscados em memória.

    Com poucos milhares de entradas, a busca por força bruta em numpy leva menos de 1 ms e não depende
    do módulo de busca vetorial do Redis. O RedisVL fica como opção se o cache crescer muito (ADR-0007).
    """

    def __init__(self, client: redis.Redis, pipeline: str):
        self.client = client
        self.key = f"{namespace(pipeline)}:semantic"

    def add(self, entry_id: str, vector: list[float]) -> None:
        self.client.hset(self.key, entry_id, np.asarray(vector, dtype=np.float32).tobytes())
        ttl = get_settings().cache_ttl_seconds
        if ttl:
            self.client.expire(self.key, ttl)

    def nearest(self, vector: list[float]) -> tuple[str | None, float | None]:
        stored = self.client.hgetall(self.key)
        if not stored:
            return None, None
        ids = [k.decode() for k in stored]
        matrix = np.stack([np.frombuffer(v, dtype=np.float32) for v in stored.values()])
        query = np.asarray(vector, dtype=np.float32)
        sims = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12)
        best = int(np.argmax(sims))
        return ids[best], float(sims[best])


def get_entry(pipeline: str, entry_id: str) -> dict | None:
    raw = get_redis().get(f"{namespace(pipeline)}:entry:{entry_id}")
    return json.loads(raw) if raw else None


def put_entry(pipeline: str, question: str, vector: list[float], entry: dict) -> str:
    """Grava a resposta e indexa a pergunta nos dois caches. Devolve o id da entrada."""
    client = get_redis()
    entry_id = exact_key(pipeline, question).rsplit(":", 1)[-1]
    ttl = get_settings().cache_ttl_seconds or None
    payload = json.dumps(entry | {"question": question})
    client.set(f"{namespace(pipeline)}:entry:{entry_id}", payload, ex=ttl)
    client.set(exact_key(pipeline, question), entry_id, ex=ttl)
    SemanticCache(client, pipeline).add(entry_id, vector)
    return entry_id


def lookup_exact(pipeline: str, question: str) -> dict | None:
    entry_id = get_redis().get(exact_key(pipeline, question))
    return get_entry(pipeline, entry_id.decode()) if entry_id else None


def lookup_semantic(pipeline: str, vector: list[float], threshold: float) -> tuple[dict | None, float | None]:
    entry_id, similarity = SemanticCache(get_redis(), pipeline).nearest(vector)
    if entry_id is None or similarity is None or similarity <= threshold:
        return None, similarity
    return get_entry(pipeline, entry_id), similarity


def clear(pipeline: str) -> int:
    client = get_redis()
    keys = list(client.scan_iter(f"{namespace(pipeline)}:*"))
    return client.delete(*keys) if keys else 0
