"""Conexão com o Postgres (pool único, com o tipo vector registrado)."""

from __future__ import annotations

from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import get_settings


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


@lru_cache
def get_pool() -> ConnectionPool:
    return ConnectionPool(
        get_settings().database_url, min_size=1, max_size=8, configure=_configure, open=True
    )


def schema_sql(embedding_dim: int) -> str:
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS chunks (
        id           TEXT PRIMARY KEY,
        doc_id       TEXT NOT NULL,
        ord          INT  NOT NULL,
        title        TEXT NOT NULL,
        section      TEXT NOT NULL DEFAULT '',
        url          TEXT NOT NULL,
        text         TEXT NOT NULL,
        token_count  INT  NOT NULL,
        content_hash TEXT NOT NULL,
        embedding    vector({embedding_dim}) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
    CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
    """
