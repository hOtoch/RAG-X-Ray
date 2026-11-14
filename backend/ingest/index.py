"""Gera embeddings e grava os chunks no pgvector. Rodar de novo só reprocessa o que mudou."""

from __future__ import annotations

import hashlib

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.db import schema_sql
from app.llm.factory import get_embeddings
from app.llm.usage import cost_usd
from ingest.chunk import Chunk, count_tokens

BATCH = 100


def content_hash(chunk: Chunk, model: str) -> str:
    return hashlib.sha256(f"{model}\n{chunk.embedding_text}".encode()).hexdigest()


def index(chunks: list[Chunk]) -> None:
    s = get_settings()
    embeddings = get_embeddings()
    with psycopg.connect(s.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(schema_sql(s.embedding_dim))
        existing = dict(conn.execute("SELECT id, content_hash FROM chunks").fetchall())

        hashes = {c.id: content_hash(c, s.embedding_model) for c in chunks}
        todo = [c for c in chunks if existing.get(c.id) != hashes[c.id]]
        stale = set(existing) - set(hashes)
        tokens = 0

        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            texts = [c.embedding_text for c in batch]
            tokens += sum(count_tokens(t) for t in texts)
            vectors = embeddings.embed_documents(texts)
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                        (id, doc_id, ord, title, section, url, text, token_count, content_hash, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id, ord = EXCLUDED.ord, title = EXCLUDED.title,
                        section = EXCLUDED.section, url = EXCLUDED.url, text = EXCLUDED.text,
                        token_count = EXCLUDED.token_count, content_hash = EXCLUDED.content_hash,
                        embedding = EXCLUDED.embedding
                    """,
                    [
                        (
                            c.id,
                            c.doc_id,
                            c.ord,
                            c.title,
                            c.section,
                            c.url,
                            c.text,
                            c.token_count,
                            hashes[c.id],
                            np.array(v, dtype=np.float32),
                        )
                        for c, v in zip(batch, vectors, strict=True)
                    ],
                )
            conn.commit()
            print(f"index: {min(start + BATCH, len(todo))}/{len(todo)}", end="\r")

        if stale:
            conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (list(stale),))
        conn.commit()
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    cost = cost_usd(s.embedding_model, tokens) if tokens else 0.0
    print(
        f"index: {len(todo)} chunks embeddados (~{tokens} tokens, ~US$ {cost:.4f}), "
        f"{len(stale)} removidos, {total} na tabela"
    )
