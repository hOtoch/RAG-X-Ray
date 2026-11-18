"""Busca lexical. Dois backends, comparados no ADR-0005:

- `bm25`: BM25 Okapi em memória (rank_bm25), com o índice montado a partir da tabela de chunks.
- `fts`: full-text search do Postgres (tsvector + ts_rank_cd), que não é BM25 puro.
"""

from __future__ import annotations

import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi

from app.db import get_pool
from app.graph.state import ScoredChunk

COLUMNS = "id, doc_id, title, section, url, text, token_count"
TOKEN = re.compile(r"[a-z0-9_]+")
# Palavras muito comuns em perguntas: não ajudam a achar o trecho e só adicionam ruído ao score.
STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or is are was were be been do does did how what when "
    "where which who why can could should would i you it this that these those my your as if not no "
    "o os as um uma de do da dos das em no na nos nas para por com e ou que como qual quais quando onde "
    "eu voce meu minha se nao pra pro".split()
)


def tokenize(text: str) -> list[str]:
    """Minúsculas, mantém identificadores (add_conditional_edges) e também as partes deles."""
    tokens: list[str] = []
    for token in TOKEN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        tokens.append(token)
        if "_" in token:
            tokens += [part for part in token.split("_") if part and part not in STOPWORDS]
    return tokens


def _row_to_chunk(row: tuple, score: float, rank: int) -> ScoredChunk:
    return ScoredChunk(
        id=row[0], doc_id=row[1], title=row[2], section=row[3], url=row[4], text=row[5], token_count=row[6],
        score=score, bm25_rank=rank,
    )  # fmt: skip


class BM25Index:
    """Índice BM25 em memória de todos os chunks. Montado uma vez, na primeira busca."""

    def __init__(self, rows: list[tuple]):
        self.rows = rows
        self.bm25 = BM25Okapi([tokenize(f"{r[2]} {r[3]} {r[5]}") for r in rows])

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(-scores)[:k]
        return [
            _row_to_chunk(self.rows[i], float(scores[i]), rank) for rank, i in enumerate(top, start=1) if scores[i] > 0
        ]


_index: BM25Index | None = None
_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    global _index
    with _lock:
        if _index is None:
            with get_pool().connection() as conn:
                rows = conn.execute(f"SELECT {COLUMNS} FROM chunks ORDER BY id").fetchall()
            _index = BM25Index(rows)
    return _index


def bm25_search(query: str, k: int) -> list[ScoredChunk]:
    return get_bm25_index().search(query, k)


FTS_SCHEMA = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || section || ' ' || text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
"""


def fts_search(query: str, k: int) -> list[ScoredChunk]:
    """Termos da pergunta combinados com OR; com AND, perguntas longas não achariam nada."""
    with get_pool().connection() as conn:
        rows = conn.execute(
            f"""
            WITH q AS (
                SELECT to_tsquery('english', string_agg(lexeme, ' | ')) AS query
                FROM unnest(to_tsvector('english', %s))
            )
            SELECT {COLUMNS}, ts_rank_cd(tsv, q.query, 32) AS score
            FROM chunks, q
            WHERE q.query IS NOT NULL AND tsv @@ q.query
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, k),
        ).fetchall()
    return [_row_to_chunk(r, float(r[7]), rank) for rank, r in enumerate(rows, start=1)]
