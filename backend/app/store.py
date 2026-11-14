"""Persistência dos eventos de cada request (base do modo replay)."""

from __future__ import annotations

import json

from app.db import get_pool
from app.events.schema import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    pipeline   TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS events (
    request_id TEXT NOT NULL REFERENCES requests (request_id) ON DELETE CASCADE,
    seq        INT  NOT NULL,
    event      JSONB NOT NULL,
    PRIMARY KEY (request_id, seq)
);
"""

_ready = False


def ensure_schema() -> None:
    global _ready
    if not _ready:
        with get_pool().connection() as conn:
            conn.execute(SCHEMA)
        _ready = True


def save_events(request_id: str, question: str, pipeline: str, events: list[Event]) -> None:
    ensure_schema()
    with get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO requests (request_id, question, pipeline) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (request_id, question, pipeline),
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO events (request_id, seq, event) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                [(request_id, e.seq, e.model_dump_json()) for e in events],
            )


def load_events(request_id: str) -> list[dict]:
    ensure_schema()
    with get_pool().connection() as conn:
        rows = conn.execute("SELECT event FROM events WHERE request_id = %s ORDER BY seq", (request_id,)).fetchall()
    return [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]
