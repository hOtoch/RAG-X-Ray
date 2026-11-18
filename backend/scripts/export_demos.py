"""Exporta requests já gravados no Postgres como demos do modo replay (sem chamar o LLM).

Uso (a partir de backend/):
    uv run python scripts/export_demos.py            # exporta todos os requests completos
    uv run python scripts/export_demos.py <id> <id>  # só os indicados

Complementa record_demos.py: este reaproveita o que já rodou; aquele grava perguntas novas.
"""

from __future__ import annotations

import json
import sys

from app.db import get_pool
from app.store import load_events
from scripts.record_demos import OUT, chunk_texts, topology


def main(ids: list[str]) -> None:
    with get_pool().connection() as conn:
        rows = conn.execute("SELECT request_id, question, pipeline FROM requests ORDER BY created_at").fetchall()
    OUT.mkdir(parents=True, exist_ok=True)
    index_file = OUT / "index.json"
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else []
    known = {d["request_id"] for d in index}
    for request_id, question, pipeline in rows:
        if (ids and request_id not in ids) or request_id in known:
            continue
        events = load_events(request_id)
        if not events or events[-1]["type"] != "done":
            continue  # só requests que terminaram
        cited = [c["chunk_id"] for c in events[-1]["payload"].get("citations", [])]
        demo = {"request_id": request_id, "question": question, "pipeline": pipeline, "label": pipeline,
                "note": "gravado de uma execução real"}  # fmt: skip
        payload = {"demo": demo, "topology": topology(pipeline), "events": events, "chunks": chunk_texts(cited)}
        (OUT / f"{request_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        index.append(demo)
        print(f"exportado {request_id} ({len(events)} eventos): {question}")
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
