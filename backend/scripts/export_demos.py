"""Exporta requests já gravados no Postgres como demos do modo replay (sem chamar o LLM).

Uso (a partir de backend/):
    uv run python scripts/export_demos.py            # exporta todos os requests completos
    uv run python scripts/export_demos.py <id> <id>  # só os indicados
    uv run python scripts/export_demos.py --refresh-maps  # acrescenta o mapa aos demos já exportados

Complementa record_demos.py: este reaproveita o que já rodou; aquele grava perguntas novas.
"""

from __future__ import annotations

import json
import sys

from app.db import get_pool
from app.retrieval.embedding_map import map_payload
from app.store import load_events
from scripts.record_demos import OUT, chunk_texts, topology


def with_map(events: list[dict]) -> list[dict]:
    """Recalcula o evento embedding_map depois de cada retrieve (o mapa muda quando a ingestão roda de novo)."""
    out: list[dict] = []
    for e in (e for e in events if e["node"] != "embedding_map"):
        out.append(e)
        if e["type"] == "data" and e["node"] == "retrieve" and (payload := map_payload(e["payload"]["chunks"])):
            out.append({**e, "node": "embedding_map", "payload": payload})
    for seq, e in enumerate(out):
        e["seq"] = seq
    return out


def refresh_maps() -> None:
    """Atualiza os demos já exportados com o evento embedding_map (sem chamar o LLM)."""
    for path in OUT.glob("*.json"):
        if path.name == "index.json":
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        trace["events"] = with_map(trace["events"])
        path.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
        print(f"mapa: {path.name}")


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
        events = with_map(load_events(request_id))
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
    if sys.argv[1:] == ["--refresh-maps"]:
        refresh_maps()
    else:
        main(sys.argv[1:])
