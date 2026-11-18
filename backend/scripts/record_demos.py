"""Grava os traces do modo replay em frontend/public/demos/.

Cada demo vira um arquivo com a topologia do grafo e todos os eventos do request. O frontend reproduz
esses arquivos sem chamar o backend nem o LLM, então o demo público nunca quebra e não gera custo.

Uso (a partir de backend/, com a infra no ar):
    uv run python scripts/record_demos.py --pipeline v3
"""

from __future__ import annotations

import argparse
import json

from app import cache
from app.config import ROOT
from app.db import get_pool
from app.graph.build import get_graph, warmup
from app.pipeline import load_pipeline
from app.service import QueryRun

OUT = ROOT / "frontend" / "public" / "demos"

# (rótulo, pergunta, nota). A lista cobre todas as rotas, um acerto de cache e um "não sei".
DEMOS = [
    ("RAG simples", "How do I pause a LangGraph graph to wait for human approval and then resume it?", ""),
    (
        "multi-hop",
        "qual a diferença entre o padrão de subagents e o de handoffs no langchain?",
        "pergunta em português",
    ),
    ("multi-hop", "What's the difference between a checkpointer and a store in LangGraph?", ""),
    ("premissa falsa", "How do I set the max_parallel_sends parameter on StateGraph to limit Send fan-out?", ""),
    ("erro", "I get GRAPH_RECURSION_LIMIT in LangGraph. What does it mean and how do I fix it?", ""),
    ("integração", "How do I set up credentials for the MistralAIEmbeddings integration in LangChain?", ""),
    ("fora de escopo", "Qual é a melhor receita de bolo de cenoura?", "recusa sem chamar o modelo de geração"),
    ("conversa", "Oi! O que você sabe fazer?", "rota trivial, modelo small"),
    ("cache", "How do I pause a LangGraph graph to wait for human approval and then resume it?", "repetida: cache"),
]


def topology(pipeline: str) -> dict:
    drawable = get_graph(pipeline).get_graph()
    return {
        "pipeline": pipeline,
        "nodes": [{"id": n, "kind": "terminal" if n.startswith("__") else "node"} for n in drawable.nodes],
        "edges": [
            {"source": e.source, "target": e.target, "conditional": bool(e.conditional), "label": e.data or None}
            for e in drawable.edges
        ],
    }


def chunk_texts(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, doc_id, title, section, url, text FROM chunks WHERE id = ANY(%s)", (ids,)
        ).fetchall()
    return [dict(zip(("id", "doc_id", "title", "section", "url", "text"), r, strict=True)) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="v3")
    args = parser.parse_args()

    config = load_pipeline(args.pipeline)
    warmup(config)
    if config.cache:
        cache.clear(args.pipeline)  # começa vazio, para a demo "cache" ser o único acerto
    graph = topology(args.pipeline)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()

    index, total = [], 0.0
    for label, question, note in DEMOS:
        run = QueryRun(question, args.pipeline)
        events = [e.model_dump(mode="json") for e in run.events()]
        demo = {"request_id": run.request_id, "question": question, "pipeline": args.pipeline, "label": label,
                "note": note}  # fmt: skip
        cited = [c.chunk_id for c in run.result.citations]
        (OUT / f"{run.request_id}.json").write_text(
            json.dumps(
                {"demo": demo, "topology": graph, "events": events, "chunks": chunk_texts(cited)}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        index.append(demo)
        total += run.result.total_cost_usd
        r = run.result
        print(f"{label:<15} rota={r.route} cache={r.cache_hit} grounded={r.grounded} US$ {r.total_cost_usd:.5f}")
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(index)} demos em {OUT} (US$ {total:.4f})")


if __name__ == "__main__":
    main()
