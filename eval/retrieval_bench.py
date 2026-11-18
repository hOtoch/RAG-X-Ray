"""Compara estratégias de retrieval sem gerar resposta (rápido e quase sem custo).

Uso (a partir da raiz do repositório):
    uv run --project backend python -m eval.retrieval_bench
    uv run --project backend python -m eval.retrieval_bench --rerankers Xenova/ms-marco-MiniLM-L-6-v2

Só as perguntas com gabarito de retrieval entram. Os embeddings das perguntas ficam em cache em
data/query_embeddings.json, então rodar de novo não chama a API. O resultado vai para
eval/results/retrieval_bench.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from app.config import ROOT, get_settings
from app.graph.state import ScoredChunk
from app.llm.factory import get_embeddings
from app.retrieval import dense, lexical
from app.retrieval.hybrid import fuse
from eval.dataset import dataset_hash, load_golden
from eval.metrics.retrieval import doc_level, mrr, ndcg_at_k, recall_at_k

OUT = Path(__file__).parent / "results" / "retrieval_bench.json"
CACHE = ROOT / "data" / "query_embeddings.json"
K = 5
CANDIDATES = 20

Strategy = Callable[[str, list[float]], list[ScoredChunk]]


def query_vectors(questions: list[str]) -> dict[str, list[float]]:
    model = get_settings().embedding_model
    cache: dict[str, list[float]] = {}
    if CACHE.exists():
        stored = json.loads(CACHE.read_text(encoding="utf-8"))
        cache = stored["vectors"] if stored.get("model") == model else {}
    missing = [q for q in questions if q not in cache]
    if missing:
        for q, vector in zip(missing, get_embeddings().embed_documents(missing), strict=True):
            cache[q] = vector
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"model": model, "vectors": cache}), encoding="utf-8")
    return cache


def evaluate(name: str, strategy: Strategy, items, vectors) -> dict:
    rows, latencies = [], []
    for item in items:
        start = time.perf_counter()
        retrieved = [c.id for c in strategy(item.question, vectors[item.question])]
        latencies.append((time.perf_counter() - start) * 1000)
        relevant = item.relevant_doc_ids
        rows.append(
            {
                f"recall@{K}": recall_at_k(retrieved, relevant, K),
                f"doc_recall@{K}": recall_at_k(retrieved, doc_level(relevant), K),
                f"recall@{CANDIDATES}": recall_at_k(retrieved, relevant, CANDIDATES),
                "mrr": mrr(retrieved, relevant),
                f"ndcg@{K}": ndcg_at_k(retrieved, relevant, K),
            }
        )
    result = {key: round(statistics.fmean(r[key] for r in rows), 4) for key in rows[0]}
    result["latency_ms_p50"] = round(statistics.median(latencies), 1)
    result["latency_ms_p95"] = round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)
    print(f"{name:<34}" + "  ".join(f"{k}={v}" for k, v in result.items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerankers", nargs="*", default=[], help="Modelos de rerank a comparar")
    parser.add_argument("--only-rerankers", action="store_true", help="Pula as estratégias sem rerank")
    args = parser.parse_args()

    items = [i for i in load_golden() if i.relevant_doc_ids]
    vectors = query_vectors([i.question for i in items])
    with dense.get_pool().connection() as conn:
        conn.execute(lexical.FTS_SCHEMA)
    lexical.get_bm25_index()  # monta o índice antes de medir latência

    def dense_only(q: str, v: list[float]) -> list[ScoredChunk]:
        return dense.dense_search(v, CANDIDATES)

    def hybrid(lexical_search: Callable[[str, int], list[ScoredChunk]]) -> Strategy:
        return lambda q, v: fuse(dense.dense_search(v, CANDIDATES), lexical_search(q, CANDIDATES))[:CANDIDATES]

    strategies: dict[str, Strategy] = {
        "dense": dense_only,
        "bm25": lambda q, v: lexical.bm25_search(q, CANDIDATES),
        "fts": lambda q, v: lexical.fts_search(q, CANDIDATES),
        "hybrid (dense + bm25)": hybrid(lexical.bm25_search),
        "hybrid (dense + fts)": hybrid(lexical.fts_search),
    }
    for model in args.rerankers:
        from app.retrieval.reranker import rerank

        strategies[f"hybrid bm25 + rerank {model.split('/')[-1]}"] = lambda q, v, model=model: rerank(
            q, hybrid(lexical.bm25_search)(q, v), CANDIDATES, model
        )

    if args.only_rerankers:
        strategies = {name: s for name, s in strategies.items() if "rerank" in name}

    print(f"{len(items)} perguntas com gabarito | k={K} | candidatos por fonte={CANDIDATES}\n")
    for name, strategy in strategies.items():
        result = evaluate(name, strategy, items, vectors)
        # Salva a cada estratégia: o rerank em CPU é lento, e uma falha no meio não perde o que já rodou.
        previous = json.loads(OUT.read_text(encoding="utf-8"))["strategies"] if OUT.exists() else {}
        OUT.write_text(
            json.dumps(
                {
                    "dataset_hash": dataset_hash(),
                    "n_questions": len(items),
                    "embedding_model": get_settings().embedding_model,
                    "strategies": previous | {name: result},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"\nsalvo em {OUT}")


if __name__ == "__main__":
    main()
