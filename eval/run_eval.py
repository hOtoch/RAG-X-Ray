"""Roda o golden dataset contra uma config de pipeline e salva o resultado.

Uso (a partir da raiz do repositório):
    uv run --project backend python -m eval.run_eval --pipeline v0
    uv run --project backend python -m eval.run_eval --pipeline v0 --subset 10

O resultado vai para eval/results/<pipeline>.json (ou <pipeline>.subset<N>.json). Se a execução cair,
rodar de novo retoma do ponto em que parou.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.graph.nodes.generate import format_context
from app.pipeline import load_pipeline
from app.service import run_query
from eval import judge as judge_module
from eval.dataset import GoldenItem, dataset_hash, load_golden
from eval.metrics.retrieval import doc_level, mrr, ndcg_at_k, recall_at_k

RESULTS = Path(__file__).parent / "results"
PARTIAL = RESULTS / ".partial"
K = 5


def stratified_subset(items: list[GoldenItem], n: int) -> list[GoldenItem]:
    """Subconjunto determinístico que mantém a proporção de cada tipo de pergunta (ao menos 1 por tipo)."""
    if n >= len(items):
        return items
    by_type: dict[str, list[GoldenItem]] = {}
    for item in sorted(items, key=lambda i: i.id):
        by_type.setdefault(item.type, []).append(item)
    picked: list[GoldenItem] = []
    for group in by_type.values():
        quota = max(1, round(n * len(group) / len(items)))
        step = len(group) / quota
        picked += [group[int(i * step)] for i in range(quota)]
    return sorted(picked, key=lambda i: i.id)[:n]


def evaluate_item(item: GoldenItem, pipeline: str, use_judge: bool) -> dict:
    result = run_query(item.question, pipeline)
    retrieved = [c.id for c in result.context]
    row: dict = {
        "id": item.id,
        "type": item.type,
        "source": item.source,
        "question": item.question,
        "answer": result.answer,
        "retrieved_ids": retrieved,
        "cited_ids": [c.chunk_id for c in result.citations],
        "model_tier_used": result.model_tier_used,
        "cost_usd": result.total_cost_usd,
        "latency_ms": result.total_latency_ms,
        "tokens_in": sum(u.tokens_in for u in result.usage),
        "tokens_out": sum(u.tokens_out for u in result.usage),
        "usage": [u.model_dump() for u in result.usage],
    }
    if item.relevant_doc_ids:
        relevant = item.relevant_doc_ids
        row["retrieval"] = {
            f"recall@{K}": recall_at_k(retrieved, relevant, K),
            f"doc_recall@{K}": recall_at_k(retrieved, doc_level(relevant), K),
            "mrr": mrr(retrieved, relevant),
            f"ndcg@{K}": ndcg_at_k(retrieved, relevant, K),
        }
    if use_judge:
        verdict = judge_module.judge(
            item.question, result.answer, format_context(result.context), item.reference_answer
        )
        row["judge"] = verdict.model_dump()
    return row


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return round(ordered[index], 1)


def aggregate(rows: list[dict]) -> dict:
    def judged(key: str, subset: list[dict]) -> list[float]:
        return [r["judge"][key] for r in subset if "judge" in r and r["judge"][key] is not None]

    answerable = [r for r in rows if "retrieval" in r]
    should_refuse = [r for r in rows if r["type"] in ("trap", "out_of_scope")]
    out: dict = {
        "n": len(rows),
        "retrieval": {
            key: mean([r["retrieval"][key] for r in answerable])
            for key in (f"recall@{K}", f"doc_recall@{K}", "mrr", f"ndcg@{K}")
        }
        | {"n": len(answerable)},
        "generation": {
            "faithfulness": mean(judged("faithfulness", rows)),
            "citation_precision": mean(judged("citation_precision", rows)),
            "correctness": mean(judged("correctness", rows)),
            "correctness_answerable": mean(judged("correctness", answerable)),
            # recusa em pergunta respondível é erro; em armadilha ou fora de escopo é o comportamento certo
            "refusal_rate_answerable": mean([float(r["judge"]["is_refusal"]) for r in answerable if "judge" in r]),
            "refusal_rate_trap_or_out_of_scope": mean(
                [float(r["judge"]["is_refusal"]) for r in should_refuse if "judge" in r]
            ),
        },
        "operational": {
            "cost_per_query_usd": round(statistics.fmean(r["cost_usd"] for r in rows), 6),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "latency_p50_ms": percentile([r["latency_ms"] for r in rows], 0.50),
            "latency_p95_ms": percentile([r["latency_ms"] for r in rows], 0.95),
            "tokens_in_per_query": round(statistics.fmean(r["tokens_in"] for r in rows)),
            "tokens_out_per_query": round(statistics.fmean(r["tokens_out"] for r in rows)),
            "judge_cost_usd": round(sum(r["judge"]["judge_cost_usd"] for r in rows if "judge" in r), 4),
        },
    }
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--subset", type=int, default=None, help="Número de perguntas (estratificado por tipo)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true", help="Só retrieval e métricas operacionais")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = load_pipeline(args.pipeline)
    items = load_golden()
    if args.subset:
        items = stratified_subset(items, args.subset)
    name = args.pipeline if not args.subset else f"{args.pipeline}.subset{args.subset}"

    PARTIAL.mkdir(parents=True, exist_ok=True)
    partial_file = PARTIAL / f"{name}.jsonl"
    done: dict[str, dict] = {}
    if partial_file.exists():
        with partial_file.open(encoding="utf-8") as f:
            done = {(row := json.loads(line))["id"]: row for line in f if line.strip()}
    todo = [i for i in items if i.id not in done]
    print(f"{name}: {len(items)} perguntas ({len(done)} já feitas, {len(todo)} a rodar)")

    errors = 0
    with partial_file.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(evaluate_item, i, args.pipeline, not args.no_judge): i for i in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                row = future.result()
            except Exception as error:  # uma pergunta com erro não derruba a execução inteira
                errors += 1
                print(f"\nERRO em {item.id}: {error!r}")
                continue
            done[row["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {n}/{len(todo)}", end="\r")

    if errors:
        raise SystemExit(f"\n{errors} perguntas falharam. Rode de novo para retomar; nada foi salvo em results/.")

    rows = [done[i.id] for i in items]
    by_type = {t: aggregate([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}
    output = {
        "pipeline": pipeline.model_dump(),
        "run": {
            "name": name,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "dataset_hash": dataset_hash(),
            "n_questions": len(rows),
            "subset": args.subset,
            "concurrency": args.concurrency,
            "models": {
                "small": settings.llm_model_small,
                "medium": settings.llm_model_medium,
                "large": settings.llm_model_large,
                "embedding": settings.embedding_model,
                "judge": None if args.no_judge else settings.model_for(judge_module.JUDGE_TIER),
            },
        },
        "metrics": aggregate(rows),
        "metrics_by_type": by_type,
        "questions": rows,
    }
    out_file = RESULTS / f"{name}.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_file.unlink()

    m = output["metrics"]
    print(f"\nsalvo em {out_file}")
    print(json.dumps({k: m[k] for k in ("retrieval", "generation", "operational")}, indent=2))


if __name__ == "__main__":
    main()
