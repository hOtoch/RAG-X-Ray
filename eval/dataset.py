"""Schema, leitura e validação do golden dataset.

Uso (a partir da raiz do repositório):
    uv run --project backend python -m eval.dataset
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

GOLDEN = Path(__file__).parent / "datasets" / "golden.jsonl"

QuestionType = Literal["simple", "multi_hop", "out_of_scope", "trap"]


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    type: QuestionType
    reference_answer: str
    relevant_doc_ids: list[str]
    source: Literal["synthetic", "manual"]


def load_golden(path: Path = GOLDEN) -> list[GoldenItem]:
    with path.open(encoding="utf-8") as f:
        return [GoldenItem(**json.loads(line)) for line in f if line.strip()]


def dataset_hash(path: Path = GOLDEN) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def check(items: list[GoldenItem], indexed_ids: set[str] | None = None) -> list[str]:
    """Retorna a lista de problemas (vazia se o dataset está consistente)."""
    problems: list[str] = []
    for field in ("id", "question"):
        counts = Counter(getattr(i, field).strip().lower() for i in items)
        problems += [f"{field} duplicado: {value}" for value, n in counts.items() if n > 1]
    for item in items:
        needs_docs = item.type in ("simple", "multi_hop")
        if needs_docs and not item.relevant_doc_ids:
            problems.append(f"{item.id}: tipo {item.type} sem relevant_doc_ids")
        if not needs_docs and item.relevant_doc_ids:
            problems.append(f"{item.id}: tipo {item.type} não deveria ter relevant_doc_ids")
        if not item.question.strip() or not item.reference_answer.strip():
            problems.append(f"{item.id}: pergunta ou resposta vazia")
        if indexed_ids is not None:
            problems += [f"{item.id}: chunk fora do índice: {c}" for c in item.relevant_doc_ids if c not in indexed_ids]
    return problems


def validate(path: Path = GOLDEN, check_index: bool = False) -> list[GoldenItem]:
    items = load_golden(path)
    indexed = None
    if check_index:
        from app.db import get_pool

        with get_pool().connection() as conn:
            indexed = {r[0] for r in conn.execute("SELECT id FROM chunks")}
    problems = check(items, indexed)
    by_type = Counter(i.type for i in items)
    by_source = Counter(i.source for i in items)
    print(f"golden: {len(items)} perguntas | por tipo {dict(by_type)} | por origem {dict(by_source)}")
    print(f"hash do dataset: {dataset_hash(path)}")
    if problems:
        raise SystemExit("Dataset inválido:\n" + "\n".join(problems))
    print("dataset válido")
    return items


if __name__ == "__main__":
    validate(check_index=True)
