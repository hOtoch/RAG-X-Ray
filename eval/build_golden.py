"""Monta eval/datasets/golden.jsonl a partir das perguntas sintéticas e das escritas à mão.

Uso (a partir da raiz do repositório):
    uv run --project backend python -m eval.build_golden

Para as perguntas manuais, o script busca os trechos mais próximos dentro das páginas indicadas e pede ao
modelo `large` uma resposta de referência usando só esses trechos, junto com os trechos que a sustentam.
As respostas ficam em cache em manual_references.jsonl, para a revisão humana não ser sobrescrita.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import get_pool
from app.llm.factory import get_embeddings, get_llm
from app.llm.usage import usage_from_message
from eval.dataset import GOLDEN, validate

DATASETS = Path(__file__).parent / "datasets"
MANUAL = DATASETS / "manual_questions.yaml"
MANUAL_REFS = DATASETS / "manual_references.jsonl"
SYNTHETIC = DATASETS / "synthetic_candidates.jsonl"
PASSAGES_PER_QUESTION = 10

REFERENCE_PROMPT = """You write the reference answer for an evaluation question about the LangChain / \
LangGraph documentation.

Use ONLY the numbered passages. Write a short, specific answer (1 to 4 sentences), in English.
List in `supporting` the numbers of the passages that are needed to support the answer (usually 1 to 3; \
never list a passage that is merely related).
If the passages do not answer the question, set answerable to false."""

OUT_OF_SCOPE_REFERENCE = (
    "The question is outside the scope of the LangChain / LangGraph documentation. "
    "The assistant should politely decline and say it only answers questions about that documentation."
)


class Reference(BaseModel):
    answerable: bool
    answer: str = ""
    supporting: list[int] = Field(default_factory=list)


def passages_for(question: str, docs: list[str]) -> list[tuple[str, str]]:
    vector = np.array(get_embeddings().embed_query(question), dtype=np.float32)
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, title, section, text FROM chunks WHERE doc_id = ANY(%s) ORDER BY embedding <=> %s LIMIT %s",
            (docs, vector, PASSAGES_PER_QUESTION),
        ).fetchall()
    return [(r[0], f"{r[1]} > {r[2]}\n{r[3]}") for r in rows]


def reference_for(entry: dict) -> tuple[dict, float]:
    passages = passages_for(entry["q"], entry["docs"])
    numbered = "\n\n---\n\n".join(f"[{n}] {text}" for n, (_, text) in enumerate(passages, start=1))
    llm = get_llm("large").with_structured_output(Reference, include_raw=True)
    out = llm.invoke([("system", REFERENCE_PROMPT), ("human", f"{numbered}\n\nQuestion: {entry['q']}")])
    cost = usage_from_message("reference", get_settings().model_for("large"), out["raw"]).cost_usd
    ref: Reference | None = out["parsed"]
    if ref is None or not ref.answerable:
        return {"question": entry["q"], "answerable": False}, cost
    ids = [passages[n - 1][0] for n in ref.supporting if 1 <= n <= len(passages)]
    return {"question": entry["q"], "answerable": True, "reference_answer": ref.answer, "relevant_doc_ids": ids}, cost


def manual_references(entries: list[dict]) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if MANUAL_REFS.exists():
        with MANUAL_REFS.open(encoding="utf-8") as f:
            cached = {(d := json.loads(line))["question"]: d for line in f}
    todo = [e for e in entries if e["q"] not in cached]
    if todo:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(reference_for, todo))
        for ref, _ in results:
            cached[ref["question"]] = ref
        print(f"referências geradas: {len(todo)} (US$ {sum(c for _, c in results):.4f})")
        with MANUAL_REFS.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(cached[e["q"]], ensure_ascii=False) + "\n")
    return cached


def main() -> None:
    manual = yaml.safe_load(MANUAL.read_text(encoding="utf-8"))
    refs = manual_references(manual["manual"])
    items: list[dict] = []

    with SYNTHETIC.open(encoding="utf-8") as f:
        items += [json.loads(line) for line in f]

    for entry in manual["manual"]:
        ref = refs[entry["q"]]
        if not ref["answerable"] or not ref["relevant_doc_ids"]:
            print(f"SEM REFERÊNCIA (fora do golden): {entry['q']}")
            continue
        multi = len({i.split("#")[0] for i in ref["relevant_doc_ids"]}) > 1
        items.append(
            {
                "question": entry["q"],
                "type": "multi_hop" if multi else "simple",
                "reference_answer": ref["reference_answer"],
                "relevant_doc_ids": ref["relevant_doc_ids"],
                "source": "manual",
            }
        )
    for entry in manual["trap"]:
        reference = (
            f"The premise is false: {entry['note']} The assistant should say the documentation does not "
            "support the premise and must not invent parameters, APIs or behavior."
        )
        items.append(
            {
                "question": entry["q"],
                "type": "trap",
                "reference_answer": reference,
                "relevant_doc_ids": [],
                "source": "manual",
            }
        )
    for question in manual["out_of_scope"]:
        items.append(
            {
                "question": question,
                "type": "out_of_scope",
                "reference_answer": OUT_OF_SCOPE_REFERENCE,
                "relevant_doc_ids": [],
                "source": "manual",
            }
        )

    golden = [{"id": f"q-{n:04d}", **item} for n, item in enumerate(items, start=1)]
    with GOLDEN.open("w", encoding="utf-8") as f:
        for item in golden:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    validate(check_index=True)


if __name__ == "__main__":
    main()
