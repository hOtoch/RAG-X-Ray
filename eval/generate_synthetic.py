"""Gera perguntas sintéticas para o golden dataset a partir dos chunks indexados.

Uso (a partir da raiz do repositório):
    uv run --project backend python -m eval.generate_synthetic --simple 70 --multi-hop 30

Cada pergunta é gerada pelo modelo `large` a partir de um chunk (simples) ou de dois chunks de seções
diferentes do mesmo documento (multi-hop). Um segundo passo, com o mesmo modelo, descarta perguntas
que dependem do trecho para fazer sentido ("neste exemplo", "no código acima") ou que são vagas.
A saída vai para eval/datasets/synthetic_candidates.jsonl, para revisão antes de entrar no golden.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import get_pool
from app.llm.factory import get_llm
from app.llm.usage import usage_from_message

OUT = Path(__file__).parent / "datasets" / "synthetic_candidates.jsonl"
SEED = 7
# Proporção de perguntas por área do corpus. O foco do projeto é LangGraph, então ele pesa mais
# do que a sua fatia do corpus; as integrações entram para o retrieval ter concorrência real.
AREA_WEIGHTS = {"langgraph": 0.40, "langchain": 0.25, "deepagents": 0.10, "integrations": 0.20, "other": 0.05}

GEN_PROMPT = """You write evaluation questions for a retrieval-augmented QA system over the LangChain / \
LangGraph documentation.

Given the passage(s) below, write ONE question a developer would realistically ask, plus its reference answer.

Requirements:
- The question must be self-contained: it must make sense without seeing the passage. Never write "in this \
passage", "the example above", "this code", "the snippet".
- Name the library, class or feature the question is about, so it is not ambiguous in a large documentation set.
- The answer must be fully supported by the passage(s). Do not use outside knowledge.
- The reference answer must be short (1 to 4 sentences) and specific.
- {mode_rule}
- If the passage(s) cannot produce a good question (navigation, changelog, boilerplate, only a link list), \
set usable to false.
"""

SIMPLE_RULE = "The question must be answerable from the single passage."
MULTI_RULE = (
    "The question must REQUIRE information from BOTH passages (compare, combine or chain them). "
    "If either passage alone would answer it, set usable to false."
)

CRITIC_PROMPT = """You review an evaluation question for a QA system over a large documentation set \
(LangChain, LangGraph, Deep Agents and hundreds of integration pages).

Reject the question if ANY of these is true:
- It refers to "the passage", "the example", "the code above" or otherwise needs the source text to make sense.
- It is ambiguous without knowing which page it came from (for example, it does not name the library, class \
or integration).
- It is trivial, or the reference answer is not actually an answer to it.
"""


class Generated(BaseModel):
    usable: bool
    question: str = ""
    reference_answer: str = ""


class Verdict(BaseModel):
    accept: bool
    reason: str = Field(description="One short sentence")


def area_of(doc_id: str) -> str:
    if "integrations" in doc_id:
        return "integrations"
    for area in ("langgraph", "langchain", "deepagents"):
        if doc_id.startswith(f"doc-{area}-"):
            return area
    return "other"


def load_chunks() -> list[dict]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, doc_id, title, section, text, token_count FROM chunks WHERE token_count >= 120 ORDER BY id"
        ).fetchall()
    keys = ("id", "doc_id", "title", "section", "text", "token_count")
    chunks = [dict(zip(keys, r, strict=True)) for r in rows]
    # changelog e notas de versão geram perguntas ruins
    chunks = [c for c in chunks if not any(x in c["doc_id"] for x in ("changelog", "releases", "contributing"))]
    # Trechos padronizados (ex.: "como iniciar o modelo") se repetem em várias páginas. Uma pergunta
    # sobre eles teria vários chunks corretos, e o gabarito marcaria só um.
    counts = Counter(c["text"] for c in chunks)
    return [c for c in chunks if counts[c["text"]] == 1]


def passage(chunk: dict) -> str:
    header = chunk["title"] if not chunk["section"] else f"{chunk['title']} > {chunk['section']}"
    return f"### {header}\n{chunk['text']}"


def sample_simple(chunks: list[dict], n: int, rng: random.Random) -> list[list[dict]]:
    by_area: dict[str, list[dict]] = {}
    for c in chunks:
        by_area.setdefault(area_of(c["doc_id"]), []).append(c)
    picked: list[list[dict]] = []
    for area, weight in AREA_WEIGHTS.items():
        pool = by_area.get(area, [])
        picked += [[c] for c in rng.sample(pool, min(round(n * weight), len(pool)))]
    return picked


def sample_multi(chunks: list[dict], n: int, rng: random.Random) -> list[list[dict]]:
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        if area_of(c["doc_id"]) in ("langgraph", "langchain", "deepagents"):
            by_doc.setdefault(c["doc_id"], []).append(c)
    docs = [d for d, cs in by_doc.items() if len({c["section"] for c in cs}) >= 2]
    pairs: list[list[dict]] = []
    for doc in rng.sample(docs, min(n, len(docs))):
        a = rng.choice(by_doc[doc])
        others = [c for c in by_doc[doc] if c["section"] != a["section"]]
        pairs.append([a, rng.choice(others)])
    return pairs


def select_by_area(items: list[dict], target: int) -> list[dict]:
    """Respeita a proporção por área; se uma área ficar curta, completa com as sobras."""
    picked: list[dict] = []
    for area, weight in AREA_WEIGHTS.items():
        picked += [i for i in items if area_of(i["relevant_doc_ids"][0]) == area][: round(target * weight)]
    leftovers = [i for i in items if i not in picked]
    return (picked + leftovers)[:target]


def generate_one(chunks: list[dict], qtype: str) -> tuple[dict | None, float]:
    """Retorna (item ou None se descartado, custo em USD)."""
    model_name = get_settings().model_for("large")
    cost = 0.0

    generator = get_llm("large").with_structured_output(Generated, include_raw=True)
    rule = SIMPLE_RULE if qtype == "simple" else MULTI_RULE
    passages = "\n\n".join(passage(c) for c in chunks)
    out = generator.invoke([("system", GEN_PROMPT.format(mode_rule=rule)), ("human", passages)])
    cost += usage_from_message("generate", model_name, out["raw"]).cost_usd
    generated: Generated | None = out["parsed"]
    if generated is None or not generated.usable or not generated.question.strip():
        return None, cost

    critic = get_llm("large").with_structured_output(Verdict, include_raw=True)
    review = critic.invoke(
        [
            ("system", CRITIC_PROMPT),
            ("human", f"Question: {generated.question}\nReference answer: {generated.reference_answer}"),
        ]
    )
    cost += usage_from_message("critic", model_name, review["raw"]).cost_usd
    if review["parsed"] is None or not review["parsed"].accept:
        return None, cost

    return (
        {
            "question": generated.question.strip(),
            "type": qtype,
            "reference_answer": generated.reference_answer.strip(),
            "relevant_doc_ids": [c["id"] for c in chunks],
            "source": "synthetic",
        },
        cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", type=int, default=70)
    parser.add_argument("--multi-hop", type=int, default=30)
    parser.add_argument(
        "--oversample", type=float, default=1.5, help="Gera mais do que o alvo, pois parte é descartada"
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    chunks = load_chunks()
    jobs = [(c, "simple") for c in sample_simple(chunks, round(args.simple * args.oversample), rng)]
    jobs += [(c, "multi_hop") for c in sample_multi(chunks, round(args.multi_hop * args.oversample), rng)]

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda job: generate_one(*job), jobs))

    total_cost = sum(cost for _, cost in results)
    seen: set[str] = set()
    items: list[dict] = []
    for item, _ in results:
        if item and item["question"].lower() not in seen:
            seen.add(item["question"].lower())
            items.append(item)

    kept = select_by_area([i for i in items if i["type"] == "simple"], args.simple)
    kept += [i for i in items if i["type"] == "multi_hop"][: args.multi_hop]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    by_type = {t: sum(1 for i in kept if i["type"] == t) for t in ("simple", "multi_hop")}
    by_area = Counter(area_of(i["relevant_doc_ids"][0]) for i in kept)
    print(f"por área: {dict(by_area)}")
    print(f"{len(jobs)} tentativas, {len(items)} aceitas, {len(kept)} gravadas {by_type} em {OUT}")
    print(f"custo da geração: US$ {total_cost:.4f}")


if __name__ == "__main__":
    main()
