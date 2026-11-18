"""Calibração do limiar θ do cache semântico.

Passo 1 (uma vez): gerar pares a partir do golden dataset.
    uv run --project backend python -m eval.cache_calibration pairs

Para cada pergunta respondível sorteada, o modelo `large` escreve:
- uma paráfrase (mesmo pedido, outras palavras, às vezes em português): deve compartilhar a resposta;
- uma pergunta vizinha (mesmo tema e vocabulário, mas pedindo outra coisa): NÃO deve compartilhar.
Os rótulos vêm da construção do par. A saída vai para eval/datasets/cache_pairs.jsonl.

Passo 2: varrer θ e medir taxa de acerto e de falso positivo.
    uv run --project backend python -m eval.cache_calibration sweep

- acerto (recall): fração dos pares "deve compartilhar" com similaridade > θ.
- falso positivo: fração dos pares "não deve compartilhar" com similaridade > θ. Servir a resposta de
  outra pergunta é o erro grave do cache, então o θ escolhido é o menor com falso positivo <= MAX_FPR.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.config import get_settings
from app.llm.factory import get_embeddings
from app.llm.structured import call_structured
from eval.dataset import load_golden

PAIRS = Path(__file__).parent / "datasets" / "cache_pairs.jsonl"
OUT = Path(__file__).parent / "results" / "cache_threshold.json"
SEED = 5
N_SOURCES = 60
MAX_FPR = 0.02

PAIR_PROMPT = """You create test pairs for the semantic cache of a documentation QA system (LangChain / LangGraph).

Given a question, write:
- paraphrase: the SAME request with different wording, as a real user would type it. It must have exactly \
the same correct answer. {language_rule}
- neighbor: a DIFFERENT question on the same topic, reusing much of the same vocabulary, whose correct \
answer is different (ask about another parameter, the opposite case, another class, a different step). \
Write it in English."""


class Pair(BaseModel):
    paraphrase: str
    neighbor: str


def make_pairs() -> None:
    rng = random.Random(SEED)
    items = [i for i in load_golden() if i.type in ("simple", "multi_hop")]
    sources = rng.sample(items, N_SOURCES)

    def one(index_item):
        index, item = index_item
        # um terço das paráfrases em português, como os usuários deste projeto escrevem
        rule = "Write it in Brazilian Portuguese." if index % 3 == 0 else "Write it in English."
        pair, usage = call_structured(
            "large", Pair, PAIR_PROMPT.format(language_rule=rule), item.question, "cache_pairs"
        )
        return [
            {"source_id": item.id, "a": item.question, "b": pair.paraphrase, "same_answer": True},
            {"source_id": item.id, "a": item.question, "b": pair.neighbor, "same_answer": False},
        ], usage.cost_usd

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(one, enumerate(sources)))
    with PAIRS.open("w", encoding="utf-8") as f:
        for rows, _ in results:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{2 * len(results)} pares em {PAIRS} (US$ {sum(c for _, c in results):.4f})")


def similarities(pairs: list[dict]) -> np.ndarray:
    texts = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    vectors = dict(zip(texts, get_embeddings().embed_documents(texts), strict=True))

    def cos(a: str, b: str) -> float:
        va, vb = np.asarray(vectors[a]), np.asarray(vectors[b])
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

    return np.array([cos(p["a"], p["b"]) for p in pairs])


def sweep() -> None:
    with PAIRS.open(encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    sims = similarities(pairs)
    same = np.array([p["same_answer"] for p in pairs])
    curve = []
    for theta in np.round(np.arange(0.70, 0.995, 0.01), 2):
        hit_rate = float((sims[same] > theta).mean())
        fpr = float((sims[~same] > theta).mean())
        curve.append({"theta": float(theta), "hit_rate": round(hit_rate, 4), "false_positive_rate": round(fpr, 4)})
    safe = [c for c in curve if c["false_positive_rate"] <= MAX_FPR]
    chosen = min(safe, key=lambda c: c["theta"]) if safe else curve[-1]

    by_language = {}
    for lang, test in (("pt", lambda b: any(ch in b for ch in "ãçéêóõáí")), ("en", lambda b: True)):
        mask = np.array([p["same_answer"] and test(p["b"]) for p in pairs])
        if mask.any():
            by_language[lang] = round(float((sims[mask] > chosen["theta"]).mean()), 4)

    result = {
        "embedding_model": get_settings().embedding_model,
        "n_pairs": len(pairs),
        "n_same": int(same.sum()),
        "n_different": int((~same).sum()),
        "max_false_positive_rate": MAX_FPR,
        "chosen": chosen,
        "paraphrase_hit_rate_by_language_at_chosen": by_language,
        "similarity_same_answer": {"mean": round(float(sims[same].mean()), 4), "min": round(float(sims[same].min()), 4)},  # noqa: E501
        "similarity_different": {"mean": round(float(sims[~same].mean()), 4), "max": round(float(sims[~same].max()), 4)},  # noqa: E501
        "curve": curve,
        "hardest_false_positives": sorted(
            ({"a": p["a"], "b": p["b"], "similarity": round(float(s), 4)} for p, s in zip(pairs, sims, strict=True)
             if not p["same_answer"]),
            key=lambda x: -x["similarity"],
        )[:5],
    }  # fmt: skip
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("curve", "hardest_false_positives")}, indent=2))
    print("\nθ  acerto  falso positivo")
    for c in curve[::3]:
        print(f"{c['theta']:.2f}  {c['hit_rate']:.3f}   {c['false_positive_rate']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pairs", "sweep"])
    args = parser.parse_args()
    make_pairs() if args.command == "pairs" else sweep()
