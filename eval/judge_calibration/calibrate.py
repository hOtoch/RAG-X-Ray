"""Calibração do juiz LLM contra rótulos humanos (kappa de Cohen).

Passo 1: gerar a planilha de rotulagem a partir de um resultado de eval:
    uv run --project backend python -m eval.judge_calibration.calibrate sample --results v0 v1 --n 50

Passo 2: uma pessoa preenche as colunas `human_correctness` e `human_faithful` em labels.csv,
sem olhar as colunas do juiz (elas ficam em judge_verdicts.json, separadas de propósito).

Passo 3: calcular a concordância:
    uv run --project backend python -m eval.judge_calibration.calibrate kappa
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
LABELS = HERE / "labels.csv"
VERDICTS = HERE / "judge_verdicts.json"
OUT = HERE / "results.json"
SEED = 11

CORRECTNESS_LABELS = ("correct", "partially_correct", "incorrect")
FAITHFUL_LABELS = ("yes", "no")
FIELDS = ["key", "type", "question", "reference_answer", "answer", "context", "human_correctness", "human_faithful"]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Concordância entre dois avaliadores, descontando a concordância esperada por acaso."""
    if len(a) != len(b) or not a:
        raise ValueError("as listas precisam ter o mesmo tamanho e não podem ser vazias")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[label] * count_b[label] for label in set(a) | set(b)) / (n * n)
    if expected == 1:
        return 1.0  # os dois usaram um único rótulo, sempre igual
    return (observed - expected) / (1 - expected)


def sample(result_names: list[str], n: int) -> None:
    from app.db import get_pool
    from eval.dataset import load_golden

    golden = {item.id: item for item in load_golden()}
    rows = []
    for name in result_names:
        data = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
        rows += [(name, q) for q in data["questions"] if "judge" in q]

    # Metade das amostras vem de respostas que o juiz reprovou em algo, senão quase tudo seria "correto".
    rng = random.Random(SEED)
    flagged = [r for r in rows if r[1]["judge"]["correctness"] < 1 or (r[1]["judge"]["faithfulness"] or 1) < 1]
    clean = [r for r in rows if r not in flagged]
    picked = rng.sample(flagged, min(n // 2, len(flagged)))
    picked += rng.sample(clean, min(n - len(picked), len(clean)))
    rng.shuffle(picked)

    ids = sorted({cid for _, q in picked for cid in q["retrieved_ids"]})
    with get_pool().connection() as conn:
        texts = dict(conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall())

    verdicts = {}
    with LABELS.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for name, q in picked:
            key = f"{name}:{q['id']}"
            context = "\n\n".join(f"[{i}] {texts.get(cid, '')}" for i, cid in enumerate(q["retrieved_ids"], start=1))
            writer.writerow(
                {
                    "key": key,
                    "type": q["type"],
                    "question": q["question"],
                    "reference_answer": golden[q["id"]].reference_answer,
                    "answer": q["answer"],
                    "context": context,
                    "human_correctness": "",
                    "human_faithful": "",
                }
            )
            faithfulness = q["judge"]["faithfulness"]
            verdicts[key] = {
                "correctness": q["judge"]["correctness_verdict"],
                # resposta sem afirmações (recusa) conta como fiel
                "faithful": "yes" if faithfulness is None or faithfulness == 1 else "no",
            }
    VERDICTS.write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    n_flagged = len([p for p in picked if p in flagged])
    print(f"{len(picked)} respostas para rotular em {LABELS} ({n_flagged} reprovadas pelo juiz)")
    print(f"human_correctness: {' | '.join(CORRECTNESS_LABELS)}   human_faithful: {' | '.join(FAITHFUL_LABELS)}")


def kappa() -> None:
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    with LABELS.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_correctness"].strip() and r["human_faithful"].strip()]
    if not rows:
        raise SystemExit(f"Nenhuma linha rotulada em {LABELS}. Preencha human_correctness e human_faithful.")
    for row in rows:
        if (
            row["human_correctness"].strip() not in CORRECTNESS_LABELS
            or row["human_faithful"].strip() not in FAITHFUL_LABELS
        ):
            raise SystemExit(f"Rótulo inválido na linha {row['key']}")

    human_c = [r["human_correctness"].strip() for r in rows]
    judge_c = [verdicts[r["key"]]["correctness"] for r in rows]
    human_f = [r["human_faithful"].strip() for r in rows]
    judge_f = [verdicts[r["key"]]["faithful"] for r in rows]
    result = {
        "n_labeled": len(rows),
        "correctness": {
            "kappa": round(cohen_kappa(human_c, judge_c), 3),
            "agreement": round(sum(h == j for h, j in zip(human_c, judge_c, strict=True)) / len(rows), 3),
        },
        "faithfulness": {
            "kappa": round(cohen_kappa(human_f, judge_f), 3),
            "agreement": round(sum(h == j for h, j in zip(human_f, judge_f, strict=True)) / len(rows), 3),
        },
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--results", nargs="+", default=["v0"])
    s.add_argument("--n", type=int, default=50)
    sub.add_parser("kappa")
    args = parser.parse_args()
    sample(args.results, args.n) if args.command == "sample" else kappa()
