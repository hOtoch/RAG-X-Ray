"""Gate de regressão: compara um resultado de eval com a linha de base e falha se piorar além da tolerância.

Uso (a partir da raiz do repositório):
    uv run --project backend python -m eval.gate --baseline eval/results/baseline/v3.subset24.json \
        --current eval/results/v3.subset24.json

Sai com código 1 se alguma métrica violar a tolerância. Escreve um resumo em Markdown em
eval/results/gate_summary.md (a CI publica esse arquivo como comentário no PR).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (caminho da métrica, direção, tolerância). "higher" = maior é melhor; a tolerância é a piora máxima aceita.
# Custo e latência usam tolerância relativa (fração); qualidade usa pontos absolutos.
CHECKS = [
    (("retrieval", "recall@5"), "higher", 0.05, "abs"),
    (("retrieval", "doc_recall@5"), "higher", 0.05, "abs"),
    (("generation", "faithfulness"), "higher", 0.05, "abs"),
    (("generation", "correctness"), "higher", 0.08, "abs"),
    (("operational", "cost_per_query_usd"), "lower", 0.25, "rel"),
]
SUMMARY = Path(__file__).parent / "results" / "gate_summary.md"


def get(metrics: dict, path: tuple[str, ...]) -> float | None:
    value = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def compare(baseline: dict, current: dict) -> tuple[list[dict], bool]:
    rows, ok = [], True
    for path, direction, tolerance, kind in CHECKS:
        before, after = get(baseline["metrics"], path), get(current["metrics"], path)
        if before is None or after is None:
            rows.append({"metric": ".".join(path), "before": before, "after": after, "status": "sem dado"})
            continue
        delta = after - before
        worse = -delta if direction == "higher" else delta
        limit = tolerance * abs(before) if kind == "rel" else tolerance
        passed = worse <= limit + 1e-12
        ok &= passed
        rows.append(
            {
                "metric": ".".join(path),
                "before": before,
                "after": after,
                "delta": delta,
                "limit": f"{tolerance:.0%}" if kind == "rel" else f"{tolerance:+.2f}".replace("+", "−"),
                "status": "ok" if passed else "REGRESSÃO",
            }
        )
    return rows, ok


def markdown(rows: list[dict], ok: bool, baseline: dict, current: dict) -> str:
    def fmt(v):
        return "—" if v is None else (f"{v:.6f}" if abs(v) < 0.01 and v != 0 else f"{v:.4f}")

    lines = [
        f"### Eval na CI: {'✅ sem regressão' if ok else '❌ regressão detectada'}",
        "",
        f"Pipeline `{current['run']['name']}`, {current['run']['n_questions']} perguntas, "
        f"linha de base do commit `{baseline['run']['git_commit']}`.",
        "",
        "| Métrica | Base | Agora | Δ | Tolerância | |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = "—" if "delta" not in r else f"{r['delta']:+.4f}"
        lines.append(f"| {r['metric']} | {fmt(r['before'])} | {fmt(r['after'])} | {delta} | {r.get('limit', '—')} | "
                     f"{r['status']} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    if baseline["run"]["dataset_hash"] != current["run"]["dataset_hash"]:
        print("aviso: o dataset mudou desde a linha de base; a comparação pode não ser justa")
    rows, ok = compare(baseline, current)
    SUMMARY.write_text(markdown(rows, ok, baseline, current), encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
