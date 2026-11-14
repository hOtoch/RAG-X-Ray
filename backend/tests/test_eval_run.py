"""Partes do eval que não dependem de LLM: dataset, subset, agregação e pontuação do juiz."""

from eval.dataset import GoldenItem, check
from eval.judge import Claim, FaithfulnessVerdict, score_faithfulness
from eval.run_eval import aggregate, percentile, stratified_subset


def item(n: int, qtype: str = "simple", docs: list[str] | None = None) -> GoldenItem:
    relevant = (docs if docs is not None else ["doc-a#chunk-0"]) if qtype in ("simple", "multi_hop") else []
    return GoldenItem(
        id=f"q-{n:04d}",
        question=f"question {n}",
        type=qtype,
        reference_answer="ref",
        relevant_doc_ids=relevant,
        source="manual",
    )


def test_check_finds_problems():
    items = [item(1), item(1), item(2, "trap"), item(3, docs=[])]
    items[2].relevant_doc_ids = ["doc-a#chunk-0"]
    problems = check(items, indexed_ids={"doc-b#chunk-0"})
    text = "\n".join(problems)
    assert "id duplicado" in text
    assert "q-0002: tipo trap não deveria ter" in text
    assert "q-0003: tipo simple sem relevant_doc_ids" in text
    assert "chunk fora do índice: doc-a#chunk-0" in text


def test_check_passes_on_valid_dataset():
    assert check([item(1), item(2, "out_of_scope")], indexed_ids={"doc-a#chunk-0"}) == []


def test_stratified_subset_keeps_every_type_and_is_deterministic():
    items = [item(n) for n in range(80)] + [item(100 + n, "trap") for n in range(10)]
    items += [item(200 + n, "out_of_scope") for n in range(10)]
    subset = stratified_subset(items, 20)
    assert len(subset) == 20
    assert {i.type for i in subset} == {"simple", "trap", "out_of_scope"}
    assert sum(i.type == "simple" for i in subset) == 16
    assert [i.id for i in subset] == [i.id for i in stratified_subset(list(reversed(items)), 20)]


def test_score_faithfulness():
    verdict = FaithfulnessVerdict(
        claims=[
            Claim(claim="a", supported=True, cited=[1], supported_by_cited=True),
            Claim(claim="b", supported=True, cited=[2], supported_by_cited=False),
            Claim(claim="c", supported=False),
            Claim(claim="d", supported=True),
        ]
    )
    assert score_faithfulness(verdict) == (0.75, 0.5)
    assert score_faithfulness(FaithfulnessVerdict()) == (None, None)


def test_percentile():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.5) == 51.0
    assert percentile(values, 0.95) == 95.0
    assert percentile([], 0.5) is None


def test_aggregate():
    def row(qtype, recall, faith, correct, refusal, cost):
        r = {
            "type": qtype, "cost_usd": cost, "latency_ms": 100.0, "tokens_in": 10, "tokens_out": 5,
            "judge": {"faithfulness": faith, "citation_precision": None, "correctness": correct,
                      "is_refusal": refusal, "judge_cost_usd": 0.01},
        }  # fmt: skip
        if recall is not None:
            r["retrieval"] = {"recall@5": recall, "doc_recall@5": recall, "mrr": recall, "ndcg@5": recall}
        return r

    rows = [row("simple", 1.0, 1.0, 1.0, False, 0.002), row("simple", 0.0, 0.5, 0.0, True, 0.004),
            row("trap", None, None, 1.0, True, 0.006)]  # fmt: skip
    m = aggregate(rows)
    assert m["retrieval"]["recall@5"] == 0.5 and m["retrieval"]["n"] == 2
    assert m["generation"]["faithfulness"] == 0.75  # None fica de fora da média
    assert m["generation"]["correctness"] == 0.6667
    assert m["generation"]["refusal_rate_answerable"] == 0.5
    assert m["generation"]["refusal_rate_trap_or_out_of_scope"] == 1.0
    assert m["operational"]["cost_per_query_usd"] == 0.004
    assert m["operational"]["judge_cost_usd"] == 0.03
