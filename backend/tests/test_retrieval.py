"""RRF, fusão híbrida, BM25, compressão de contexto e os nós da v1, sem banco e sem modelos reais."""

from __future__ import annotations

import pytest

from app.graph.build import build_graph
from app.graph.nodes.compress import compress, dedupe, fit_budget, similarity
from app.graph.state import ScoredChunk
from app.pipeline import PipelineConfig
from app.retrieval.hybrid import fuse
from app.retrieval.lexical import BM25Index, tokenize
from app.retrieval.rrf import rrf_fuse, rrf_scores


def chunk(cid: str, text: str = "text", tokens: int = 10, **ranks) -> ScoredChunk:
    return ScoredChunk(
        id=cid, doc_id=cid.split("#")[0], title="T", section="S", url="u", text=text, token_count=tokens, **ranks
    )


# --- RRF ---


def test_rrf_scores_by_hand():
    scores = rrf_scores([["a", "b", "c"], ["b", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_item_in_both_lists_beats_item_in_one():
    assert rrf_fuse([["x", "a"], ["y", "a"]])[0] == "a"


def test_rrf_ties_keep_first_seen_order():
    assert rrf_fuse([["a", "b"], ["b", "a"]]) == ["a", "b"]


def test_rrf_handles_lists_of_different_sizes_and_empty():
    assert rrf_fuse([["a", "b", "c"], []]) == ["a", "b", "c"]
    assert rrf_fuse([]) == []
    assert rrf_fuse([["a"], ["b", "c", "d"]])[:2] == ["a", "b"]


def test_fuse_keeps_rank_from_each_source():
    dense = [chunk("a#chunk-0", dense_rank=1), chunk("b#chunk-0", dense_rank=2)]
    lexical = [chunk("b#chunk-0", bm25_rank=1), chunk("c#chunk-0", bm25_rank=2)]
    fused = fuse(dense, lexical)
    assert [c.id for c in fused] == ["b#chunk-0", "a#chunk-0", "c#chunk-0"]
    b, a, c = fused
    assert (b.dense_rank, b.bm25_rank, b.rrf_rank) == (2, 1, 1)
    assert (a.dense_rank, a.bm25_rank, a.rrf_rank) == (1, None, 2)
    assert (c.dense_rank, c.bm25_rank, c.rrf_rank) == (None, 2, 3)


# --- BM25 ---


def test_tokenize_keeps_identifiers_and_drops_stopwords():
    assert tokenize("How do I use add_conditional_edges?") == [
        "use",
        "add_conditional_edges",
        "add",
        "conditional",
        "edges",
    ]
    assert tokenize("como usar o Send") == ["usar", "send"]


ROWS = [
    ("d1#chunk-0", "d1", "Graph API", "Edges", "u", "Use add_conditional_edges to route between nodes.", 9),
    ("d2#chunk-0", "d2", "Persistence", "Threads", "u", "A checkpointer saves state for each thread_id.", 9),
    ("d3#chunk-0", "d3", "Streaming", "Modes", "u", "Stream modes include updates, values and messages.", 9),
]


def test_bm25_finds_exact_identifier():
    hits = BM25Index(ROWS).search("how to call add_conditional_edges", 3)
    assert hits[0].id == "d1#chunk-0" and hits[0].bm25_rank == 1


def test_bm25_returns_nothing_without_matching_terms():
    index = BM25Index(ROWS)
    assert index.search("kubernetes ingress", 3) == []
    assert index.search("the of and", 3) == []


# --- compress_context ---


def test_similarity_and_dedupe():
    a = chunk("a#chunk-0", "install the package with pip install langgraph")
    b = chunk("b#chunk-0", "Install the package with pip install langgraph.")
    c = chunk("c#chunk-0", "checkpointers persist state between runs")
    assert similarity(a.text, b.text) == 1.0
    assert similarity(a.text, c.text) < 0.2
    assert [x.id for x in dedupe([a, b, c])] == ["a#chunk-0", "c#chunk-0"]


def test_fit_budget_keeps_order_and_always_keeps_first():
    chunks = [chunk("a#chunk-0", tokens=300), chunk("b#chunk-0", tokens=300), chunk("c#chunk-0", tokens=300)]
    assert [c.id for c in fit_budget(chunks, 700)] == ["a#chunk-0", "b#chunk-0"]
    assert [c.id for c in fit_budget(chunks, 100)] == ["a#chunk-0"]
    assert fit_budget(chunks, None) == chunks


def test_compress_never_exceeds_budget_after_first():
    chunks = [chunk(f"d{i}#chunk-0", text=f"unique text number {i} about topic {i}", tokens=200) for i in range(6)]
    out = compress(chunks, 500)
    assert sum(c.token_count for c in out) <= 500 and len(out) == 2


# --- grafo v1 ---


def test_graph_nodes_follow_pipeline_flags():
    def nodes(**flags) -> list[str]:
        graph = build_graph(PipelineConfig(name="t", **flags))
        return [n for n in graph.get_graph().nodes if not n.startswith("__")]

    assert nodes() == ["retrieve", "generate"]
    assert nodes(retrieval_mode="hybrid", rerank=True) == ["retrieve", "rerank", "generate"]
    assert nodes(rerank=True, compress=True) == ["retrieve", "rerank", "compress_context", "generate"]


def test_v1_flow_with_fakes(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.config import ModelPrice
    from app.service import QueryRun

    dense_hits = [chunk(f"d{i}#chunk-0", text=f"dense text {i}", dense_rank=i + 1) for i in range(4)]
    lexical_hits = [
        chunk("d3#chunk-0", text="dense text 3", bm25_rank=1),
        chunk("x#chunk-0", text="lexical only", bm25_rank=2),
    ]

    class FakeLLM:
        def invoke(self, messages, config=None):
            return AIMessage(
                content="answer [1]", usage_metadata={"input_tokens": 9, "output_tokens": 2, "total_tokens": 11}
            )

    def fake_rerank(query, chunks, top_k, model=None):
        ordered = sorted(chunks, key=lambda c: c.id, reverse=True)[:top_k]  # ordem previsível: x, d3, d2...
        return [c.model_copy(update={"rerank_rank": n, "score": 1 / n}) for n, c in enumerate(ordered, start=1)]

    prices = {
        "test-medium": ModelPrice(input=1.0, output=2.0),
        "test-large": ModelPrice(input=10.0, output=20.0),
        "test-embedding": ModelPrice(input=0.1),
    }
    monkeypatch.setattr("app.llm.usage.get_prices", lambda: prices)
    monkeypatch.setattr("app.graph.nodes.generate.get_llm", lambda tier: FakeLLM())
    monkeypatch.setattr("app.retrieval.dense.embed_query", lambda q: [0.0] * 8)
    monkeypatch.setattr("app.retrieval.dense.dense_search", lambda v, k: dense_hits[:k])
    monkeypatch.setattr("app.retrieval.lexical.bm25_search", lambda q, k: lexical_hits[:k])
    monkeypatch.setattr("app.retrieval.reranker.rerank", fake_rerank)
    pipeline = PipelineConfig(
        name="t1", retrieval_mode="hybrid", candidates_k=4, top_k=2, rerank=True, compress=True, context_token_budget=15
    )
    monkeypatch.setattr("app.service.get_graph", lambda name: build_graph(pipeline))
    monkeypatch.setattr("app.service.load_pipeline", lambda name: pipeline)

    run = QueryRun("question", "t1")
    events = list(run.events())
    flow = [(e.type, e.node) for e in events if e.type in ("node_start", "data")]
    assert flow == [
        ("node_start", "retrieve"), ("data", "retrieve"),
        ("node_start", "rerank"), ("data", "rerank"),
        ("node_start", "compress_context"), ("data", "compress_context"),
        ("node_start", "generate"),
    ]  # fmt: skip
    retrieved = next(e for e in events if e.type == "data" and e.node == "retrieve").payload["chunks"]
    assert retrieved[0]["id"] == "d3#chunk-0" and retrieved[0]["rrf_rank"] == 1  # aparece nas duas fontes
    assert {c["id"] for c in retrieved} >= {"x#chunk-0", "d0#chunk-0"}
    reranked = next(e for e in events if e.type == "data" and e.node == "rerank").payload["chunks"]
    assert [c["id"] for c in reranked] == ["x#chunk-0", "d3#chunk-0"]
    compressed = next(e for e in events if e.type == "data" and e.node == "compress_context").payload
    assert (compressed["chunks_before"], compressed["chunks_after"], compressed["tokens_after"]) == (2, 1, 10)
    assert [c.id for c in run.result.context] == ["x#chunk-0"]
    assert [u.node for u in run.result.usage] == ["retrieve", "rerank", "compress_context", "generate"]
