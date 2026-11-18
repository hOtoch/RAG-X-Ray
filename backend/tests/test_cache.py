"""Cache exato e semântico com um Redis falso em memória."""

from __future__ import annotations

import fakeredis
import pytest

from app import cache
from app.graph.nodes.cache import should_write


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    client = fakeredis.FakeRedis()
    monkeypatch.setattr("app.cache.get_redis", lambda: client)
    return client


def test_normalize():
    assert cache.normalize("  Como  usar o SEND? ") == "como usar o send"
    assert cache.normalize("O que é um reducer?") == cache.normalize("o que e um reducer")


def test_exact_hit_ignores_case_spaces_and_punctuation():
    cache.put_entry("v3", "What is a reducer?", [1.0, 0.0], {"answer": "A function [1].", "citations": []})
    assert cache.lookup_exact("v3", "what is a  reducer")["answer"] == "A function [1]."
    assert cache.lookup_exact("v3", "What is a checkpointer?") is None


def test_namespaces_are_isolated_by_pipeline():
    cache.put_entry("v3", "q", [1.0, 0.0], {"answer": "a"})
    assert cache.lookup_exact("other", "q") is None


def test_semantic_hit_depends_on_threshold():
    cache.put_entry("v3", "How do I run nodes in parallel?", [1.0, 0.0, 0.0], {"answer": "Use Send."})
    near = [0.98, 0.2, 0.0]  # cosseno ~0,98
    entry, sim = cache.lookup_semantic("v3", near, threshold=0.95)
    assert entry["answer"] == "Use Send." and sim == pytest.approx(0.98 / (0.98**2 + 0.04) ** 0.5)
    entry, sim = cache.lookup_semantic("v3", near, threshold=0.99)
    assert entry is None and sim is not None


def test_semantic_empty_cache():
    assert cache.lookup_semantic("v3", [1.0, 0.0], 0.5) == (None, None)


def test_semantic_returns_nearest_of_many():
    cache.put_entry("v3", "a", [1.0, 0.0], {"answer": "A"})
    cache.put_entry("v3", "b", [0.0, 1.0], {"answer": "B"})
    entry, _ = cache.lookup_semantic("v3", [0.1, 0.9], 0.5)
    assert entry["answer"] == "B"


def test_clear():
    cache.put_entry("v3", "a", [1.0, 0.0], {"answer": "A"})
    assert cache.clear("v3") > 0 and cache.lookup_exact("v3", "a") is None


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"route": "simple_rag", "grounded": True}, True),
        ({"route": "simple_rag", "grounded": False}, False),
        ({"route": "multi_hop", "grounded": True, "gave_up": True}, False),
        ({"route": "out_of_scope"}, True),
        ({"route": "trivial"}, False),
        ({"route": "simple_rag", "grounded": True, "cache_hit": "exact"}, False),
        ({"grounded": None}, False),
    ],
)
def test_only_grounded_answers_are_cached(state, expected):
    assert should_write(state) is expected


def test_graph_serves_second_identical_question_from_cache(monkeypatch, fake_redis):
    from langchain_core.messages import AIMessage

    from app.config import ModelPrice
    from app.graph.build import build_graph
    from app.graph.state import ScoredChunk
    from app.pipeline import PipelineConfig
    from app.service import QueryRun

    calls = []

    class FakeLLM:
        def invoke(self, messages, config=None):
            calls.append("generate")
            return AIMessage(
                content="Answer [1].", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
            )

    chunk = ScoredChunk(id="d#chunk-0", doc_id="d", title="T", section="S", url="u", text="t", token_count=1)
    prices = {"test-medium": ModelPrice(input=1.0, output=1.0), "test-large": ModelPrice(input=2.0, output=2.0),
              "test-embedding": ModelPrice(input=0.1)}  # fmt: skip
    monkeypatch.setattr("app.llm.usage.get_prices", lambda: prices)
    monkeypatch.setattr("app.graph.nodes.generate.get_llm", lambda tier: FakeLLM())
    monkeypatch.setattr("app.retrieval.dense.embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr("app.retrieval.dense.dense_search", lambda v, k: [chunk])
    # sem groundedness ligada, a v-teste grava se o estado disser grounded; simulamos com groundedness desligada
    monkeypatch.setattr("app.graph.nodes.cache.should_write", lambda state: not state.get("cache_hit"))
    pipeline = PipelineConfig(name="ct", cache=True, cache_threshold=0.9)
    monkeypatch.setattr("app.service.get_graph", lambda name: build_graph(pipeline))
    monkeypatch.setattr("app.service.load_pipeline", lambda name: pipeline)

    first = QueryRun("What is X?", "ct")
    list(first.events())
    second = QueryRun("what is x", "ct")
    nodes = [e.node for e in second.events() if e.type == "node_start"]
    assert calls == ["generate"]  # a segunda não chamou o LLM
    assert nodes == ["cache_exact"] and second.result.cache_hit == "exact"
    third = QueryRun("Tell me what X is", "ct")  # mesmo vetor falso: acerto semântico
    nodes = [e.node for e in third.events() if e.type == "node_start"]
    assert nodes == ["cache_exact", "cache_semantic"] and third.result.cache_hit == "semantic"
    assert third.result.answer == "Answer [1]." and calls == ["generate"]
