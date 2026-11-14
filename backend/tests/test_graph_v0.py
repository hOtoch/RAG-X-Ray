"""Grafo v0 e rota /query com LLM e banco substituídos por dublês."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.config import ModelPrice
from app.graph.nodes.generate import extract_citations, format_context
from app.graph.state import ScoredChunk
from app.pipeline import PipelineConfig, load_pipeline, load_pipelines

CHUNKS = [
    ScoredChunk(
        id=f"doc-a#chunk-{i}",
        doc_id="doc-a",
        title="A",
        section=f"S{i}",
        url=f"https://x/a#s{i}",
        text=f"text {i}",
        token_count=2,
        score=1 - i / 10,
        dense_rank=i + 1,
    )
    for i in range(3)
]


class FakeLLM:
    def __init__(self, text: str):
        self.text = text
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.text, usage_metadata={"input_tokens": 50, "output_tokens": 7, "total_tokens": 57})


@pytest.fixture
def fakes(monkeypatch):
    llm = FakeLLM("Send runs nodes in parallel [1][3]. Unknown ref [9].")
    prices = {"test-medium": ModelPrice(input=1.0, output=2.0), "test-embedding": ModelPrice(input=0.1)}
    monkeypatch.setattr("app.llm.usage.get_prices", lambda: prices)
    monkeypatch.setattr("app.graph.nodes.generate.get_llm", lambda tier: llm)
    monkeypatch.setattr("app.retrieval.dense.embed_query", lambda q: [0.0] * 8)
    monkeypatch.setattr("app.retrieval.dense.dense_search", lambda v, k: CHUNKS[:k])
    return llm


def test_pipeline_v0_loads():
    v0 = load_pipeline("v0")
    assert v0.retrieval_mode == "dense" and not v0.rerank and not v0.router and not v0.cache
    assert all(isinstance(p, PipelineConfig) for p in load_pipelines().values())
    with pytest.raises(KeyError):
        load_pipeline("nao-existe")


def test_extract_citations_ignores_invalid_and_duplicates():
    citations = extract_citations("a [2] b [2][1] c [7]", CHUNKS)
    assert [c.n for c in citations] == [2, 1]
    assert citations[0].chunk_id == "doc-a#chunk-1"


def test_context_is_numbered():
    assert format_context(CHUNKS).startswith("[1] A > S0\ntext 0")


def test_run_query_v0(fakes):
    from app.service import run_query

    result = run_query("How do I fan out?", "v0")
    assert result.answer.startswith("Send runs")
    assert [c.n for c in result.citations] == [1, 3]
    assert [u.node for u in result.usage] == ["retrieve", "generate"]
    assert result.total_cost_usd == pytest.approx(sum(u.cost_usd for u in result.usage))
    assert result.total_cost_usd > 0
    # o prompt leva o contexto numerado e a pergunta
    assert "[1] A > S0" in fakes.calls[0][1].content and "How do I fan out?" in fakes.calls[0][1].content


def test_query_route(fakes):
    from app.main import app

    client = TestClient(app)
    r = client.post("/query", json={"question": "How do I fan out?"})
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] and body["citations"][0]["url"] == "https://x/a#s0"
    assert client.post("/query", json={"question": "x", "pipeline": "nope"}).status_code == 404
