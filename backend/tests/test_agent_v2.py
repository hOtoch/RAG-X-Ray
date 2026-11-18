"""Grafo completo (router, multi-hop, groundedness, escalada) com LLM e retrieval substituídos por dublês."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.config import ModelPrice
from app.graph.build import build_graph
from app.graph.nodes.direct import detect_language
from app.graph.nodes.groundedness import GroundednessVerdict
from app.graph.nodes.multihop import DocsGrade, SubQueries
from app.graph.nodes.router import RouteDecision
from app.graph.state import ScoredChunk
from app.pipeline import PipelineConfig

USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


def chunk(cid: str) -> ScoredChunk:
    return ScoredChunk(
        id=cid, doc_id=cid.split("#")[0], title="T", section="S", url="u", text=f"text {cid}", token_count=5
    )


class Script:
    """Respostas programadas por tipo de chamada, e registro das chamadas feitas."""

    def __init__(self, route="simple_rag", grounded=(True,), sufficient=(True,), answer="Answer [1]."):
        self.route = route
        self.grounded = list(grounded)
        self.sufficient = list(sufficient)
        self.answer = answer
        self.calls: list[tuple[str, str]] = []  # (tipo, nível)

    def structured(self, schema, tier):
        self.calls.append((schema.__name__, tier))
        if schema is RouteDecision:
            return RouteDecision(route=self.route, reason="test", confidence=0.9)
        if schema is SubQueries:
            return SubQueries(queries=["part one", "part two"])
        if schema is DocsGrade:
            ok = self.sufficient.pop(0) if self.sufficient else True
            return DocsGrade(sufficient=ok, reason="r", rewritten_query="" if ok else "better query")
        if schema is GroundednessVerdict:
            ok = self.grounded.pop(0) if self.grounded else True
            return GroundednessVerdict(grounded=ok, unsupported_claims=[] if ok else ["made up"])
        raise AssertionError(schema)


class FakeLLM:
    def __init__(self, script: Script, tier: str):
        self.script, self.tier = script, tier

    def with_structured_output(self, schema, include_raw=True):
        script, tier = self.script, self.tier

        class Runner:
            def invoke(self, messages):
                return {"raw": AIMessage(content="{}", usage_metadata=USAGE), "parsed": script.structured(schema, tier)}

        return Runner()

    def invoke(self, messages, config=None):
        self.script.calls.append(("generate", self.tier))
        return AIMessage(content=self.script.answer, usage_metadata=USAGE)


@pytest.fixture
def agent(monkeypatch):
    searches: list[str] = []

    def setup(script: Script, **overrides):
        prices = {
            "test-small": ModelPrice(input=0.1, output=0.4),
            "test-medium": ModelPrice(input=1.0, output=4.0),
            "test-large": ModelPrice(input=10.0, output=40.0),
            "test-embedding": ModelPrice(input=0.01),
        }
        monkeypatch.setattr("app.llm.usage.get_prices", lambda: prices)
        fake = lambda tier: FakeLLM(script, tier)  # noqa: E731
        for module in ("app.llm.structured", "app.graph.nodes.generate", "app.graph.nodes.direct"):
            monkeypatch.setattr(f"{module}.get_llm", fake)

        def fake_search(query, pipeline, vector=None):
            searches.append(query)
            return [chunk(f"{query.replace(' ', '-')}#chunk-{i}") for i in range(3)]

        for module in ("app.graph.nodes.retrieve", "app.graph.nodes.multihop"):
            monkeypatch.setattr(f"{module}.search", fake_search)
        monkeypatch.setattr("app.retrieval.dense.embed_query", lambda q: [0.0] * 8)
        monkeypatch.setattr(
            "app.retrieval.reranker.rerank",
            lambda q, chunks, top_k, model=None: [
                c.model_copy(update={"rerank_rank": n}) for n, c in enumerate(chunks[:top_k], 1)
            ],
        )
        config = dict(name="v2t", retrieval_mode="hybrid", rerank=True, compress=True, router=True, groundedness=True,
                      route_tiers={"simple_rag": "medium", "multi_hop": "large"})  # fmt: skip
        pipeline = PipelineConfig(**(config | overrides))
        monkeypatch.setattr("app.service.get_graph", lambda name: build_graph(pipeline))
        monkeypatch.setattr("app.service.load_pipeline", lambda name: pipeline)

        from app.service import QueryRun

        run = QueryRun("How do I fan out?", "v2t")
        events = list(run.events())
        nodes = [e.node for e in events if e.type == "node_start"]
        return run.result, nodes, events

    setup.searches = searches
    return setup


def test_trivial_route_skips_retrieval(agent):
    result, nodes, _ = agent(Script(route="trivial", answer="Hi! Ask me about LangGraph."))
    assert nodes == ["router", "direct_answer"]
    assert result.route == "trivial" and result.model_tier_used == "small" and result.citations == []


def test_out_of_scope_refuses_without_generation(agent):
    script = Script(route="out_of_scope")
    result, nodes, _ = agent(script)
    assert nodes == ["router", "refuse"]
    assert "LangChain" in result.answer
    assert [c[0] for c in script.calls] == ["RouteDecision"]  # nenhuma chamada além do router


def test_simple_rag_path_uses_medium_and_checks_groundedness(agent):
    result, nodes, _ = agent(Script(route="simple_rag"))
    assert nodes == ["router", "retrieve", "rerank", "compress_context", "generate", "groundedness_check"]
    assert result.model_tier_used == "medium" and result.grounded is True and not result.gave_up


def test_multi_hop_fans_out_in_parallel_and_uses_large(agent):
    result, nodes, events = agent(Script(route="multi_hop"))
    assert nodes.count("retrieve_sub") == 3  # pergunta original + 2 sub-perguntas
    assert nodes[:2] == ["router", "decompose"]
    assert nodes[-5:] == ["rerank", "grade_docs", "compress_context", "generate", "groundedness_check"]
    assert set(agent.searches) == {"How do I fan out?", "part one", "part two"}
    assert result.model_tier_used == "large"
    assert len(result.context) <= 8


def test_corrective_retry_is_bounded_by_max_retries(agent):
    # docs nunca suficientes: tenta de novo até MAX_RETRIES (2 no ambiente de teste) e depois gera mesmo assim
    result, nodes, _ = agent(Script(route="multi_hop", sufficient=[False] * 10))
    assert nodes.count("retry_retrieve") == 2 and nodes.count("grade_docs") == 3
    assert result.retries == 2 and nodes[-2:] == ["generate", "groundedness_check"]


def test_groundedness_failure_escalates_once_then_succeeds(agent):
    script = Script(route="simple_rag", grounded=[False, True])
    result, nodes, _ = agent(script)
    assert nodes[-4:] == ["groundedness_check", "escalate", "generate", "groundedness_check"]
    assert [tier for kind, tier in script.calls if kind == "generate"] == ["medium", "large"]
    assert result.model_tier_used == "large" and result.grounded and result.escalations == 1


def test_groundedness_failing_twice_gives_up(agent):
    result, nodes, _ = agent(Script(route="simple_rag", grounded=[False, False]))
    assert nodes[-2:] == ["groundedness_check", "give_up"]
    assert result.gave_up and result.citations == [] and "documentation" in result.answer


def test_large_route_cannot_escalate(agent):
    result, nodes, _ = agent(Script(route="multi_hop", grounded=[False]))
    assert "escalate" not in nodes and result.gave_up


def test_budget_blocks_escalation_and_retries(agent):
    result, nodes, _ = agent(
        Script(route="multi_hop", sufficient=[False] * 5, grounded=[False]), request_budget_usd=0.0
    )
    assert "retry_retrieve" not in nodes and "escalate" not in nodes and result.gave_up


def test_counterfactual_cost_is_higher_than_cascade(agent):
    result, _, events = agent(Script(route="simple_rag"))
    done = events[-1].payload
    assert done["counterfactual_cost_usd"] == pytest.approx(result.counterfactual_cost_usd)
    assert result.counterfactual_cost_usd > result.total_cost_usd > 0


def test_detect_language():
    assert detect_language("Qual é a capital da França?") == "pt"
    assert detect_language("como faço pra rodar o grafo") == "pt"
    assert detect_language("What is the capital of France?") == "en"


def test_router_pipeline_without_rerank_selects_top_k(agent):
    result, nodes, _ = agent(Script(route="simple_rag"), rerank=False)
    assert nodes == ["router", "retrieve", "select_context", "compress_context", "generate", "groundedness_check"]
    assert len(result.context) == 3
