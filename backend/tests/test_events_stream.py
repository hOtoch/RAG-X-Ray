"""Contrato de eventos e streaming SSE, com LLM e banco substituídos por dublês."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.events.schema import DATA_PAYLOADS, Event, EventFactory, json_schema
from tests.test_graph_v0 import fakes  # noqa: F401  (fixture)


def test_factory_seq_is_monotonic_and_payload_validated():
    factory = EventFactory("r1")
    a = factory.make("node_start", "retrieve")
    b = factory.make("node_end", "retrieve", {"latency_ms": 3.0, "model": "m"})
    assert (a.seq, b.seq) == (0, 1)
    assert b.payload["tokens_in"] == 0 and b.payload["cost_usd"] == 0.0
    with pytest.raises(ValidationError):
        factory.make("node_end", "retrieve", {"model": "m"})  # falta latency_ms
    with pytest.raises(ValidationError):
        factory.make("data", "router", {"route": "nope", "reason": "", "confidence": 1})


def test_event_roundtrip_and_sse_format():
    event = EventFactory("r1").make("token", "generate", {"text": "olá"})
    assert Event.model_validate_json(event.model_dump_json()) == event
    lines = event.sse().split("\n")
    assert lines[0] == "event: token" and lines[1].startswith("data: {") and event.sse().endswith("\n\n")


def test_json_schema_covers_all_payloads():
    schema = json_schema()
    assert set(schema["data_payloads"]) == set(DATA_PAYLOADS)
    assert "seq" in schema["event"]["properties"]


def parse_sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.split("\n") if line.startswith("data: ")]


def test_query_stream_emits_events_in_order(fakes, memory_store):  # noqa: F811
    from app.main import app

    client = TestClient(app)
    response = client.post("/query/stream", json={"question": "How do I fan out?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)

    assert [e["seq"] for e in events] == list(range(len(events)))
    assert len({e["request_id"] for e in events}) == 1
    flow = [(e["type"], e["node"]) for e in events if e["type"] != "token"]
    assert flow == [
        ("node_start", "retrieve"),
        ("data", "retrieve"),
        ("node_end", "retrieve"),
        ("node_start", "generate"),
        ("node_end", "generate"),
        ("done", None),
    ]
    retrieve_data = next(e for e in events if e["type"] == "data")
    assert retrieve_data["payload"]["chunks"][0]["dense_rank"] == 1
    end = next(e for e in events if e["type"] == "node_end" and e["node"] == "generate")
    assert end["payload"]["tokens_in"] == 50 and end["payload"]["model"] == "test-medium"
    done = events[-1]["payload"]
    assert done["answer"].startswith("Send runs") and [c["n"] for c in done["citations"]] == [1, 3]

    # o stream foi gravado e pode ser lido de volta, idêntico
    request_id = events[0]["request_id"]
    assert response.headers["x-request-id"] == request_id
    assert client.get(f"/traces/{request_id}").json() == events
    assert client.get("/traces/nao-existe").status_code == 404


def test_error_becomes_error_event(fakes, monkeypatch):  # noqa: F811
    from app.main import app

    def boom(vector, k):
        raise RuntimeError("banco fora do ar")

    monkeypatch.setattr("app.retrieval.dense.dense_search", boom)
    events = parse_sse(TestClient(app).post("/query/stream", json={"question": "x"}).text)
    assert events[-1]["type"] == "error" and "banco fora do ar" in events[-1]["payload"]["message"]
