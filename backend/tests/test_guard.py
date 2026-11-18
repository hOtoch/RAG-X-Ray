"""Rate limit por IP, teto diário e feedback."""

from fastapi.testclient import TestClient

from app import guard
from app.config import get_settings
from tests.test_graph_v0 import fakes  # noqa: F401  (fixture)


def test_rate_limit_blocks_after_limit(fakes, monkeypatch):  # noqa: F811
    from app.main import app

    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    client = TestClient(app)
    codes = [client.post("/query", json={"question": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    # outro IP (atrás de proxy) tem a própria janela
    other = client.post("/query", json={"question": "q"}, headers={"x-forwarded-for": "9.9.9.9"})
    assert other.status_code == 200


def test_daily_budget_blocks_live_mode(fakes, monkeypatch):  # noqa: F811
    from app.main import app

    monkeypatch.setattr(get_settings(), "daily_budget_usd", 0.001)
    client = TestClient(app)
    guard.record_spend(0.002)
    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 429 and "replay" in response.json()["detail"]
    assert client.get("/budget").json()["spent_today_usd"] >= 0.002


def test_spend_is_recorded_after_each_query(fakes):  # noqa: F811
    from app.main import app

    before = guard.spent_today_usd()
    TestClient(app).post("/query", json={"question": "q"})
    assert guard.spent_today_usd() > before


def test_live_mode_switch(fakes, monkeypatch):  # noqa: F811
    from app.main import app

    monkeypatch.setattr(get_settings(), "live_mode_enabled", False)
    assert TestClient(app).post("/query/stream", json={"question": "q"}).status_code == 503


def test_feedback_requires_known_request(fakes, monkeypatch):  # noqa: F811
    from app.main import app

    scores = []
    monkeypatch.setattr("app.tracing.create_score", lambda rid, name, value, comment=None: scores.append((name, value)))
    client = TestClient(app)
    assert client.post("/feedback", json={"request_id": "nope", "value": 1}).status_code == 404
    rid = client.post("/query", json={"question": "q"}).json()["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "value": -1}).status_code == 200
    assert ("user_feedback", -1.0) in scores
