from fastapi.testclient import TestClient

from app.main import app


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_graph_topology_and_pipelines():
    client = TestClient(app)
    names = [p["name"] for p in client.get("/pipelines").json()]
    assert "v0" in names
    topology = client.get("/graph/v0").json()
    ids = [n["id"] for n in topology["nodes"]]
    assert ids == ["__start__", "retrieve", "generate", "__end__"]
    assert {"source": "retrieve", "target": "generate", "conditional": False, "label": None} in topology["edges"]
    assert client.get("/graph/nao-existe").status_code == 404
