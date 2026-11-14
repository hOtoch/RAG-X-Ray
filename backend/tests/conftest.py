"""Ambiente fixo para os testes: nenhum teste lê o .env real nem chama API externa."""

import os

TEST_ENV = {
    "LLM_PROVIDER": "openai",
    "LLM_MODEL_SMALL": "test-small",
    "LLM_MODEL_MEDIUM": "test-medium",
    "LLM_MODEL_LARGE": "test-large",
    "EMBEDDING_PROVIDER": "openai",
    "EMBEDDING_MODEL": "test-embedding",
    "EMBEDDING_DIM": "8",
    "OPENAI_API_KEY": "test-key",
    "RAGXRAY_ENV_FILE": "nao-existe.env",
    "LANGFUSE_ENABLED": "false",
}
os.environ.update(TEST_ENV)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def memory_store(monkeypatch):
    """Troca a gravação de eventos no Postgres por um dicionário em memória."""
    saved: dict[str, list[dict]] = {}

    def save_events(request_id, question, pipeline, events):
        saved[request_id] = [e.model_dump(mode="json") for e in events]

    monkeypatch.setattr("app.store.save_events", save_events)
    monkeypatch.setattr("app.store.load_events", lambda request_id: saved.get(request_id, []))
    return saved
