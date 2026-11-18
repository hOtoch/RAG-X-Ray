"""Atributos GenAI do OpenTelemetry gravados em cada nó."""

from app.tracing import genai_attributes


def test_chat_node_uses_genai_conventions():
    attrs = genai_attributes(
        {
            "model": "test-medium",
            "tokens_in": 1200,
            "tokens_out": 80,
            "tokens_cached": 1024,
            "cost_usd": 0.0012,
            "latency_ms": 812.34,
        }  # fmt: skip
    )
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "test-medium"
    assert attrs["gen_ai.usage.input_tokens"] == 1200
    assert attrs["gen_ai.usage.output_tokens"] == 80
    assert attrs["ragxray.tokens_cached"] == 1024
    assert attrs["ragxray.latency_ms"] == 812.3


def test_embedding_call_is_an_embeddings_operation():
    attrs = genai_attributes({"model": "test-embedding", "tokens_in": 11, "latency_ms": 5})
    assert attrs["gen_ai.operation.name"] == "embeddings"
    assert "gen_ai.usage.output_tokens" not in attrs


def test_node_without_model_has_only_project_attributes():
    attrs = genai_attributes({"model": None, "latency_ms": 40.0})
    assert not any(k.startswith("gen_ai.") for k in attrs)
    assert attrs["ragxray.cost_usd"] == 0.0
